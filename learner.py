# coding=utf-8
from dynet import *
import dynet
from utils import read_conll, read_conll_predict, write_conll, load_embeddings_file
from operator import itemgetter
import utils, time, random, decoder
import numpy as np
from mnnl import FFSequencePredictor, Layer, RNNSequencePredictor, BiRNNSequencePredictor

class jPosDepLearner:
    def __init__(self, vocab, pos, rels, w2i, c2i, options):
        self.model = ParameterCollection()
        random.seed(1)
        self.trainer = AdamTrainer(self.model)
        #if options.learning_rate is not None:
        #    self.trainer = AdamTrainer(self.model, alpha=options.learning_rate)
        #    print("Adam initial learning rate:", options.learning_rate)
        self.activations = {'tanh': tanh, 'sigmoid': logistic, 'relu': rectify,
                            'tanh3': (lambda x: tanh(cwise_multiply(cwise_multiply(x, x), x)))}
        self.activation = self.activations[options.activation]

        self.blstmFlag = options.blstmFlag
        self.labelsFlag = options.labelsFlag
        self.costaugFlag = options.costaugFlag
        self.bibiFlag = options.bibiFlag

        self.ldims = options.lstm_dims
        self.wdims = options.wembedding_dims
        self.cdims = options.cembedding_dims
        self.layers = options.lstm_layers
        self.wordsCount = vocab
        self.vocab = {word: ind + 3 for word, ind in w2i.iteritems()}
        self.pos = {word: ind for ind, word in enumerate(pos)}
        self.id2pos = {ind: word for ind, word in enumerate(pos)}
        self.c2i = c2i
        self.rels = {word: ind for ind, word in enumerate(rels)}
        self.irels = rels
        self.pdims = options.pembedding_dims

        self.vocab['*PAD*'] = 1
        self.vocab['*INITIAL*'] = 2
        self.wlookup = self.model.add_lookup_parameters((len(vocab) + 3, self.wdims))
        self.clookup = self.model.add_lookup_parameters((len(c2i), self.cdims))
        self.plookup = self.model.add_lookup_parameters((len(pos), self.pdims))

        if options.external_embedding is not None:
            ext_embeddings, ext_emb_dim = load_embeddings_file(options.external_embedding, lower=True)
            assert (ext_emb_dim == self.wdims)
            print("Initializing word embeddings by pre-trained vectors")
            count = 0
            for word in self.vocab:
                _word = unicode(word, "utf-8")
                if _word in ext_embeddings:
                    count += 1
                    self.wlookup.init_row(self.vocab[word], ext_embeddings[_word])
            print("Vocab size: %d; #words having pretrained vectors: %d" % (len(self.vocab), count))

        self.pos_builders = [VanillaLSTMBuilder(1, self.wdims + self.cdims * 2, self.ldims, self.model),
                             VanillaLSTMBuilder(1, self.wdims + self.cdims * 2, self.ldims, self.model)]
        self.pos_bbuilders = [VanillaLSTMBuilder(1, self.ldims * 2, self.ldims, self.model),
                              VanillaLSTMBuilder(1, self.ldims * 2, self.ldims, self.model)]

        if self.bibiFlag:
            self.builders = [VanillaLSTMBuilder(1, self.wdims + self.cdims * 2 + self.pdims, self.ldims, self.model),
                             VanillaLSTMBuilder(1, self.wdims + self.cdims * 2 + self.pdims, self.ldims, self.model)]
            self.bbuilders = [VanillaLSTMBuilder(1, self.ldims * 2, self.ldims, self.model),
                              VanillaLSTMBuilder(1, self.ldims * 2, self.ldims, self.model)]
        elif self.layers > 0:
            self.builders = [VanillaLSTMBuilder(self.layers, self.wdims + self.cdims * 2 + self.pdims, self.ldims, self.model),
                             VanillaLSTMBuilder(self.layers, self.wdims + self.cdims * 2 + self.pdims, self.ldims, self.model)]
        else:
            self.builders = [SimpleRNNBuilder(1, self.wdims + self.cdims * 2, self.ldims, self.model),
                             SimpleRNNBuilder(1, self.wdims + self.cdims * 2, self.ldims, self.model)]

        self.ffSeqPredictor = FFSequencePredictor(Layer(self.model, self.ldims * 2, len(self.pos), softmax))

        self.hidden_units = options.hidden_units
        
        self.hidBias = self.model.add_parameters((self.ldims * 8))
        self.hidLayer = self.model.add_parameters((self.hidden_units, self.ldims * 8))
        self.hid2Bias = self.model.add_parameters((self.hidden_units))

        self.outLayer = self.model.add_parameters((1, self.hidden_units if self.hidden_units > 0 else self.ldims * 8))

        if self.labelsFlag:
            self.rhidBias = self.model.add_parameters((self.ldims * 8))
            self.rhidLayer = self.model.add_parameters((self.hidden_units, self.ldims * 8))
            self.rhid2Bias = self.model.add_parameters((self.hidden_units))
            self.routLayer = self.model.add_parameters(
                (len(self.irels), self.hidden_units if self.hidden_units > 0 else self.ldims * 8))
            self.routBias = self.model.add_parameters((len(self.irels)))
            self.ffRelPredictor = FFSequencePredictor(
                Layer(self.model, self.hidden_units if self.hidden_units > 0 else self.ldims * 8, len(self.irels),
                      softmax))

        self.char_rnn = RNNSequencePredictor(LSTMBuilder(1, self.cdims, self.cdims, self.model))

    def __getExpr(self, sentence, i, j):

        if sentence[i].headfov is None:
            sentence[i].headfov = concatenate([sentence[i].lstms[0], sentence[i].lstms[1]])
        if sentence[j].modfov is None:
            sentence[j].modfov = concatenate([sentence[j].lstms[0], sentence[j].lstms[1]])

        _inputVector = concatenate(
            [sentence[i].headfov, sentence[j].modfov, dynet.abs(sentence[i].headfov - sentence[j].modfov),
             dynet.cmult(sentence[i].headfov, sentence[j].modfov)])

        if self.hidden_units > 0:
            output = self.outLayer.expr() * self.activation(
                self.hid2Bias.expr() + self.hidLayer.expr() * self.activation(
                    _inputVector + self.hidBias.expr()))
        else:
            output = self.outLayer.expr() * self.activation(_inputVector + self.hidBias.expr())

        return output

    def __evaluate(self, sentence):
        exprs = [[self.__getExpr(sentence, i, j) for j in xrange(len(sentence))] for i in xrange(len(sentence))]
        scores = np.array([[output.scalar_value() for output in exprsRow] for exprsRow in exprs])

        return scores, exprs

    def pick_neg_log(self, pred, gold):
        return -dynet.log(dynet.pick(pred, gold))

    def __getRelVector(self, sentence, i, j):
        if sentence[i].rheadfov is None:
            sentence[i].rheadfov = concatenate([sentence[i].lstms[0], sentence[i].lstms[1]])
        if sentence[j].rmodfov is None:
            sentence[j].rmodfov = concatenate([sentence[j].lstms[0], sentence[j].lstms[1]])
        _outputVector = concatenate(
            [sentence[i].rheadfov, sentence[j].rmodfov, abs(sentence[i].rheadfov - sentence[j].rmodfov),
             cmult(sentence[i].rheadfov, sentence[j].rmodfov)])

        if self.hidden_units > 0:
            return self.rhid2Bias.expr() + self.rhidLayer.expr() * self.activation(
                _outputVector + self.rhidBias.expr())
        else:
            return _outputVector

    def Save(self, filename):
        self.model.save(filename)

    def Load(self, filename):
        self.model.populate(filename)

    def Predict(self, conll_path):
        with open(conll_path, 'r') as conllFP:
            for iSentence, sentence in enumerate(read_conll_predict(conllFP, self.c2i, self.wordsCount)):
                conll_sentence = [entry for entry in sentence if isinstance(entry, utils.ConllEntry)]

                for entry in conll_sentence:
                    wordvec = self.wlookup[int(self.vocab.get(entry.norm, 0))] if self.wdims > 0 else None

                    last_state = self.char_rnn.predict_sequence([self.clookup[c] for c in entry.idChars])[-1]
                    rev_last_state = self.char_rnn.predict_sequence([self.clookup[c] for c in reversed(entry.idChars)])[
                        -1]

                    entry.vec = concatenate(filter(None, [wordvec, last_state, rev_last_state]))

                    entry.pos_lstms = [entry.vec, entry.vec]
                    entry.headfov = None
                    entry.modfov = None

                    entry.rheadfov = None
                    entry.rmodfov = None

                #Predicted pos tags
                lstm_forward = self.pos_builders[0].initial_state()
                lstm_backward = self.pos_builders[1].initial_state()
                for entry, rentry in zip(conll_sentence, reversed(conll_sentence)):
                    lstm_forward = lstm_forward.add_input(entry.vec)
                    lstm_backward = lstm_backward.add_input(rentry.vec)

                    entry.pos_lstms[1] = lstm_forward.output()
                    rentry.pos_lstms[0] = lstm_backward.output()

                for entry in conll_sentence:
                    entry.pos_vec = concatenate(entry.pos_lstms)

                blstm_forward = self.pos_bbuilders[0].initial_state()
                blstm_backward = self.pos_bbuilders[1].initial_state()

                for entry, rentry in zip(conll_sentence, reversed(conll_sentence)):
                    blstm_forward = blstm_forward.add_input(entry.pos_vec)
                    blstm_backward = blstm_backward.add_input(rentry.pos_vec)
                    entry.pos_lstms[1] = blstm_forward.output()
                    rentry.pos_lstms[0] = blstm_backward.output()

                concat_layer = [concatenate(entry.pos_lstms) for entry in conll_sentence]
                outputFFlayer = self.ffSeqPredictor.predict_sequence(concat_layer)
                predicted_pos_indices = [np.argmax(o.value()) for o in outputFFlayer]
                predicted_postags = [self.id2pos[idx] for idx in predicted_pos_indices]

                # Add predicted pos tags for parsing prediction
                for entry, posid in zip(conll_sentence, predicted_pos_indices):
                    entry.vec = concatenate([entry.vec, self.plookup[posid]])
                    entry.lstms = [entry.vec, entry.vec]

                if self.blstmFlag:
                    lstm_forward = self.builders[0].initial_state()
                    lstm_backward = self.builders[1].initial_state()

                    for entry, rentry in zip(conll_sentence, reversed(conll_sentence)):
                        lstm_forward = lstm_forward.add_input(entry.vec)
                        lstm_backward = lstm_backward.add_input(rentry.vec)

                        entry.lstms[1] = lstm_forward.output()
                        rentry.lstms[0] = lstm_backward.output()

                    if self.bibiFlag:
                        for entry in conll_sentence:
                            entry.vec = concatenate(entry.lstms)

                        blstm_forward = self.bbuilders[0].initial_state()
                        blstm_backward = self.bbuilders[1].initial_state()

                        for entry, rentry in zip(conll_sentence, reversed(conll_sentence)):
                            blstm_forward = blstm_forward.add_input(entry.vec)
                            blstm_backward = blstm_backward.add_input(rentry.vec)

                            entry.lstms[1] = blstm_forward.output()
                            rentry.lstms[0] = blstm_backward.output()

                scores, exprs = self.__evaluate(conll_sentence)
                heads = decoder.parse_proj(scores)

                # Multiple roots: heading to the previous "rooted" one
                rootCount = 0
                rootWid = -1
                for index, head in enumerate(heads):
                    if head == 0:
                        rootCount += 1
                        if rootCount == 1:
                            rootWid = index
                        if rootCount > 1:
                            heads[index] = rootWid
                            rootWid = index

                for entry, head, pos in zip(conll_sentence, heads, predicted_postags):
                    entry.pred_parent_id = head
                    entry.pred_relation = '_'
                    entry.pred_pos = pos

                dump = False

                if self.labelsFlag:
                    concat_layer = [self.__getRelVector(conll_sentence, head, modifier + 1) for modifier, head in
                                    enumerate(heads[1:])]
                    outputFFlayer = self.ffRelPredictor.predict_sequence(concat_layer)
                    predicted_rel_indices = [np.argmax(o.value()) for o in outputFFlayer]
                    predicted_rels = [self.irels[idx] for idx in predicted_rel_indices]
                    for modifier, head in enumerate(heads[1:]):
                        conll_sentence[modifier + 1].pred_relation = predicted_rels[modifier]

                renew_cg()
                if not dump:
                    yield sentence

    def Train(self, conll_path):
        eloss = 0.0
        mloss = 0.0
        eerrors = 0
        etotal = 0
        start = time.time()

        with open(conll_path, 'r') as conllFP:
            shuffledData = list(read_conll(conllFP, self.c2i))
            random.shuffle(shuffledData)

            errs = []
            lerrs = []
            posErrs = []

            for iSentence, sentence in enumerate(shuffledData):
                if iSentence % 500 == 0 and iSentence != 0:
                    print "Processing sentence number: %d" % iSentence, ", Loss: %.4f" % (
                                eloss / etotal), ", Time: %.2f" % (time.time() - start)
                    start = time.time()
                    eerrors = 0
                    eloss = 0.0
                    etotal = 0

                conll_sentence = [entry for entry in sentence if isinstance(entry, utils.ConllEntry)]

                for entry in conll_sentence:
                    c = float(self.wordsCount.get(entry.norm, 0))
                    dropFlag = (random.random() < (c / (0.25 + c)))
                    wordvec = self.wlookup[
                        int(self.vocab.get(entry.norm, 0)) if dropFlag else 0] if self.wdims > 0 else None

                    last_state = self.char_rnn.predict_sequence([self.clookup[c] for c in entry.idChars])[-1]
                    rev_last_state = self.char_rnn.predict_sequence([self.clookup[c] for c in reversed(entry.idChars)])[
                        -1]

                    entry.vec = dynet.dropout(concatenate(filter(None, [wordvec, last_state, rev_last_state])), 0.33)

                    entry.pos_lstms = [entry.vec, entry.vec]
                    entry.headfov = None
                    entry.modfov = None

                    entry.rheadfov = None
                    entry.rmodfov = None

                #POS tagging loss
                lstm_forward = self.pos_builders[0].initial_state()
                lstm_backward = self.pos_builders[1].initial_state()
                for entry, rentry in zip(conll_sentence, reversed(conll_sentence)):
                    lstm_forward = lstm_forward.add_input(entry.vec)
                    lstm_backward = lstm_backward.add_input(rentry.vec)

                    entry.pos_lstms[1] = lstm_forward.output()
                    rentry.pos_lstms[0] = lstm_backward.output()

                for entry in conll_sentence:
                    entry.pos_vec = concatenate(entry.pos_lstms)

                blstm_forward = self.pos_bbuilders[0].initial_state()
                blstm_backward = self.pos_bbuilders[1].initial_state()

                for entry, rentry in zip(conll_sentence, reversed(conll_sentence)):
                    blstm_forward = blstm_forward.add_input(entry.pos_vec)
                    blstm_backward = blstm_backward.add_input(rentry.pos_vec)
                    entry.pos_lstms[1] = blstm_forward.output()
                    rentry.pos_lstms[0] = blstm_backward.output()

                concat_layer = [dynet.dropout(concatenate(entry.pos_lstms), 0.33) for entry in conll_sentence]
                outputFFlayer = self.ffSeqPredictor.predict_sequence(concat_layer)
                posIDs = [self.pos.get(entry.pos) for entry in conll_sentence]
                for pred, gold in zip(outputFFlayer, posIDs):
                    posErrs.append(self.pick_neg_log(pred, gold))

                # Add predicted pos tags
                for entry, poses in zip(conll_sentence, outputFFlayer):
                    entry.vec = concatenate([entry.vec, dynet.dropout(self.plookup[np.argmax(poses.value())], 0.33)])
                    entry.lstms = [entry.vec, entry.vec]

                #Parsing losses
                if self.blstmFlag:
                    lstm_forward = self.builders[0].initial_state()
                    lstm_backward = self.builders[1].initial_state()

                    for entry, rentry in zip(conll_sentence, reversed(conll_sentence)):
                        lstm_forward = lstm_forward.add_input(entry.vec)
                        lstm_backward = lstm_backward.add_input(rentry.vec)

                        entry.lstms[1] = lstm_forward.output()
                        rentry.lstms[0] = lstm_backward.output()

                    if self.bibiFlag:
                        for entry in conll_sentence:
                            entry.vec = concatenate(entry.lstms)

                        blstm_forward = self.bbuilders[0].initial_state()
                        blstm_backward = self.bbuilders[1].initial_state()

                        for entry, rentry in zip(conll_sentence, reversed(conll_sentence)):
                            blstm_forward = blstm_forward.add_input(entry.vec)
                            blstm_backward = blstm_backward.add_input(rentry.vec)

                            entry.lstms[1] = blstm_forward.output()
                            rentry.lstms[0] = blstm_backward.output()

                scores, exprs = self.__evaluate(conll_sentence)
                gold = [entry.parent_id for entry in conll_sentence]
                heads = decoder.parse_proj(scores, gold if self.costaugFlag else None)

                if self.labelsFlag:

                    concat_layer = [dynet.dropout(self.__getRelVector(conll_sentence, head, modifier + 1), 0.33) for
                                    modifier, head in enumerate(gold[1:])]
                    outputFFlayer = self.ffRelPredictor.predict_sequence(concat_layer)
                    relIDs = [self.rels[conll_sentence[modifier + 1].relation] for modifier, _ in enumerate(gold[1:])]
                    for pred, goldid in zip(outputFFlayer, relIDs):
                        lerrs.append(self.pick_neg_log(pred, goldid))

                e = sum([1 for h, g in zip(heads[1:], gold[1:]) if h != g])
                eerrors += e
                if e > 0:
                    loss = [(exprs[h][i] - exprs[g][i]) for i, (h, g) in enumerate(zip(heads, gold)) if h != g]  # * (1.0/float(e))
                    eloss += (e)
                    mloss += (e)
                    errs.extend(loss)

                etotal += len(conll_sentence)

                if iSentence % 1 == 0:
                    if len(errs) > 0 or len(lerrs) > 0 or len(posErrs) > 0:
                        eerrs = (esum(errs + lerrs + posErrs))
                        eerrs.scalar_value()
                        eerrs.backward()
                        self.trainer.update()
                        errs = []
                        lerrs = []
                        posErrs = []

                    renew_cg()

        print "Loss: %.4f" % (mloss / iSentence)

