# coding=utf-8
from dynet import *
import dynet
from utils import read_conll, write_conll, load_embeddings_file
from operator import itemgetter
import utils, time, random, decoder
import numpy as np
from mnnl import FFSequencePredictor, Layer, RNNSequencePredictor, BiRNNSequencePredictor


class jPosDepLearner:
    def __init__(self, vocab, pos, rels, w2i, c2i, options):
        self.model = ParameterCollection()
        random.seed(1)
        self.trainer = AdamTrainer(self.model)
        if options.learning_rate is not None:
            self.trainer = AdamTrainer(self.model, alpha=options.learning_rate)
            #print("Adam initial learning rate: ", options.learning_rate)
        self.activations = {'tanh': tanh, 'sigmoid': logistic, 'relu': rectify, 'tanh3': (lambda x: tanh(cwise_multiply(cwise_multiply(x, x), x)))}
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
        self.vocab = {word: ind+3 for word, ind in w2i.iteritems()}
        self.pos = {word: ind for ind, word in enumerate(pos)}
        self.id2pos = {ind: word for ind, word in enumerate(pos)}
        self.c2i = c2i
        self.rels = {word: ind for ind, word in enumerate(rels)}
        self.irels = rels

        self.vocab['*PAD*'] = 1
        self.vocab['*INITIAL*'] = 2
        self.wlookup = self.model.add_lookup_parameters((len(vocab) + 3, self.wdims))
        self.clookup = self.model.add_lookup_parameters((len(c2i), self.cdims))
        
        if options.external_embedding is not None:
            ext_embeddings, ext_emb_dim = load_embeddings_file(options.external_embedding, lower=True)
            assert(ext_emb_dim==self.wdims)
            for word in self.vocab.keys():
                if word in ext_embeddings.keys():
                    self.wlookup.init_row(self.vocab[word], ext_embeddings[word])


        if self.bibiFlag:
            self.builders = [VanillaLSTMBuilder(1, self.wdims + self.cdims * 2, self.ldims, self.model),
                             VanillaLSTMBuilder(1, self.wdims + self.cdims * 2, self.ldims, self.model)]
            self.bbuilders = [VanillaLSTMBuilder(1, self.ldims * 2, self.ldims, self.model),
                              VanillaLSTMBuilder(1, self.ldims * 2, self.ldims, self.model)]
        elif self.layers > 0:
            self.builders = [VanillaLSTMBuilder(self.layers, self.wdims, self.ldims, self.model),
                             VanillaLSTMBuilder(self.layers, self.wdims, self.ldims, self.model)]
        else:
            self.builders = [SimpleRNNBuilder(1, self.wdims + self.cdims * 2, self.ldims, self.model),
                             SimpleRNNBuilder(1, self.wdims + self.cdims * 2, self.ldims, self.model)]
            
        self.ffSeqPredictor = FFSequencePredictor(Layer(self.model, self.ldims*2, len(self.pos), softmax))    

        self.hidden_units = options.hidden_units
        self.hidden2_units = options.hidden2_units

        self.hidLayerFOH = self.model.add_parameters((self.hidden_units, self.ldims * 2))
        self.hidLayerFOM = self.model.add_parameters((self.hidden_units, self.ldims * 2))
        self.hidBias = self.model.add_parameters((self.hidden_units))

        self.hid2Layer = self.model.add_parameters((self.hidden2_units, self.hidden_units))
        self.hid2Bias = self.model.add_parameters((self.hidden2_units))

        self.outLayer = self.model.add_parameters((1, self.hidden2_units if self.hidden2_units > 0 else self.hidden_units))

        if self.labelsFlag:
            self.rhidLayerFOH = self.model.add_parameters((self.hidden_units, 2 * self.ldims))
            self.rhidLayerFOM = self.model.add_parameters((self.hidden_units, 2 * self.ldims))
            self.rhidBias = self.model.add_parameters((self.hidden_units))

            self.rhid2Layer = self.model.add_parameters((self.hidden2_units, self.hidden_units))
            self.rhid2Bias = self.model.add_parameters((self.hidden2_units))

            self.routLayer = self.model.add_parameters((len(self.irels), self.hidden2_units if self.hidden2_units > 0 else self.hidden_units))
            self.routBias = self.model.add_parameters((len(self.irels)))
        
        self.char_rnn = RNNSequencePredictor(LSTMBuilder(1, self.cdims, self.cdims, self.model))

    def  __getExpr(self, sentence, i, j, train):

        if sentence[i].headfov is None:
            sentence[i].headfov = self.hidLayerFOH.expr() * concatenate([sentence[i].lstms[0], sentence[i].lstms[1]])
        if sentence[j].modfov is None:
            sentence[j].modfov  = self.hidLayerFOM.expr() * concatenate([sentence[j].lstms[0], sentence[j].lstms[1]])

        if self.hidden2_units > 0:
            output = self.outLayer.expr() * self.activation(self.hid2Bias.expr() + self.hid2Layer.expr() * self.activation(sentence[i].headfov + sentence[j].modfov + self.hidBias.expr())) # + self.outBias
        else:
            output = self.outLayer.expr() * self.activation(sentence[i].headfov + sentence[j].modfov + self.hidBias.expr()) # + self.outBias

        return output


    def __evaluate(self, sentence, train):
        exprs = [ [self.__getExpr(sentence, i, j, train) for j in xrange(len(sentence))] for i in xrange(len(sentence)) ]
        scores = np.array([ [output.scalar_value() for output in exprsRow] for exprsRow in exprs ])

        return scores, exprs

    def pick_neg_log(self, pred, gold):
        return -dynet.log(dynet.pick(pred, gold))

    def __evaluateLabel(self, sentence, i, j):
        if sentence[i].rheadfov is None:
            sentence[i].rheadfov = self.rhidLayerFOH.expr() * concatenate([sentence[i].lstms[0], sentence[i].lstms[1]])
        if sentence[j].rmodfov is None:
            sentence[j].rmodfov  = self.rhidLayerFOM.expr() * concatenate([sentence[j].lstms[0], sentence[j].lstms[1]])

        if self.hidden2_units > 0:
            output = self.routLayer.expr() * self.activation(self.rhid2Bias.expr() + self.rhid2Layer.expr() * self.activation(sentence[i].rheadfov + sentence[j].rmodfov + self.rhidBias.expr())) + self.routBias.expr()
        else:
            output = self.routLayer.expr() * self.activation(sentence[i].rheadfov + sentence[j].rmodfov + self.rhidBias.expr()) + self.routBias.expr()

        return output.value(), output


    def Save(self, filename):
        self.model.save(filename)


    def Load(self, filename):
        self.model.populate(filename)


    def Predict(self, conll_path):
        with open(conll_path, 'r') as conllFP:
            for iSentence, sentence in enumerate(read_conll(conllFP, self.c2i)):
                conll_sentence = [entry for entry in sentence if isinstance(entry, utils.ConllEntry)]

                for entry in conll_sentence:
                    wordvec = self.wlookup[int(self.vocab.get(entry.norm, 0))] if self.wdims > 0 else None
                
                    last_state = self.char_rnn.predict_sequence([self.clookup[c] for c in entry.idChars])[-1]
                    rev_last_state = self.char_rnn.predict_sequence([self.clookup[c] for c in reversed(entry.idChars)])[-1]
                      
                    entry.vec = concatenate(filter(None, [wordvec, last_state, rev_last_state]))

                    entry.lstms = [entry.vec, entry.vec]
                    entry.headfov = None
                    entry.modfov = None

                    entry.rheadfov = None
                    entry.rmodfov = None

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

                scores, exprs = self.__evaluate(conll_sentence, True)
                heads = decoder.parse_proj(scores)
                
                #Multiple roots: heading to the previous "rooted" one
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
                        

                concat_layer = [concatenate(entry.lstms) for entry in conll_sentence]
                outputFFlayer = self.ffSeqPredictor.predict_sequence(concat_layer)
                predicted_pos_indices = [np.argmax(o.value()) for o in outputFFlayer]  
                predicted_postags = [self.id2pos[idx] for idx in predicted_pos_indices]
                
                
                for entry, head, pos in zip(conll_sentence, heads, predicted_postags):
                    entry.pred_parent_id = head
                    entry.pred_relation = '_'
                    entry.pred_pos = pos

                dump = False

                if self.labelsFlag:
                    for modifier, head in enumerate(heads[1:]):
                        scores, exprs = self.__evaluateLabel(conll_sentence, head, modifier+1)
                        conll_sentence[modifier+1].pred_relation = self.irels[max(enumerate(scores), key=itemgetter(1))[0]]

                renew_cg()
                if not dump:
                    yield sentence


    def Train(self, conll_path):
        errors = 0
        batch = 0
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
            eeloss = 0.0

            for iSentence, sentence in enumerate(shuffledData):
                if iSentence % 500 == 0 and iSentence != 0:
                    print "Processing sentence number: %d" % iSentence, ", Loss: %.2f" % (eloss / etotal), ", Time: %.2f" % (time.time()-start)
                    start = time.time()
                    eerrors = 0
                    eloss = 0.0
                    etotal = 0
                    lerrors = 0
                    ltotal = 0

                conll_sentence = [entry for entry in sentence if isinstance(entry, utils.ConllEntry)]

                for entry in conll_sentence:
                    c = float(self.wordsCount.get(entry.norm, 0))
                    dropFlag = (random.random() < (c/(0.25+c)))
                    wordvec = self.wlookup[int(self.vocab.get(entry.norm, 0)) if dropFlag else 0] if self.wdims > 0 else None
                    
                    last_state = self.char_rnn.predict_sequence([self.clookup[c] for c in entry.idChars])[-1]
                    rev_last_state = self.char_rnn.predict_sequence([self.clookup[c] for c in reversed(entry.idChars)])[-1]
                    
                    entry.vec = concatenate([dynet.noise(fe,0.2) for fe in filter(None, [wordvec, last_state, rev_last_state])])
                    
                    entry.lstms = [entry.vec, entry.vec]
                    entry.headfov = None
                    entry.modfov = None

                    entry.rheadfov = None
                    entry.rmodfov = None

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

                scores, exprs = self.__evaluate(conll_sentence, True)
                gold = [entry.parent_id for entry in conll_sentence]
                heads = decoder.parse_proj(scores, gold if self.costaugFlag else None)

                if self.labelsFlag:
                    for modifier, head in enumerate(gold[1:]):
                        rscores, rexprs = self.__evaluateLabel(conll_sentence, head, modifier+1)
                        goldLabelInd = self.rels[conll_sentence[modifier+1].relation]
                        wrongLabelInd = max(((l, scr) for l, scr in enumerate(rscores) if l != goldLabelInd), key=itemgetter(1))[0]
                        if rscores[goldLabelInd] < rscores[wrongLabelInd] + 1:
                            lerrs.append(rexprs[wrongLabelInd] - rexprs[goldLabelInd])

                e = sum([1 for h, g in zip(heads[1:], gold[1:]) if h != g])
                eerrors += e
                if e > 0:
                    loss = [(exprs[h][i] - exprs[g][i]) for i, (h,g) in enumerate(zip(heads, gold)) if h != g] # * (1.0/float(e))
                    eloss += (e)
                    mloss += (e)
                    errs.extend(loss)

                etotal += len(conll_sentence)
                
                
                concat_layer = [concatenate(entry.lstms) for entry in conll_sentence]
                concat_layer = [dynet.noise(fe,0.2) for fe in concat_layer]
                outputFFlayer = self.ffSeqPredictor.predict_sequence(concat_layer)
                posIDs  = [self.pos.get(entry.pos) for entry in conll_sentence ]
                for pred, gold in zip(outputFFlayer, posIDs):
                    posErrs.append(self.pick_neg_log(pred,gold))

                if iSentence % 1 == 0 or len(errs) > 0 or len(lerrs) > 0 or len(posErrs) > 0:
                    eeloss = 0.0

                    if len(errs) > 0 or len(lerrs) > 0 or len(posErrs) > 0:
                        eerrs = (esum(errs + lerrs + posErrs)) #* (1.0/(float(len(errs))))
                        eerrs.scalar_value()
                        eerrs.backward()
                        self.trainer.update()
                        errs = []
                        lerrs = []
                        posErrs = []
                        
                    renew_cg()

        if len(errs) > 0:
            eerrs = (esum(errs + lerrs + posErrs)) #* (1.0/(float(len(errs))))
            eerrs.scalar_value()
            eerrs.backward()
            self.trainer.update()

            errs = []
            lerrs = []
            posErrs = []
            eeloss = 0.0

            renew_cg()

        self.trainer.update()
        print "Loss: %.2f" % (mloss/iSentence)
