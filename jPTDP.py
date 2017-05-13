# coding=utf-8
from optparse import OptionParser
import pickle, utils, learner, os, os.path, time


if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option("--train", dest="conll_train", help="Path to annotated CONLL train file", metavar="FILE", default="N/A")
    parser.add_option("--dev", dest="conll_dev", help="Path to annotated CONLL dev file", metavar="FILE", default="N/A")
    parser.add_option("--test", dest="conll_test", help="Path to CONLL test file", metavar="FILE", default="N/A")
    parser.add_option("--output", dest="conll_test_output", help="File name for predicted output", metavar="FILE", default="N/A")
    parser.add_option("--extrn", dest="external_embedding", help="External embeddings", metavar="FILE")
    parser.add_option("--params", dest="params", help="Parameters file", metavar="FILE", default="params.pickle")
    parser.add_option("--model", dest="model", help="Load/Save model file", metavar="FILE", default="neuralfirstorder.model")
    parser.add_option("--wembedding", type="int", dest="wembedding_dims", default=128)
    parser.add_option("--cembedding", type="int", dest="cembedding_dims", default=64)
    parser.add_option("--epochs", type="int", dest="epochs", default=30)
    parser.add_option("--hidden", type="int", dest="hidden_units", default=100)
    parser.add_option("--hidden2", type="int", dest="hidden2_units", default=0)
    parser.add_option("--lr", type="float", dest="learning_rate", default=0.1)
    parser.add_option("--outdir", type="string", dest="output", default="results")
    parser.add_option("--activation", type="string", dest="activation", default="tanh")
    parser.add_option("--lstmlayers", type="int", dest="lstm_layers", default=2)
    parser.add_option("--lstmdims", type="int", dest="lstm_dims", default=128)
    parser.add_option("--disableblstm", action="store_false", dest="blstmFlag", default=True)
    parser.add_option("--disablelabels", action="store_false", dest="labelsFlag", default=True)
    parser.add_option("--predict", action="store_true", dest="predictFlag", default=False)
    parser.add_option("--bibi-lstm", action="store_false", dest="bibiFlag", default=True)
    parser.add_option("--disablecostaug", action="store_false", dest="costaugFlag", default=True)
    parser.add_option("--dynet-seed", type="int", dest="seed", default=0)
    parser.add_option("--dynet-mem", type="int", dest="mem", default=0)

    (options, args) = parser.parse_args()

    print 'Using external embedding:', options.external_embedding

    if options.predictFlag:
        with open(options.params, 'r') as paramsfp:
            words, w2i, c2i, pos, rels, stored_opt = pickle.load(paramsfp)
            
        stored_opt.external_embedding = options.external_embedding
        
        print 'Loading pre-trained joint model'
        parser = learner.jPosDepLearner(words, pos, rels, w2i, c2i, stored_opt)
        parser.Load(options.model)
        
        tespath = os.path.join(options.output, options.conll_test_output)
        print 'Predicting POS tags and parsing dependencies'
        ts = time.time()
        test_res = list(parser.Predict(options.conll_test))
        te = time.time()
        print 'Finished in', te-ts, 'seconds.'
        utils.write_conll(tespath, test_res)

        #conllu = (os.path.splitext(options.conll_test.lower())[1] == '.conllu')
        #if not conllu:#Scored with punctuation
        #    os.system('perl utils/eval07.pl -q -g ' + options.conll_test + ' -s ' + tespath  + ' > ' + tespath + '.scores.txt')
        #else:
        #    os.system('python utils/evaluation_script/conll17_ud_eval.py -v -w utils/evaluation_script/weights.clas ' + options.conll_test + ' ' + tespath + ' > ' + tespath + '.scores.txt')
    else:
        print 'Extracting vocabulary'
        words, w2i, c2i, pos, rels = utils.vocab(options.conll_train)
        
        with open(os.path.join(options.output, options.params), 'w') as paramsfp:
            pickle.dump((words, w2i, c2i, pos, rels, options), paramsfp)
        
        print 'Initializing joint model'
        parser = learner.jPosDepLearner(words, pos, rels, w2i, c2i, options)
        
        highestScore = 0.0
        eId = 0
        for epoch in xrange(options.epochs):
            print '\n-----------------\nStarting epoch', epoch + 1
            parser.Train(options.conll_train)
            
            if options.conll_dev == "N/A":  
                parser.Save(os.path.join(options.output, os.path.basename(options.model)))
                
            else: 
                devPredSents = parser.Predict(options.conll_dev)
                
                count = 0
                lasCount = 0
                posCount = 0
                poslasCount = 0
                for idSent, devSent in enumerate(devPredSents):
                    conll_devSent = [entry for entry in devSent if isinstance(entry, utils.ConllEntry)]
                    
                    for entry in conll_devSent:
                        if entry.id <= 0:
                            continue
                        if entry.pos == entry.pred_pos and entry.parent_id == entry.pred_parent_id and entry.pred_relation == entry.relation:
                            poslasCount += 1
                        if entry.pos == entry.pred_pos:
                            posCount += 1
                        if entry.parent_id == entry.pred_parent_id and entry.pred_relation == entry.relation:
                            lasCount += 1
                        count += 1
                        
                print "---\nLAS accuracy:\t%.2f" % (float(lasCount) * 100 / count)
                print "POS accuracy:\t%.2f" % (float(posCount) * 100 / count)
                print "POS&LAS:\t%.2f" % (float(poslasCount) * 100 / count)
                
                score = float(poslasCount) * 100 / count
                if score >= highestScore:
                    parser.Save(os.path.join(options.output, os.path.basename(options.model)))
                    highestScore = score
                    eId = epoch + 1
                
                print "Highest POS&LAS: %.2f at epoch %d" % (highestScore, eId)