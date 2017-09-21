# A Novel Neural Network Model for Joint POS Tagging and Graph-based Dependency Parsing 

The implementation of the jPTDP model, as described in my paper:

    @InProceedings{nguyen-dras-johnson-2017,
      author    = {Nguyen, Dat Quoc  and  Dras, Mark  and  Johnson, Mark},
      title     = {{A Novel Neural Network Model for Joint POS Tagging and Graph-based Dependency Parsing}},
      booktitle = {Proceedings of the CoNLL 2017 Shared Task: Multilingual Parsing from Raw Text to Universal Dependencies},
      year      = {2017},
      pages     = {134--142},
      url       = {http://www.aclweb.org/anthology/K17-3014}
    }

Please cite the paper above when jPTDP is used to produce published results or incorporated into other software. I would highly appreciate to have your bug reports, comments and suggestions about jPTDP. As a free open-source implementation, jPTDP is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

### Installation

jPTDP requires the following software packages:

* `Python 2.7`
* [`DyNet` version 2.0](http://dynet.readthedocs.io/en/latest/python.html)

Once you installed the prerequisite packages above, you can clone or download (and then unzip) jPTDP. Next sections show jPTDP instructions to train a new joint model for POS tagging and dependency parsing, and then to utilize pre-trained models.

### Train a joint model 

Suppose that `SOURCE_DIR` is simply used to denote the source code directory. Similar to file `train.conllu` (and file `dev.conllu`) in folder `SOURCE_DIR/sample` or treebanks in the [Universal Dependencies project](http://universaldependencies.org/), the training (and development) data is formatted following 10-column data format. See details at [http://universaldependencies.org/format.html](http://universaldependencies.org/format.html). For training, jPTDP will only use information from columns 1 (ID), 2 (FORM), 4 (Coarse-grained POS tags---UPOSTAG), 7 (HEAD) and 8 (DEPREL). If you would like to use the fine-grained language-specific POS tags (in 5th column) instead of the coarse-grained POS tags (in 4th column), you should use `swapper.py` in folder `SOURCE_DIR/utils` to swap contents in 4th and 5th columns:

    SOURCE_DIR$ python utils/swapper.py <path-to-train-(and-dev)-file>
    
For example:
    
    SOURCE_DIR$ python utils/swapper.py sample/train.conllu
    SOURCE_DIR$ python utils/swapper.py sample/dev.conllu

will generate two new files for training: `train.conllu.ux2xu` and `dev.conllu.ux2xu` in folder `SOURCE_DIR/sample`. 

__To train a joint model for POS tagging and dependency parsing, you perform:__

    SOURCE_DIR$ python jPTDP.py --dynet-seed 123456789 [--dynet-mem <int>] [--epochs <int>] [--lstmdims <int>] [--lstmlayers <int>] [--wembedding <int>] [--cembedding <int>] [--model <String>] [--params <String>] --outdir <path-to-output-directory> --train <path-to-train-file>  [--dev <path-to-dev-file>]

where hyper-parameters in [] are optional:

 * `--dynet-mem`: Specify DyNet memory in MB.
 * `--epochs`: Specify number of traning epochs. Default value is 30.
 * `--lstmdims`: Specify number of BiLSTM dimensions. Default value is 128.
 * `--lstmlayers`: Specify number of BiLSTM layers. Default value is 2.
 * `--wembedding`: Specify size of word embeddings. Default value is 128.
 * `--cembedding`: Specify size of character embeddings. Default value is 64.
 * `--model`: Specify a  name for model parameters file. Default value is "model".
 * `--params`: Specify a  name for model hyper-parameters file. Default value is "model.params".
 * `--outdir`: Specify path to directory where the trained model will be saved. 
 * `--train`: Specify path to training data file.
 * `--dev`: Specify path to development data file. 

__For example:__

    SOURCE_DIR$ python jPTDP.py --dynet-seed 123456789 --dynet-mem 1000 --epochs 30 --lstmdims 128 --lstmlayers 2 --wembedding 128 --cembedding 64 --model trialmodel --params trialmodel.params --outdir sample/ --train sample/train.conllu --dev sample/dev.conllu
    
will produce model files `trialmodel` and `trialmodel.params` in folder `SOURCE_DIR/sample`.

### Utilize a pre-trained model

Pre-trained joint models for *universal* POS tagging and dependency parsing to 40+ languages can be found at   [__HERE__](https://drive.google.com/drive/folders/0B5eBgc8jrKtpTXhfRmpKbUEtdlE?usp=sharing). The pre-trained models are learned with default hyper-parameters, using dependency treebanks from [the Universal Dependencies project](http://universaldependencies.org/) [v2.0](http://hdl.handle.net/11234/1-1983). 

Assume that you are going to utilize a pre-trained model for annotating a corpus whose _each line represents a tokenized/word-segmented sentence_. You  should use `converter.py` in folder `SOURCE_DIR/utils`  to obtain a 10-column data format of this corpus:

    SOURCE_DIR$ python utils/converter.py <file-path>

For example:
    
    SOURCE_DIR$ python utils/converter.py sample/test

will generate  in folder `SOURCE_DIR/sample` a file named `test.conllu` which can be used later as input to the pre-trained model. 

__To utilize a pre-trained model for POS tagging and dependency parsing, you perform:__

    SOURCE_DIR$ python jPTDP.py --predict --model <path-to-model-parameters-file> --params <path-to-model-hyper-parameters-file> --test <path-to-10-column-input-file> --outdir <path-to-output-directory> --output <String>
 
* `--model`: Specify path to model parameters file.
* `--params`: Specify path to model hyper-parameters file.
* `--test`: Specify path to 10-column input file.
* `--outdir`: Specify path to directory where output file will be saved.
* `--output`: Specify name of the  output file.


For example: 

    SOURCE_DIR$ python jPTDP.py --predict --model sample/trialmodel --params sample/trialmodel.params --test sample/test.conllu --outdir sample/ --output test.conllu.pred
    SOURCE_DIR$ python jPTDP.py --predict --model sample/trialmodel --params sample/trialmodel.params --test sample/dev.conllu --outdir sample/ --output dev.conllu.pred
    
will produce output files `test.conllu.pred` and `dev.conllu.pred` in folder `SOURCE_DIR/sample`.

### Acknowledgments
jPTDP code is based on the implementations of Plank et al. (2016)'s POS tagger and Kiperwasser and Goldberg (2016b)'s graph-based dependency parser. I would like to thank Plank et al. (2016) and Kiperwasser and Goldberg (2016b) for making their codes available.
