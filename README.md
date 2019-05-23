# Neural Network Models for Joint POS Tagging and Dependency Parsing 

<img width="750" alt="jptdpv2" src="https://user-images.githubusercontent.com/2412555/48745055-ef25b500-ecbd-11e8-8f83-7160e42e61f7.png">


Implementations of joint models for  POS tagging and dependency parsing, as described in my papers:
  
  1. Dat Quoc Nguyen and Karin Verspoor. **2018**. [An improved neural network model for joint POS tagging and dependency parsing](http://www.aclweb.org/anthology/K18-2008). In *Proceedings of the CoNLL 2018 Shared Task: Multilingual Parsing from Raw Text to Universal Dependencies*, pages 81-91. [[.bib]](http://www.aclweb.org/anthology/K18-2008.bib) (**jPTDP v2.0**)
  2. Dat Quoc Nguyen, Mark Dras and Mark Johnson. **2017**. [A Novel Neural Network Model for Joint POS Tagging and Graph-based Dependency Parsing](http://www.aclweb.org/anthology/K17-3014). In *Proceedings of the CoNLL 2017 Shared Task: Multilingual Parsing from Raw Text to Universal Dependencies*, pages 134-142. [[.bib]](http://www.aclweb.org/anthology/K17-3014.bib)  (**jPTDP v1.0**)

This github project currently supports jPTDP v2.0, while v1.0 can be found in the [release](https://github.com/datquocnguyen/jPTDP/releases) section. Please **cite** paper [1] when jPTDP is used to produce published results or incorporated into other software. I would highly appreciate to have your bug reports, comments and suggestions about jPTDP. As a free open-source implementation, jPTDP is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

### Installation

jPTDP requires the following software packages:

* `Python 2.7`
* [`DyNet` v2.0](http://dynet.readthedocs.io/en/latest/python.html)

      $ virtualenv -p python2.7 .DyNet
      $ source .DyNet/bin/activate
      $ pip install cython numpy
      $ pip install dynet==2.0.3

Once you installed the prerequisite packages above, you can clone or download (and then unzip) jPTDP. Next sections show instructions to train a new joint model for POS tagging and dependency parsing, and then to utilize a pre-trained model.

**NOTE:** [jPTDP is also ported to run with Python 3.4+ by Santiago Castro](https://github.com/bryant1410/jPTDP/tree/python3). Also note that _pre-trained models I provide in the last section would not work with this ported version_  ([see a discussion](https://github.com/datquocnguyen/jPTDP/pull/3#issuecomment-423718383)).   Thus, you may want to retrain jPTDP if using this ported version.

### Train a joint model 

Suppose that `SOURCE_DIR` is simply used to denote the source code directory. Similar to files `train.conllu` and `dev.conllu` in folder `SOURCE_DIR/sample` or treebanks in the [Universal Dependencies (UD) project](http://universaldependencies.org/), the training and development files are formatted following 10-column data format. For training, jPTDP will only use information from columns 1 (ID), 2 (FORM), 4 (Coarse-grained POS tags---UPOSTAG), 7 (HEAD) and 8 (DEPREL). 


__To train a joint model for POS tagging and dependency parsing, you perform:__

    SOURCE_DIR$ python jPTDP.py --dynet-seed 123456789 [--dynet-mem <int>] [--epochs <int>] [--lstmdims <int>] [--lstmlayers <int>] [--hidden <int>] [--wembedding <int>] [--cembedding <int>] [--pembedding <int>] [--prevectors <path-to-pre-trained-word-embedding-file>] [--model <String>] [--params <String>] --outdir <path-to-output-directory> --train <path-to-train-file>  --dev <path-to-dev-file>

where hyper-parameters in [] are optional:

 * `--dynet-mem`: Specify DyNet memory in MB.
 * `--epochs`: Specify number of training epochs. Default value is 30.
 * `--lstmdims`: Specify number of BiLSTM dimensions. Default value is 128.
 * `--lstmlayers`: Specify number of BiLSTM layers. Default value is 2.
 * `--hidden`: Specify size of MLP hidden layer. Default value is 100.
 * `--wembedding`: Specify size of word embeddings. Default value is 100.
 * `--cembedding`: Specify size of character embeddings. Default value is 50.
 * `--pembedding`: Specify size of POS tag embeddings. Default value is 100.
 * `--prevectors`: Specify path to the pre-trained word embedding file for initialization. Default value is "None" (i.e. word embeddings are randomly initialized).
 * `--model`: Specify a  name for model parameters file. Default value is "model".
 * `--params`: Specify a  name for model hyper-parameters file. Default value is "model.params".
 * `--outdir`: Specify path to directory where the trained model will be saved. 
 * `--train`: Specify path to the training data file.
 * `--dev`: Specify path to the development data file. 


__For example:__

    SOURCE_DIR$ python jPTDP.py --dynet-seed 123456789 --dynet-mem 1000 --epochs 30 --lstmdims 128 --lstmlayers 2 --hidden 100 --wembedding 100 --cembedding 50 --pembedding 100  --model trialmodel --params trialmodel.params --outdir sample/ --train sample/train.conllu --dev sample/dev.conllu
    
will produce model files `trialmodel` and `trialmodel.params` in folder `SOURCE_DIR/sample`. 

If you would like to use the fine-grained language-specific POS tags in the 5th column instead of the coarse-grained POS tags in the 4th column, you should use `swapper.py` in folder `SOURCE_DIR/utils` to swap contents in the 4th and 5th columns:

    SOURCE_DIR$ python utils/swapper.py <path-to-train-(and-dev)-file>
    
For example:
    
    SOURCE_DIR$ python utils/swapper.py sample/train.conllu
    SOURCE_DIR$ python utils/swapper.py sample/dev.conllu

will generate two new files for training: `train.conllu.ux2xu` and `dev.conllu.ux2xu` in folder `SOURCE_DIR/sample`.

### Utilize a pre-trained model

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

### Pre-trained models

Pre-trained jPTDP v2.0 models, which were trained on English WSJ Penn treebank, GENIA and UD v2.2 treebanks, can be found at [__HERE__](https://drive.google.com/drive/folders/1my2w3zf4BPSX18QpfY1QhgeEcm_auS3h?usp=sharing).  Results on test sets (as detailed in paper [1]) are as follows:

  Treebank                       | Model name      | POS   | UAS   | LAS 
  ------------------------------ | --------------- | ----  | ----  | ----
  English WSJ Penn treebank      | model256        | 97.97 | 94.51 | 92.87
  English WSJ Penn treebank      | model           | 97.88 | 94.25 | 92.58

`model256` and `model` denote the pre-trained models which use 256- and 128-dimensional LSTM hidden states, respectively, i.e. `model256` is more accurate but slower.
  
  Treebank                       | Code            | UPOS  | UAS   | LAS 
  ------------------------------ | --------------- | ----  | ----  | ----
  UD_Afrikaans-AfriBooms         | af_afribooms    | 95.73 | 82.57 | 78.89
  UD_Ancient_Greek-PROIEL        | grc_proiel      | 96.05 | 77.57 | 72.84
  UD_Ancient_Greek-Perseus       | grc_perseus     | 88.95 | 65.09 | 58.35
  UD_Arabic-PADT                 | ar_padt         | 96.33 | 86.08 | 80.97
  UD_Basque-BDT                  | eu_bdt          | 93.62 | 79.86 | 75.07
  UD_Bulgarian-BTB               | bg_btb          | 98.07 | 91.47 | 87.69
  UD_Catalan-AnCora              | ca_ancora       | 98.46 | 90.78 | 88.40
  UD_Chinese-GSD                 | zh_gsd          | 93.26 | 82.50 | 77.51
  UD_Croatian-SET                | hr_set          | 97.42 | 88.74 | 83.62
  UD_Czech-CAC                   | cs_cac          | 98.87 | 89.85 | 87.13
  UD_Czech-FicTree               | cs_fictree      | 97.98 | 88.94 | 85.64
  UD_Czech-PDT                   | cs_pdt          | 98.74 | 89.64 | 87.04
  UD_Czech-PUD                   | cs_pud          | 96.71 | 87.62 | 82.28
  UD_Danish-DDT                  | da_ddt          | 96.18 | 82.17 | 78.88
  UD_Dutch-Alpino                | nl_alpino       | 95.62 | 86.34 | 82.37
  UD_Dutch-LassySmall            | nl_lassysmall   | 95.21 | 86.46 | 82.14
  UD_English-EWT                 | en_ewt          | 95.48 | 87.55 | 84.71
  UD_English-GUM                 | en_gum          | 94.10 | 84.88 | 80.45
  UD_English-LinES               | en_lines        | 95.55 | 80.34 | 75.40
  UD_English-PUD                 | en_pud          | 95.25 | 87.49 | 84.25
  UD_Estonian-EDT                | et_edt          | 96.87 | 85.45 | 82.13
  UD_Finnish-FTB                 | fi_ftb          | 94.53 | 86.10 | 82.45
  UD_Finnish-PUD                 | fi_pud          | 96.44 | 87.54 | 84.60
  UD_Finnish-TDT                 | fi_tdt          | 96.12 | 86.07 | 82.92
  UD_French-GSD                  | fr_gsd          | 97.11 | 89.45 | 86.43
  UD_French-Sequoia              | fr_sequoia      | 97.92 | 89.71 | 87.43
  UD_French-Spoken               | fr_spoken       | 94.25 | 79.80 | 73.45
  UD_Galician-CTG                | gl_ctg          | 97.12 | 85.09 | 81.93
  UD_Galician-TreeGal            | gl_treegal      | 93.66 | 77.71 | 71.63
  UD_German-GSD                  | de_gsd          | 94.07 | 81.45 | 76.68
  UD_Gothic-PROIEL               | got_proiel      | 93.45 | 79.80 | 71.85
  UD_Greek-GDT                   | el_gdt          | 96.59 | 87.52 | 84.64
  UD_Hebrew-HTB                  | he_htb          | 96.24 | 87.65 | 82.64
  UD_Hindi-HDTB                  | hi_hdtb         | 96.94 | 93.25 | 89.83
  UD_Hungarian-Szeged            | hu_szeged       | 92.07 | 76.18 | 69.75
  UD_Indonesian-GSD              | id_gsd          | 93.29 | 84.64 | 77.71
  UD_Irish-IDT                   | ga_idt          | 89.74 | 75.72 | 65.78
  UD_Italian-ISDT                | it_isdt         | 98.01 | 92.33 | 90.20
  UD_Italian-PoSTWITA            | it_postwita     | 95.41 | 84.20 | 79.11
  UD_Japanese-GSD                | ja_gsd          | 97.27 | 94.21 | 92.02
  UD_Japanese-Modern             | ja_modern       | 70.53 | 66.88 | 49.51
  UD_Korean-GSD                  | ko_gsd          | 93.35 | 81.32 | 76.58
  UD_Korean-Kaist                | ko_kaist        | 93.53 | 83.59 | 80.74
  UD_Latin-ITTB                  | la_ittb         | 98.12 | 82.99 | 79.96
  UD_Latin-PROIEL                | la_proiel       | 95.54 | 74.95 | 69.76
  UD_Latin-Perseus               | la_perseus      | 82.36 | 57.21 | 46.28
  UD_Latvian-LVTB                | lv_lvtb         | 93.53 | 81.06 | 76.13
  UD_North_Sami-Giella           | sme_giella      | 87.48 | 65.79 | 58.09
  UD_Norwegian-Bokmaal           | no_bokmaal      | 97.73 | 89.83 | 87.57
  UD_Norwegian-Nynorsk           | no_nynorsk      | 97.33 | 89.73 | 87.29
  UD_Norwegian-NynorskLIA        | no_nynorsklia   | 85.22 | 64.14 | 54.31
  UD_Old_Church_Slavonic-PROIEL  | cu_proiel       | 93.69 | 80.59 | 73.93
  UD_Old_French-SRCMF            | fro_srcmf       | 95.12 | 86.65 | 81.15
  UD_Persian-Seraji              | fa_seraji       | 96.66 | 88.07 | 84.07
  UD_Polish-LFG                  | pl_lfg          | 98.22 | 95.29 | 93.10
  UD_Polish-SZ                   | pl_sz           | 97.05 | 90.98 | 87.66
  UD_Portuguese-Bosque           | pt_bosque       | 96.76 | 88.67 | 85.71
  UD_Romanian-RRT                | ro_rrt          | 97.43 | 88.74 | 83.54
  UD_Russian-SynTagRus           | ru_syntagrus    | 98.51 | 91.00 | 88.91
  UD_Russian-Taiga               | ru_taiga        | 85.49 | 65.52 | 56.33
  UD_Serbian-SET                 | sr_set          | 97.40 | 89.32 | 85.03
  UD_Slovak-SNK                  | sk_snk          | 95.18 | 85.88 | 81.89
  UD_Slovenian-SSJ               | sl_ssj          | 97.79 | 88.26 | 86.10
  UD_Slovenian-SST               | sl_sst          | 89.50 | 66.14 | 58.13
  UD_Spanish-AnCora              | es_ancora       | 98.57 | 90.30 | 87.98
  UD_Swedish-LinES               | sv_lines        | 95.51 | 83.60 | 78.97
  UD_Swedish-PUD                 | sv_pud          | 92.10 | 79.53 | 74.53
  UD_Swedish-Talbanken           | sv_talbanken    | 96.55 | 86.53 | 83.01
  UD_Turkish-IMST                | tr_imst         | 92.93 | 70.53 | 62.55
  UD_Ukrainian-IU                | uk_iu           | 95.24 | 83.47 | 79.38
  UD_Urdu-UDTB                   | ur_udtb         | 93.35 | 86.74 | 80.44
  UD_Uyghur-UDT                  | ug_udt          | 87.63 | 76.14 | 63.37
  UD_Vietnamese-VTB              | vi_vtb          | 87.63 | 67.72 | 58.27


