# coding=utf-8
from collections import Counter
import re


class ConllEntry:
    def __init__(self, id, form, lemma, pos, xpos, feats=None, parent_id=None, relation=None, deps=None, misc=None):
        self.id = id
        self.form = form
        self.norm = normalize(form)
        self.xpos = xpos 
        self.pos = pos 
        self.parent_id = parent_id
        self.relation = relation

        self.lemma = lemma
        self.feats = feats
        self.deps = deps
        self.misc = misc

        self.pred_parent_id = None
        self.pred_relation = None
        self.pred_pos = None
        
        self.idChars = []

    def __str__(self):
        values = [str(self.id), self.form, self.lemma, self.pred_pos, self.xpos, self.feats, str(self.pred_parent_id) if self.pred_parent_id is not None else None, self.pred_relation, self.deps, self.misc]
        return '\t'.join(['_' if v is None else v for v in values])


def vocab(conll_path):
    wordsCount = Counter()
    posCount = Counter()
    relCount = Counter()

    #Character vocabulary
    c2i = {}
    c2i["_UNK"] = 0  # unk char
    c2i["<w>"] = 1   # word start
    c2i["</w>"] = 2  # word end index
    c2i["NUM"] = 3
    
    root = ConllEntry(0, '*root*', '*root*', 'ROOT-POS', 'ROOT-CPOS', '_', -1, 'rroot', '_', '_')
    root.idChars = [1,2]
    tokens = [root]
    
    for line in open(conll_path, 'r'):
        tok = line.strip().split('\t')
        if not tok or line.strip() == '':
            if len(tokens)>1: 
                wordsCount.update([node.norm for node in tokens if isinstance(node, ConllEntry)])
                posCount.update([node.pos for node in tokens if isinstance(node, ConllEntry)])
                relCount.update([node.relation for node in tokens if isinstance(node, ConllEntry)])
            tokens = [root]
        else:
            if line[0] == '#' or '-' in tok[0] or '.' in tok[0]:
                tokens.append(line.strip())
            else:
                entry = ConllEntry(int(tok[0]), tok[1], tok[2], tok[3], tok[4], tok[5], int(tok[6]) if tok[6] != '_' else -1, tok[7], tok[8], tok[9])
                
                if entry.norm == 'NUM':
                    entry.idChars = [1,3,2]
                else:    
                    chars_of_word = [1]
                    for char in tok[1]:
                        if char not in c2i:
                            c2i[char] = len(c2i)
                        chars_of_word.append(c2i[char])
                    chars_of_word.append(2)
                    entry.idChars = chars_of_word
                
                tokens.append(entry)
                
                
    if len(tokens) > 1:
        wordsCount.update([node.norm for node in tokens if isinstance(node, ConllEntry)])
        posCount.update([node.pos for node in tokens if isinstance(node, ConllEntry)])
        relCount.update([node.relation for node in tokens if isinstance(node, ConllEntry)])
    
    return (wordsCount, {w: i for i, w in enumerate(wordsCount.keys())}, c2i, posCount.keys(), relCount.keys())


def read_conll(fh,c2i):
    #Character vocabulary
    root = ConllEntry(0, '*root*', '*root*', 'ROOT-POS', 'ROOT-CPOS', '_', -1, 'rroot', '_', '_')
    root.idChars = [1,2]
    tokens = [root]
    
    for line in fh:
        tok = line.strip().split('\t')
        if not tok or line.strip() == '':
            if len(tokens)>1: yield tokens
            tokens = [root]
        else:
            if line[0] == '#' or '-' in tok[0] or '.' in tok[0]:
                tokens.append(line.strip())
            else:
                entry = ConllEntry(int(tok[0]), tok[1], tok[2], tok[3], tok[4], tok[5], int(tok[6]) if tok[6] != '_' else -1, tok[7], tok[8], tok[9])
                
                if entry.norm == 'NUM':
                    entry.idChars = [1,3,2]
                else:    
                    chars_of_word = [1]
                    for char in tok[1]:
                        if char in c2i:
                            chars_of_word.append(c2i[char])
                        else:
                            chars_of_word.append(0)
                    chars_of_word.append(2)
                    entry.idChars = chars_of_word
                
                tokens.append(entry)
                
                
    if len(tokens) > 1:
        yield tokens


def write_conll(fn, conll_gen):
    with open(fn, 'w') as fh:
        for sentence in conll_gen:
            for entry in sentence[1:]:
                fh.write(str(entry) + '\n')
            fh.write('\n')


numberRegex = re.compile("[0-9]+|[0-9]+\\.[0-9]+|[0-9]+[0-9,]+");
def normalize(word):
    return 'NUM' if numberRegex.match(word) else word.lower()

