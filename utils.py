# coding=utf-8
from collections import Counter
import os, re, codecs, string


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
        values = [str(self.id), self.form, self.lemma, self.pred_pos, self.xpos, self.feats,
                  str(self.pred_parent_id) if self.pred_parent_id is not None else None, self.pred_relation, self.deps,
                  self.misc]
        return '\t'.join(['_' if v is None else v for v in values])


def vocab(conll_path):
    wordsCount = Counter()
    posCount = Counter()
    relCount = Counter()

    # Character vocabulary
    c2i = {}
    c2i["_UNK"] = 0  # unk char
    c2i["<w>"] = 1  # word start
    c2i["</w>"] = 2  # word end index
    c2i["NUM"] = 3
    c2i["EMAIL"] = 4
    c2i["URL"] = 5

    root = ConllEntry(0, '*root*', '*root*', 'ROOT-POS', 'ROOT-CPOS', '_', -1, 'rroot', '_', '_')
    root.idChars = [1, 2]
    tokens = [root]

    for line in open(conll_path, 'r'):
        tok = line.strip().split('\t')
        if not tok or line.strip() == '':
            if len(tokens) > 1:
                wordsCount.update([node.norm for node in tokens if isinstance(node, ConllEntry)])
                posCount.update([node.pos for node in tokens if isinstance(node, ConllEntry)])
                relCount.update([node.relation for node in tokens if isinstance(node, ConllEntry)])
            tokens = [root]
        else:
            if line[0] == '#' or '-' in tok[0] or '.' in tok[0]:
                tokens.append(line.strip())
            else:
                entry = ConllEntry(int(tok[0]), tok[1], tok[2], tok[3], tok[4], tok[5],
                                   int(tok[6]) if tok[6] != '_' else -1, tok[7], tok[8], tok[9])

                if entry.norm == 'NUM':
                    entry.idChars = [1, 3, 2]
                elif entry.norm == 'EMAIL':
                    entry.idChars = [1, 4, 2]
                elif entry.norm == 'URL':
                    entry.idChars = [1, 5, 2]
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


def read_conll(fh, c2i):
    # Character vocabulary
    root = ConllEntry(0, '*root*', '*root*', 'ROOT-POS', 'ROOT-CPOS', '_', -1, 'rroot', '_', '_')
    root.idChars = [1, 2]
    tokens = [root]

    for line in fh:
        tok = line.strip().split('\t')
        if not tok or line.strip() == '':
            if len(tokens) > 1: yield tokens
            tokens = [root]
        else:
            if line[0] == '#' or '-' in tok[0] or '.' in tok[0]:
                tokens.append(line.strip())
            else:
                entry = ConllEntry(int(tok[0]), tok[1], tok[2], tok[3], tok[4], tok[5],
                                   int(tok[6]) if tok[6] != '_' else -1, tok[7], tok[8], tok[9])

                if entry.norm == 'NUM':
                    entry.idChars = [1, 3, 2]
                elif entry.norm == 'EMAIL':
                    entry.idChars = [1, 4, 2]
                elif entry.norm == 'URL':
                    entry.idChars = [1, 5, 2]
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

#puncts = re.compile("^[\\\!\"#$%&'()*+,-./:;<=>?@[\]^_`{|}~]+$")
brackets = {"-lrb-" : "(", "-rrb-" : ")", "-lsb-" : "[", "-rsb-" : "]", "-lcb-" : "{", "-rcb-" : "}", "&lt;" : "<", "&gt;" : ">", "&amp;" : "&"}
def read_conll_predict(fh, c2i, wordsCount):
    # Character vocabulary
    root = ConllEntry(0, '*root*', '*root*', 'ROOT-POS', 'ROOT-CPOS', '_', -1, 'rroot', '_', '_')
    root.idChars = [1, 2]
    tokens = [root]

    brackets_train = {} 
    for word in wordsCount:
    	if word in brackets:
    		brackets_train[brackets[word]] = word

    for line in fh:
        tok = line.strip().split('\t')
        if not tok or line.strip() == '':
            if len(tokens) > 1: yield tokens
            tokens = [root]
        else:
            if line[0] == '#' or '-' in tok[0] or '.' in tok[0]:
                tokens.append(line.strip())
            else:
                entry = ConllEntry(int(tok[0]), tok[1], tok[2], tok[3], tok[4], tok[5],
                                   int(tok[6]) if tok[6] != '_' else -1, tok[7], tok[8], tok[9])

                if entry.norm == 'NUM':
                    entry.idChars = [1, 3, 2]
                elif entry.norm == 'EMAIL':
                    entry.idChars = [1, 4, 2]
                elif entry.norm == 'URL':
                    entry.idChars = [1, 5, 2]
                else:
                    if entry.norm == "”" or entry.norm == "’":
                        tok[1] = "''"
                        entry.norm = '"'
                    if entry.norm == "“" or entry.norm == "‘":
                        tok[1] = "``"
                        entry.norm = '"'
                    if "’" in entry.norm:
                        entry.norm = re.sub(r"’", "'", entry.norm)
                        tok[1] = entry.norm
                    if entry.norm == "—":
                        entry.norm = "-"
                        tok[1] = "-"

                    if entry.norm in brackets:
                    	entry.norm = brackets[entry.norm]
                    	tok[1] = entry.norm
                    if entry.norm in brackets_train:
                    	entry.norm = brackets_train[entry.norm]
                    	tok[1] = str(entry.norm).upper()
                        if tok[1].lower() in ["&lt;", "&gt;", "&amp;"]:
                            tok[1] = tok[1].lower()

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

numberRegex = re.compile("[0-9]+|[0-9]+\\.[0-9]+|[0-9]+[0-9,]+")
def normalize(word):
    if numberRegex.match(word):
        return 'NUM'
    else:
        w = word.lower()
        w = re.sub(r".+@.+", "EMAIL", w)
        w = re.sub(r"(https?://|www\.).*", "URL", w)
        w = re.sub(r"``", '"', w)
        w = re.sub(r"''", '"', w)
        return w

#try:
#    import lzma
#except ImportError:
#    from backports import lzma

def load_embeddings_file(file_name, lower=False):
        """
        Load embeddings file. Uncomment comments above and below if file format is .xz
        """
        if not os.path.isfile(file_name):
            print(file_name, "does not exist")
            return {}, 0

        emb={}
        print("Load pre-trained word embeddings: {}".format(file_name))

        open_func = codecs.open

        #if file_name.endswith('.xz'):
        #    open_func = lzma.open
        #else:
        #    open_func = codecs.open

        with open_func(file_name, 'rb') as f:
            reader = codecs.getreader('utf-8')(f, errors='ignore')
            reader.readline()

            count = 0
            for line in reader:
                try:
                    fields = line.strip().split()
                    vec = [float(x) for x in fields[1:]]
                    word = fields[0]
                    if lower:
                        word = word.lower()
                    if word not in emb:
                        emb[word] = vec
                except ValueError:
                    #print("Error converting: {}".format(line))
                    pass

                count += 1
                if count >= 1500000:
                    break
        return emb, len(emb[word])


