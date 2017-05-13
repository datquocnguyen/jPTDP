# coding=utf-8
import os
import sys

#Convert word-segmented corpus into 10-column format for dependency parsing
def conllConverter(inputFilePath):
    writer = open(inputFilePath + ".conllu", "w")
    lines = open(inputFilePath, "r").readlines()
    for line in lines:
        tok = line.strip().split()
        if not tok or line.strip() == '':
            writer.write("\n")
        else:
            count = 0
            for word in tok:
                count += 1
                writer.write(str(count) + "\t" + word + "\t" + '\t'.join(['_'] * 8) + "\n")
        writer.write("\n")    
    writer.close()

if __name__ == "__main__":
    conllConverter(sys.argv[1])
    pass