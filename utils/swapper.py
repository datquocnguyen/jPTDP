# coding=utf-8
import os
import sys

def swapUPosXPos(inputFilePath):
    writer = open(inputFilePath + ".ux2xu", "w")
    lines = open(inputFilePath, "r").readlines()
    for line in lines:
        tok = line.strip().split('\t')
        if not tok or line.strip() == '':
            writer.write("\n")
        else:
            if line[0] == '#' or '-' in tok[0] or '.' in tok[0]:
                writer.write(line.strip() + "\n")
            else:
                if tok[4] != "_":
                    temp = tok[3]
                    tok[3] = tok[4]
                    tok[4] = temp
                writer.write('\t'.join(['_' if v is None else v for v in tok]) + "\n")
    writer.close()

def swapUPosXPos_folder(inputFolderPath):
    for path, subdirs, files in os.walk(inputFolderPath):
        folPath = path.replace("\\", "/") + "/"
        for name in files:
            if name.endswith(".conllu") > 0 or name.endswith(".conll") > 0:
                print folPath + name
                writer = open(folPath + name + ".ux2xu", "w")
                lines = open(folPath + name, "r").readlines()
                for line in lines:
                    tok = line.strip().split('\t')
                    if not tok or line.strip() == '':
                        writer.write("\n")
                    else:
                        if line[0] == '#' or '-' in tok[0] or '.' in tok[0]:
                            writer.write(line.strip() + "\n")
                        else:
                            if tok[4] != "_":
                                temp = tok[3]
                                tok[3] = tok[4]
                                tok[4] = temp
                            writer.write('\t'.join(['_' if v is None else v for v in tok]) + "\n")
                writer.close()
    
if __name__ == "__main__":
    #swapUPosXPos_folder("/home/dqnguyen/workspace/UD/UD-conll2017")
    swapUPosXPos(sys.argv[1])
    pass