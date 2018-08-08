import argparse
import string
from collections import defaultdict
from nltk import TreebankWordTokenizer

def iconize_corpus(args):
    """
    This script retrives the sentences that contains
    at least one icon term
    - fdata is the current corpus
    - fembed is the icon embedding file
    """
    # load embedding terms
    embdwrds = defaultdict(str)
    embdsyns = defaultdict(str)
    with open(args.fembd, encoding = "utf-8") as f:
        for line in f:
            line = line.strip()
            terms = line.split()
            term = terms[0]
            code = terms[1]
            wtype = terms[2]
            if wtype == "main":
                embdwrds[term] = code
            else:
                embdsyns[term] = code
  
    # filter sentences that are oov w.r.t. the embedding
    tbt = TreebankWordTokenizer()
    plist = ["..", "...", "``", "''", "."]
    with open(args.fdata, encoding = "utf-8") as f:
        for line in f:
            line = line.strip()
            sen = line.lower()
            sen = ''.join(i for i in sen if ord(i)<123)
            sen = tbt.tokenize(sen)
            sen = [x for x in sen if not x in string.punctuation]
            sen = [x for x in sen if not x in plist]
            sentence = []
            for word in sen:
                code = embdwrds[word]
                if code != "":
                    sentence.append(code)
                elif embdsyns[word] != "":
                    code = embdsyns[word]
                    sentence.append(code)
                #else: # comment for pure icon mode
                    #sentence.append(word) # pure icon mode
            sentence = str.join(" ", sentence)    
            print(sentence)	

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='Iconize Data')
    parser.add_argument('--fembd', type=str, help='dictionary path')
    parser.add_argument('--fdata', type=str, help='corpus path')
    args = parser.parse_args()
    iconize_corpus(args)
