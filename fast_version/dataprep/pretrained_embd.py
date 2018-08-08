import argparse
import os
from collections import defaultdict


def load_corpus(corpus):
    # load embedding terms
    uniq = defaultdict(bool)
    with open(corpus, encoding="ISO-8859-1") as f:
      for line in f:
        for term in line.split():
          uniq[term] = True
    uniq[">"] = True
    return uniq


def write(fembd, corpus, foutpath):
    # load embedding terms and filter
    fout = open(foutpath, "w")
    with open(fembd, encoding="ISO-8859-1") as f:
      for line in f:
        line = line.strip()
        term, vec = line.split(" ", 1)
        if corpus[term] == True:
          fout.write(line + "\n")
      fout.close()


def refine_embed(args):
    """
    The purpose of this script is to
    retrieve the corpus embeddings
    """
    # extract corpus words' list
    words = load_corpus(args.corpus)
    # parse the embedding and write to file
    write(args.embdpath, words, args.fout)


if __name__ == "__main__":
    """
    Extract just the relevant embeddings
    from the pretrained (non-icon) data
    - pretrained embeddings
    - corpus unique terms
    - refined embedding filename (output)
    """
    parser = argparse.ArgumentParser(description='Retrieve corpus Pretrained embeddings')
    parser.add_argument('--embdpath', type=str, help='embedding path')
    parser.add_argument('--corpus', type=str, help='corpus path')
    parser.add_argument('--fout', type=str, help='out folder')
    args = parser.parse_args()
    # assert input path validity
    assert os.path.exists(os.path.dirname(
        args.embdpath)), "%r is not a valid path" % args.embdpath
    refine_embed(args)
