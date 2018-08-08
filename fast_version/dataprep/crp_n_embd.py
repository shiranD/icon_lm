import argparse

def refine_corpus(args):
  """
  This script refines the textual data
  of sentences to be of sentecens
  that contain embbedings only
  - fdata is the sentecne corpus
  - fembed is the embedding file
  """
  
  # load embedding terms
  embds = {}
  with open(args.fembd, encoding = "ISO-8859-1") as f:
    for line in f:
      line = line.strip()
      term = line.split(" ")[0]
      embds[term.lower()]=1

  corpus = []
  with open(args.fdata, encoding = "ISO-8859-1") as f:
    for line in f:
      line = line.strip()
      sen = line.split()
      sentence = []
      for term in sen:
        if "_N_" in term:
          sentence.append(term)
          continue
        try:
          embds[term]
          sentence.append(term)
        except:
          sentence.append("<unk>")
      sen = " ".join(sentence)    
      print(sen)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='Fiterout embeddingless sentences')
    parser.add_argument('--fdata', type=str, help='data path')
    parser.add_argument('--fembd', type=str, help='embedding path')
    args = parser.parse_args()
    refine_corpus(args)
