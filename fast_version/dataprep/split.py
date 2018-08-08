import argparse
import random

# shuffle sentneces
# split to N folds
# write N folds on a word level

random.seed(10)

def split(args):
  """
  This script split the sentence sorpus to N folds
  """
  # filter sentences that are oov w.r.t. the embedding
  corpus = []
  with open(args.fdata, encoding = "ISO-8859-1") as f:
    for line in f:
      if line=='\n':
        continue
      line = line.strip()
      sen = line.split()
      sen = " ".join(sen)    
      corpus.append(sen)

  # break into folds
  for itr in range(args.folds):
    # an outer loop makes it N fold sets
    ffold = open(args.odir+"_"+str(itr), "w")
    unit = corpus[itr::args.folds]
    for tst in unit:
      ffold.write(tst)
      ffold.write("\n")
         
    ffold.close()

if __name__ == "__main__":

  parser = argparse.ArgumentParser(
      description='Split Data')
  parser.add_argument('--fdata', type=str, help='corpus of sentences file name')
  parser.add_argument('--odir', type=str, help='output dir')
  parser.add_argument('--folds', type=int, help='folds')
  args = parser.parse_args()
  split(args)
  
