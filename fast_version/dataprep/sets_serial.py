import argparse
import random

# create sets from N folds

def create_sets(args):
    random.seed(10)
    for i in range(args.numfolds):

        path2set = args.path2sets + str(i) + "/" 
        ftrain = open(path2set + "train", "w")
        ftest = open(path2set + "test", "w")
        fvalid = open(path2set + "valid", "w")
    
        for fold in range(args.numfolds):
            if fold == foldnum:
                with open(args.foldspath + "_" + str(fold), encoding="ISO-8859-1") as f:
                    for sentence in f:
                        ftest.write(sentence)
            elif fold == foldnum + 1 or (foldnum == args.numfolds - 1 and fold == 0):
                with open(args.foldspath + "_" + str(fold), encoding="ISO-8859-1") as f:
                    for sentence in f:
                        fvalid.write(sentence)
            else:
                with open(args.foldspath + "_" + str(fold), encoding="ISO-8859-1") as f:
                    for sentence in f:
                        ftrain.write(sentence)
    
        ftest.close()
        ftrain.close()
        fvalid.close()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='Split Data')
    parser.add_argument('--foldspath', type=str, help='folds path')
    parser.add_argument('--path2sets', type=str, help='path to output sets')
    parser.add_argument('--numfolds', type=int, help='total number of folds')
    args = parser.parse_args()
    create_sets(args)
