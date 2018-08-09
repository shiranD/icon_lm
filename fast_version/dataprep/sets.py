import argparse
import random

# create_sets from N folds


def create_sets(args):
    random.seed(10)

    ftrain = open(args.path2sets + "train", "w")
    ftest = open(args.path2sets + "test", "w")
    fvalid = open(args.path2sets + "valid", "w")

    for fold in range(args.numfolds):
        if fold == args.foldnum:
            with open(args.foldspath + "_" + str(fold), encoding="ISO-8859-1") as f:
                for sentence in f:
                    ftest.write(sentence)
        elif fold == args.foldnum + 1 or (args.foldnum == args.numfolds - 1 and fold == 0):
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
    parser.add_argument('--path2sets', type=str, help='path t output sets')
    parser.add_argument('--foldnum', type=int, help='current fold')
    parser.add_argument('--numfolds', type=int, help='total number of folds')
    args = parser.parse_args()
    create_sets(args)
