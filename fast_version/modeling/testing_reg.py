import argparse
import torch
import torch.nn as nn
from torch.autograd import Variable
from embedder import sym2vec 
import logging
import os
from fwd_reg import feedfwd
import pickle

parser = argparse.ArgumentParser(
    description='PyTorch RNN/LSTM training for Language Models')
parser.add_argument('--data', type=str, help='location of the data corpus')
parser.add_argument('--aug', action='store_true', help='use icons or not')
parser.add_argument('--icon', type=str, help='path to icon embeddings', default='')
parser.add_argument('--embd', type=str,
                    help='location of the embeddings')
parser.add_argument('--modeltype', type=str,
                    help='basic, KL, or embedding')
parser.add_argument('--modelunit', type=str, default='LSTM',
                    help='type of recurrent net (RNN_TANH, RNN_RELU, LSTM, GRU)')
parser.add_argument('--emsize', type=int, default=200,
                    help='size of word embeddings')
parser.add_argument('--nhid', type=int, default=200,
                    help='number of hidden units per layer')
parser.add_argument('--nlayers', type=int, default=2,
                    help='number of layers')
parser.add_argument('--lr', type=float, default=20,
                    help='initial learning rate')
parser.add_argument('--clip', type=float, default=0.25,
                    help='gradient clipping')
parser.add_argument('--batch_size', type=int, default=20, metavar='N',
                    help='batch size')
parser.add_argument('--bptt', type=int, default=100,
                    help='sequence length')
parser.add_argument('--dropout', type=float, default=0.2,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--testlog', type=str, help='log filename')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--fold', type=int, default=1111,
                    help='random seed')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--gpu', type=int, default=[0,1], nargs='+', help='used gpu')
parser.add_argument('--load', type=str, default='model.pt',
                    help='path to load the final model')
args = parser.parse_args()
# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)

logging.basicConfig(filename=args.testlog,level=logging.INFO)

if torch.cuda.is_available():
    logging.info("CUDA is up")
    if not args.cuda:
        logging.info("WARNING: You have a CUDA device, so you should probably run with --cuda")
    else:
        torch.cuda.manual_seed(args.seed)

###############################################################################
# Load data
###############################################################################

assert type(args.aug) == bool, "%r is not a boolian" % args.aug
assert args.modeltype in ["basic", "nce", "embd"], "invalid model %r" % args.modeltype

if args.aug:
    assert os.path.exists(os.path.dirname(
        args.icon)), "%r is not a valid path" % args.icon
    embdict = sym2vec(args.data, args.embd, args.icon)
else:
    embdict = sym2vec(args.data, args.embd)

with open(args.data+'corpus.pickle', 'rb') as f:
    corpus = pickle.load(f)

def batchify(data, bsz):
    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    data = data.view(bsz, -1).t().contiguous()
    if args.cuda:
        logging.info("using cuda for data")
        data = data.cuda()
    return data

eval_batch_size = 5
test_data = batchify(corpus.test, eval_batch_size)
tst_ln, col = test_data.size()

with open(args.load, 'rb') as f:
    model = torch.load(f)

model.set_mode('eval')

criterion = nn.CrossEntropyLoss()
# check for icon embedding incorporation
###############################################################################
# Testing code
###############################################################################

logging.info('Test resutls')
acc_token, mrr_token, acc10 = feedfwd(model, test_data, embdict, args.bptt, eval_batch_size, corpus.dictionary, args.emsize, criterion)
logging.info('Token Accuracy is {}'.format(acc_token))
logging.info('Token Accuracy @ 10 is {}'.format(acc10))
logging.info('Token MRR is {}'.format(mrr_token))
logging.info('=' * 89)
