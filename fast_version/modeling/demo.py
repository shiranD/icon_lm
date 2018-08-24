import argparse
import data
import torch
from torch.autograd import Variable
import cupy as cp
from embedder import sym2vec, index2embed, term2sym
import os
import pickle

parser = argparse.ArgumentParser(
    description='PyTorch RNN/LSTM training for Language Models')
parser.add_argument('--aug', action='store_true', help='use icons or not')
parser.add_argument('--data', type=str, help='location of the data corpus')
parser.add_argument('--icon', type=str, help='path to icon embeddings', default='')
iparser.add_argument('--iconD', type=str, help='path to icon dict', default='')
parser.add_argument('--embd', type=str,
                    help='location of the embeddings')
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
parser.add_argument('--bptt', type=int, default=35,
                    help='sequence length')
parser.add_argument('--dropout', type=float, default=0.2,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--tied', action='store_true',
                    help='tie the word embedding and softmax weights')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--log-interval', type=int, default=200, metavar='N',
                    help='report interval')
parser.add_argument('--load', type=str, default='model.pt',
                    help='path to load the model')
args = parser.parse_args()

# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    else:
        torch.cuda.manual_seed(args.seed)

###############################################################################
# Load data
###############################################################################

# check for icon embedding incorporation
assert type(args.aug) == bool, "%r is not a boolian" % args.aug

if args.aug:
    assert os.path.exists(os.path.dirname(
        args.icon)), "%r is not a valid path" % args.icon
    embdict = sym2vec(args.embd, args.icon)
    embedding_dict, syn_dict = term2sym(args.data, args.embd, args.iconD)
else:
    embdict = sym2vec(args.embd)

with open(args.data + 'corpus.pickle', 'rb') as f:
    corpus = pickle.load(f)

# build the specific type of model
ntokens = len(corpus.dictionary)

# Load the best saved model.
with open(args.load, 'rb') as f:
    model = torch.load(f)

###############################################################################
# Demo code
###############################################################################


def repackage_hidden(h):
    """Wraps hidden states in new Variables, to detach them from their history."""
    if isinstance(h, Variable):
        return Variable(h.data)
    else:
        return tuple(repackage_hidden(v) for v in h)


def get_batch(source, i, evaluation=False):
    seq_len = min(args.bptt, len(source) - 1 - i)
    data = Variable(source[i:i + seq_len], volatile=evaluation)
    target = Variable(source[i + 1:i + 1 + seq_len].view(-1))
    return data, target


def tokenize(sent):
    sent = sentence.lower().split()
    # Assert that words are in dictionary
    assert len(sent) > 0,\
        "the input was empty of tokens"
    for word in sent:
        try:
            assert word in embedding_dict, \
                "%r contains invalid symbol" % word
        except:
            assert word in syn_dict, \
                "%r contains invalid symbol" % word
    ids = torch.LongTensor(len(sent))
    syms = []
    for i, word in enumerate(sent):
        try:
            sym = embedding_dict[word]
        except:
            sym = syn_dict[word]
        ids[i] = corpus.dictionary.word2idx[sym]
        syms.append(sym)
    print(" ".join(syms))
    return Variable(ids)


def evaluate(data_source, embedict, data_dict, ntokens):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    total_loss = 0
    k = 3
    eval_batch_size = len(data_source)
    len_src = len(data_source)
    hidden = model.init_hidden(eval_batch_size)
    for i in range(0, len_src):
        data_source = data_source.view(1, -1)
        data = index2embed(data_source, embedict, corpus.dictionary, args.emsize)
        output, hidden = model(data, hidden)
        output_flat = output.view(-1, ntokens)
        preds = output_flat[len_src - 1, :]
        preds = preds.data.cpu().numpy()
        # retrive top k
        preds = cp.array(preds)
        topk_idx = cp.argsort(preds)[ntokens - k:]
        topk = []
        for top in topk_idx:
            term = corpus.dictionary[int(top)]
            topk.append(term)
        # show a gradual prediction for each word
        hidden = repackage_hidden(hidden)

    return topk


if __name__ == "__main__":

    sentence = ""
    print("Press Q to stop")
    while sentence != "Q":
        sentence = input('\nEnter partial sentence: ')
        demo_data = tokenize(sentence)
        topk = evaluate(demo_data, embdict, corpus.dictionary, ntokens)
        print(topk)
