import argparse
import time
import math
import torch
import torch.nn as nn
from torch.autograd import Variable
import data
import nmodel
from embedder import sym2vec, index2embed
import logging
import os
import pickle
import nce_loss

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
parser.add_argument('--epochs', type=int, default=40,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=20, metavar='N',
                    help='batch size')
parser.add_argument('--bptt', type=int, default=100,
                    help='sequence length')
parser.add_argument('--dropout', type=float, default=0.2,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--trainlog', type=str, help='log filename')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--pred', type=str,
                    help='path to save the predictions')
parser.add_argument('--tgt', type=str,
                    help='path to save the targets')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--save', type=str, default='model.pt',
                    help='path to save the final model')
args = parser.parse_args()
# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)

logging.basicConfig(filename=args.trainlog, level=logging.INFO)

if torch.cuda.is_available():
    logging.info("CUDA is up")
    if not args.cuda:
        logging.info("WARNING: You have a CUDA device, so you should probably run with --cuda")
    else:
        torch.cuda.manual_seed(args.seed)
if args.cuda:
    gpu = torch.cuda.current_device()

log_interval = 200

###############################################################################
# Load data
###############################################################################

if args.aug:
    assert os.path.exists(os.path.dirname(
        args.icon)), "%r is not a valid path" % args.icon
    embdict = sym2vec(args.data, args.embd, args.icon)
else:
    embdict = sym2vec(args.data, args.embd)

assert args.modeltype in ["basic", "nce", "embd"], "invalid model %r" % args.modeltype

if args.modeltype == "nce":
    try:
        with open(args.data + 'corpusNCE.pickle', 'rb') as f:
            corpus = pickle.load(f)
    except:
        corpus = data.Corpus(args.data)
        with open(args.data + 'corpusNCE.pickle', 'wb') as f:
            pickle.dump(corpus, f)

if args.modeltype == "basic":
    try:
        with open(args.data + 'corpus.pickle', 'rb') as f:
            corpus = pickle.load(f)
    except:
        corpus = data.Corpus(args.data)
        with open(args.data + 'corpus.pickle', 'wb') as f:
            pickle.dump(corpus, f)


def batchify(data, bsz):
    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    data = data.view(bsz, -1).t().contiguous()
    return data


eval_batch_size = 10
train_data = batchify(corpus.train, args.batch_size)
val_data = batchify(corpus.valid, eval_batch_size)

###############################################################################
# Build the model
###############################################################################
# check for icon embedding incorporation
assert type(args.aug) == bool, "%r is not a boolian" % args.aug

# build the specific type of model
ntokens = len(corpus.dictionary)
if args.modeltype == "nce":
    model = nmodel.RNNModel(
        args.modelunit,
        len(corpus.dictionary),
        args.emsize,
        args.nhid,
        args.nlayers,
        args.modeltype,
        args.dropout,
        corpus.dictionary.unigram)
    criterion = nce_loss.nce_loss()
    criterion_test = nn.CrossEntropyLoss()

elif args.modeltype == "basic":
    model = nmodel.RNNModel(
        args.modelunit,
        len(corpus.dictionary),
        args.emsize,
        args.nhid,
        args.nlayers,
        args.dropout)

    criterion = nn.CrossEntropyLoss()

logging.info(model)
if args.cuda:
    model.cuda(gpu)


###############################################################################
# Training code
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
    return data.cuda(), target.cuda()


def evaluate(data_source, embedict, data_dict):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    model.set_mode('eval')
    total_loss = 0
    total_words = 0
    ntokens = len(corpus.dictionary)
    hidden = model.init_hidden(eval_batch_size)
    for i in range(0, data_source.size(0) - 1, args.bptt):
        if i % 40000 == 0:
            logging.info("round " + str(i) + " " + str(data_source.size(0) - 1))
        data, targets = get_batch(data_source, i, evaluation=True)
        data = index2embed(data, embedict, corpus.dictionary, args.emsize)
        output, hidden = model(data, hidden)
        output_flat = output.view(-1, ntokens)
        if args.modeltype == 'nce':
            total_loss += len(output_flat) * criterion_test(output_flat, targets).data
        else:
            total_loss += len(output_flat) * criterion(output_flat, targets).data
        hidden = repackage_hidden(hidden)
        total_words += len(output_flat)
    return total_loss[0] / total_words


def train(embdict):
    # Turn on training mode which enables dropout.
    model.train()
    model.set_mode('train')
    total_loss = 0
    start_time = time.time()
    ntokens = len(corpus.dictionary)
    hidden = model.init_hidden(args.batch_size)
    for batch, i in enumerate(range(0, train_data.size(0) - 1, args.bptt)):
        data, targets = get_batch(train_data, i)
        data = index2embed(data, embdict, corpus.dictionary, args.emsize)
        # Starting each batch, we detach the hidden state from how it was previously produced.
        # If we didn't, the model would try backpropagating all the way to
        # start of the dataset.
        hidden = repackage_hidden(hidden)
        model.zero_grad()
        if args.modeltype == 'nce':
            output, hidden = model(data, hidden, targets)
            loss = criterion(output)
            loss.backward()
        else:
            output, hidden = model(data, hidden)
            loss = criterion(output.view(-1, ntokens), targets)
            loss.backward()
        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        torch.nn.utils.clip_grad_norm(model.parameters(), args.clip)
        for p in model.parameters():
            p.data.add_(-lr, p.grad.data)

        total_loss += loss.data

        if batch % log_interval == 0 and batch > 0:
            cur_loss = total_loss[0] / log_interval
            elapsed = time.time() - start_time
            if args.modeltype == 'nce':
                logging.info('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.2f} | ms/batch {:5.2f} | '
                             'nce_loss {:5.2f}'.format(
                                 epoch, batch, len(train_data) // args.bptt, lr,
                                 elapsed * 1000 / log_interval, cur_loss))
            else:
                logging.info('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.2f} | ms/batch {:5.2f} | '
                             'loss {:5.2f} | ppl {:8.2f}'.format(
                                 epoch, batch, len(train_data) // args.bptt, lr,
                                 elapsed * 1000 / log_interval, cur_loss, math.exp(cur_loss)))

            total_loss = 0
            start_time = time.time()


# Loop over epochs.
lr = args.lr
best_val_loss = None

# At any point you can hit Ctrl + C to break out of training early.
for epoch in range(1, args.epochs + 1):
    epoch_start_time = time.time()
    train(embdict)
    logging.info('finished train with epoch {:1d}\n'.format(epoch))
    val_loss = evaluate(val_data, embdict, corpus.dictionary)
    logging.info('-' * 89)
    logging.info(
        '| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
        'valid ppl {:8.2f}'.format(
            epoch,
            (time.time() - epoch_start_time),
            val_loss,
            math.exp(val_loss)))
    logging.info('-' * 89)
    # Save the model if the validation loss is the best we've seen so
    # far.
    if not best_val_loss or val_loss < best_val_loss:
        with open(args.save, 'wb') as f:
            torch.save(model, f)
        best_val_loss = val_loss
    else:
        # Anneal the learning rate if no improvement has been seen in
        # the validation dataset.
        lr /= 4.0
