import numpy as np
import torch
from torch.autograd import Variable

def sym2vec(data, embd, iconpath=""):
    """
    load embeddings and if augment=True
    load icon embedding for the same dir
    and prioritize them
    """
    edict = {}
    for h, line in enumerate(open(embd, 'r').readlines()):
        line = line.strip()
        line = line.split()
        word = line[0]
        vector = [float(item) for item in line[1:]]
        edict[word] = torch.FloatTensor(vector)

    ln = len(vector)
    if iconpath:
        # override original terms w icon terms
        for h, line in enumerate(open(iconpath, 'r').readlines()):
            line = line.strip()
            line = line.split()
            word = line[1]
            vector = [float(item) for item in line[2:]]
            if len(vector) == ln:
                edict[word] = torch.FloatTensor(vector)
            else:
                print("length problem")
    return edict


def index2embed(sequence, term2vec, int2term, dim):
    """
    Replace index with embedding 
    """
    # Tokenize file content
    seq = []
    row, col = sequence.size()
    new_seq = sequence.view(row * col, -1).data.cpu().numpy()
    new = np.zeros((row * col, dim), dtype=float)
    for i, integer in enumerate(new_seq):
        term = int2term[int(integer)]
        vec = term2vec[term].numpy()
        new[i, :] = vec
    try:
        new = torch.FloatTensor(new).cuda()
    except:
        new = torch.FloatTensor(new)
    new = Variable(new)
    new = new.view(row, col, dim)
    return new


def term2sym(data, embd, iconpath=""):
    """
    load embeddings and if augment=True
    load icon embedding for the same dir
    and prioritize them
    """
    edict = {}
    for h, line in enumerate(open(embd, 'r').readlines()):
        line = line.strip()
        line = line.split()
        word = line[0]
        edict[word] = word

    if iconpath:
        # override original terms w icon terms
        for h, line in enumerate(open(iconpath, 'r').readlines()):
            line = line.strip()
            line = line.split()
            term = line[0]
            sym = line[1]
            edict[term] = sym
    return edict
