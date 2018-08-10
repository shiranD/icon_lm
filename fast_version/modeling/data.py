import os
import torch


class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []
        self.unigram = []

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
            self.unigram.append(0)
        self.unigram[self.word2idx[word]] += 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)

    def __getitem__(self, idx):
        return self.idx2word[idx]

    def to_idx(self, word):
        return self.word2idx[word]


class Corpus(object):
    def __init__(self, path):
        self.dictionary = Dictionary()
        # assert input path validity
        assert os.path.exists(os.path.dirname(
            path + "train")), "%r is not a valid path" % path + "train"
        assert os.path.exists(os.path.dirname(
            path + "valid")), "%r is not a valid path" % path + "valid"
        assert os.path.exists(os.path.dirname(
            path + "test")), "%r is not a valid path" % path + "test"
        self.train = self.tokenize(path + 'train')
        self.valid = self.tokenize(path + 'valid')
        self.test = self.tokenize(path + 'test')

    def tokenize(self, path):
        """
        Tokenizes a text file
        and represent it as indecies
        """
        assert os.path.exists(path)
        # Add words to the dictionary
        with open(path, 'r') as f:
            tokens = 0
            for line in f:
                words = line.split() + ['>']
                tokens += len(words)
                for word in words:
                    self.dictionary.add_word(word)

        # help bring back to sentences

        # Tokenize file content
        with open(path, 'r') as f:
            ids = torch.LongTensor(tokens)
            token = 0
            for line in f:
                words = line.split() + ['>']
                # print words
                for word in words:
                    ids[token] = self.dictionary.word2idx[word]
                    token += 1
        return ids
