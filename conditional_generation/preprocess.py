import os
import argparse

from data_utils import *

parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str, default="./text_data", help="Data directory before tokenize")
parser.add_argument("--from_vocab", type=str, default="data/vocab_20000",help="from vocab path")
args = parser.parse_args()
data_path = args.data_path

class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.word2idx['_PAD'] = 0
        self.word2idx['_GO'] = 1
        self.word2idx['_EOS'] = 2
        self.word2idx['_UNK'] = 3
        self.wordcounts = {}

    # to track word counts
    def add_word(self, word):
        if word not in self.wordcounts:
            self.wordcounts[word] = 1
        else:
            self.wordcounts[word] += 1

    # prune vocab based on count k cutoff or most frequently seen k words
    def prune_vocab(self, k=5, cnt=False):
        # get all words and their respective counts
        vocab_list = [(word, count) for word, count in self.wordcounts.items()]
        if cnt:
            # prune by count
            self.pruned_vocab = \
                [pair[0] for pair in vocab_list if pair[1] > k]
        else:
            # prune by most frequently seen words
            vocab_list.sort(key=lambda x: (x[1], x[0]), reverse=True)
            k = min(k, len(vocab_list))
            self.pruned_vocab = [pair[0] for pair in vocab_list[:k]]
        # sort to make vocabulary determistic
        self.pruned_vocab.sort()

        # add all chosen words to new vocabulary/dict
        for word in self.pruned_vocab:
            if word not in self.word2idx:
                self.word2idx[word] = len(self.word2idx)
        print("original vocab {}; pruned to {}".
              format(len(self.wordcounts), len(self.word2idx)))
        self.idx2word = {v: k for k, v in self.word2idx.items()}

    def __len__(self):
        return len(self.word2idx)


class Corpus(object):
    def __init__(self, path, maxlen, threshold=11000, lowercase=False):
        self.train_path = os.path.join(path, 'train.txt')
        self.valid_path = os.path.join(path, 'valid.txt')
        self.test_path = os.path.join(path, 'test.txt')
        # load the vocab dictonary from "./data/vocab_20000"
        self.make_vocab()
        if (os.path.exists(self.train_path) and os.path.exists(self.valid_path) and os.path.exists(self.test_path)):
            self.train = self.tokenize(self.train_path)
            self.valid = self.tokenize(self.valid_path)
            self.test = self.tokenize(self.test_path)
        else:
            raise ValueError("Either {train/valid/test}.txt not exist ")

    def make_vocab(self):

        to_vocab, rev_to_vocab = initialize_vocabulary("data/vocab_20000")
        self.dictionary = to_vocab

    def tokenize(self, path):
        """Tokenizes a text file."""
        with open(path, 'r') as f:
            linecount = 0
            lines = []
            for line in f:
                linecount += 1
                sentences = line.strip().split(".")
                # Since 5 sentences seperate by "." will result to 6 partitions
                if (len(sentences)!= 6):
                    continue
                indices= []
                unk_idx = self.dictionary['_UNK']
                for i in range (0,5):
                    words = sentences[i].lower().strip().split(" ")
                    indices.extend([self.dictionary[w] if w in self.dictionary else unk_idx for w in words])
                    indices.extend([4,-1])
                lines.append(indices)
        return lines

# create corpus
corpus = Corpus(data_path,
                maxlen=100,
                threshold=0,
                lowercase=True)

# # write to file
with open(f"{data_path}/train_test.ids", "w") as f:
    for line in corpus.train:
        f.write(' '.join(map(str, line)) + '\n')

with open(f"{data_path}/valid_test.ids", "w") as f:
    for line in corpus.valid:
        f.write(' '.join(map(str, line)) + '\n')

with open(f"{data_path}/test_test.ids", "w") as f:
    for line in corpus.test:
        f.write(' '.join(map(str, line)) + '\n')
