import re

import numpy as np

import torch
from torch.utils.data.dataset import Dataset

class TextDataset(Dataset):
    def __init__(self, word2idx, fp_dataset, train=True, val=False, split=True):
        self.word2idx = word2idx
        self.train = train
        self.val = val

        if train:
            self.text = []
            self.label = []
            for line in open(fp_dataset):
                label, line = line.split('+++$+++')
                line = line.lower()
                line = re.sub(r'(.)\1{2,}', r'\1\1', line)    
                line = re.sub(r'[^a-z!? ]', '', line)
                self.text.append(line.split())
                self.label.append(int(label))

            if split:
                r = np.random.RandomState(42)
                idx = r.permutation(200000)
                if val:
                    self.text = [self.text[i] for i in idx[:10000]]
                    self.label = [self.label[i] for i in idx[:10000]]
                else:
                    self.text = [self.text[i] for i in idx[10000:]]
                    self.label = [self.label[i] for i in idx[10000:]]
        else:
            self.text = []
            for line in open(fp_dataset):
                line = line.split(',', 1)[1]
                line = line.lower()
                line = re.sub(r'(.)\1{2,}', r'\1\1', line)    
                line = re.sub(r'[^a-z!? ]', '', line)
                self.text.append(line.split())

    def __getitem__(self, index):
        line = self.text[index]
        tokens = []
        for idx, word in enumerate(line):
            if word in self.word2idx:
                tokens.append(self.word2idx[word])
        tokens = tokens[:40]
        tokens += [self.word2idx['<pad>']] * (40 - len(tokens))

        if self.train:
            return torch.LongTensor(tokens), torch.LongTensor([self.label[index]])
        else:
            return torch.LongTensor(tokens)

    def __len__(self):
        return len(self.text)
