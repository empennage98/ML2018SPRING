import os
import sys
import pickle
import re
import time

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader

import gensim

from data import TextDataset
from model import LSTMClassifier

def train(epoch, model, criterion, optimizer, dataloader):
    model.train()

    total = 0.0
    total_loss = 0.0
    total_acc = 0.0
    start_time = time.time()

    for idx, data in enumerate(dataloader):
        inputs, labels = data
        inputs = Variable(inputs.cuda()).t()
        labels = Variable(torch.squeeze(labels).cuda())
 
        model.zero_grad()
        model.batch_size = len(labels)
        model.hidden = model.init_hidden()

        output = model(inputs)

        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()

        _, pred = torch.max(output.data, 1)
        total_loss += loss.data[0]
        total_acc += (pred == labels.data).sum()
        total += len(labels)

    print('Epoch: %3d Accuracy: %.3f Training Loss: %.3f Time: %d' %\
         (epoch, total_acc, total_loss / total, time.time() - start_time))

def validate(epoch, model, dataloader):
    model.eval()

    total = 0.0
    total_acc = 0.0

    for data in dataloader:
        inputs, labels = data
        inputs = Variable(inputs.cuda()).t()
        labels = Variable(torch.squeeze(labels).cuda())
 
        model.batch_size = len(labels)
        model.hidden = model.init_hidden()

        output = model(inputs)

        _, pred = torch.max(output.data, 1)
        total_acc += (pred == labels.data).sum()
        total += len(labels)

    print('Epoch: %3d Val Accuracy: %.3f' %\
         (epoch, total_acc))

#os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
#os.environ['CUDA_VISIBLE_DEVICES']='0'

if __name__ == '__main__':
    if len(sys.argv) == 3:
        fp_train_labeled = sys.argv[1]
        fp_train_unlabeled = sys.argv[2]
    else:
        print('Usage: python3 train.py [training label data] [training unlabel data]')
        exit(0)

    torch.manual_seed(42)

    embedding_dim = 128
    hidden_dim = 128
    num_layers = 2
    batch_size = 128

    ### Prepare raw data
    print('Prepare raw data ... ', end='')

    raw = []
    raw += [line.split('+++$+++')[1] for line in open(fp_train_labeled)]
    raw += [line for line in open(fp_train_unlabeled)]

    print('Done !')

    ### Clean data
    print('Clean data ... ', end='')

    for idx, line in enumerate(raw):
        line = line.lower()
        line = re.sub(r'(.)\1{2,}', r'\1\1', line)    
        line = re.sub(r'[^a-z!? ]', '', line)
        raw[idx] = line.split()

    print('Done !')

    ### Build dictionary
    print('Build dictionary ... ', end='')

    dictionary = gensim.corpora.Dictionary(raw)
    dictionary.filter_extremes(no_below=5)
    dictionary.add_documents([['<pad>', '<unk>']])

    print('Done !')

    ### Prepare cooked data
    print('Prepare cooked data ... ', end='')

    cooked = []
    for line in raw:
        cooked_line = line[:40]
        tokens = [dictionary.token2id.get(word, -1)  for word in cooked_line]
        cooked_line += ['<pad>'] * (40 - len(cooked_line))
        for idx, token in enumerate(tokens):
            if token == -1:
                cooked_line[idx] = '<unk>'
        cooked.append(cooked_line)

    print('Done !')

    ### Build word2vec
    print('Build word2vec ... ', end='')

    word2vec = gensim.models.Word2Vec(cooked, size=128, min_count=1, iter=1, workers=4)

    print('Done !')

    ### Rebuild dictionary
    print('Build word2idx ... ', end='')

    word2idx = {}
    for k, v in word2vec.wv.vocab.items():
        word2idx[k] = v.index
    word2vec.wv.syn0[word2idx['<pad>']] = np.zeros(embedding_dim)
    pickle.dump(word2idx, open('_word2vec.pkl', 'wb'))

    print('Done !')

    ### Load dataset
    print('Load dataset ... ', end='')

    d_train = TextDataset(word2idx, fp_train_labeled, train=True)
    d_val = TextDataset(word2idx, fp_train_labeled, train=True, val=True)

    train_loader = DataLoader(d_train, batch_size = batch_size, shuffle=True)
    val_loader = DataLoader(d_val, batch_size = batch_size, shuffle=False)

    ### Train model
    print('Train LSTM ... ')

    model = LSTMClassifier(embedding_dim, hidden_dim, num_layers, batch_size)
    model.init_weights()
    model.embedding.weight = torch.nn.Parameter(torch.Tensor(word2vec.wv.syn0))
    model.embedding.weight.requires_grad = False
    model.cuda()
    print(model)

    criterion = nn.NLLLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()))

    for epoch in range(1):
        train(epoch, model, criterion, optimizer, train_loader)
        validate(epoch, model, val_loader)

    print('Done !')

    ### Save model
    print('Save model ... ', end='')

    torch.save(model.state_dict(), '_model.pt')

    print('Done !')
