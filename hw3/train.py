import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torchvision import transforms
from torch.autograd import Variable

from PIL import Image

from data import SentimentDataset, load_data
from model import WideResNet

import pandas as pd
import numpy as np

def train(model, optimizer, epoch, dataloader):

    model.train()

    for batch_idx, (data, target) in enumerate(dataloader):
        data = Variable(data).cuda()
        target = Variable(target).cuda()

        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)

        loss.backward()
        optimizer.step()

    print('Train Epoch: {} \tLoss: {:.6f}'.format(epoch, loss.data[0]))  

### CUDA environment ###
#os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
#os.environ['CUDA_VISIBLE_DEVICES']='0'

if __name__ == '__main__':

    if len(sys.argv) == 2:
        fp_train = sys.argv[1]
    else:
        print('Usage:')
        print('    python3 train.py [training data]')

    ### Load data ###
    print('Loading data ...')

    train_loader = load_data(fp_train)

    print('Done!')

    ### Building model ###

    model = WideResNet()
    model.cuda()

    optimizer = optim.SGD(model.parameters(), lr=.1, momentum=.9, nesterov=True)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [60, 120, 180], 0.2)

    num_epochs = 180

    for epoch in range(num_epochs):
        scheduler.step()
        train(model, optimizer, epoch, train_loader)

    torch.save(model.state_dict(), fp_model + '.' + str(epoch+1) + '.pt')
