import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torchvision import transforms
from torch.autograd import Variable

from PIL import Image

import pandas as pd
import numpy as np

class WideBlock(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1, downsample=False, dropout=0):
        super(WideBlock, self).__init__()
        self.block1 = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3,
                      stride=stride, padding=1),
            #nn.Dropout2d(dropout),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3,
                      stride=1, padding=1),
        )

        if downsample:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                          stride=stride),
            )
        else:
            self.downsample = None

    def forward(self, x):
        out = self.block1(x)

        if self.downsample is not None:
            res = self.downsample(out)
        else:
            res = x

        out = self.block2(out)
        
        return out + res

class WideResNet(nn.Module):

    def __init__(self):
        super(WideResNet, self).__init__()

        self.k = 8

        self.init = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
        )

        self.layer1 = nn.Sequential(
            WideBlock(16, 16*self.k, 2, True),
            WideBlock(16*self.k, 16*self.k, 1, False, .3),
            WideBlock(16*self.k, 16*self.k, 1, False, .3)
        )

        self.layer2 = nn.Sequential(
            WideBlock(16*self.k, 32*self.k, 2, True),
            WideBlock(32*self.k, 32*self.k, 1, False, .3),
            WideBlock(32*self.k, 32*self.k, 1, False, .3)
        )

        self.layer3 = nn.Sequential(
            WideBlock(32*self.k, 64*self.k, 2, True),
            WideBlock(64*self.k, 64*self.k, 1, False, .3),
            WideBlock(64*self.k, 64*self.k, 1, False, .3)
        )

        self.end = nn.Sequential(
            nn.BatchNorm2d(64*self.k),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(6, stride=1)
        )

        self.fc = nn.Sequential(
            nn.Linear(64*self.k, 7),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        x = self.init(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.end(x)
        x = x.view(-1, 64*self.k)
        x = self.fc(x)

        return x

class SentimentDataset(torch.utils.data.Dataset):
    def __init__(self, fp, height, width,
                 preprocess=True, train=True, transforms=None):

        self.fp = fp
        self.height = height
        self.width = width
        self.train = train
        self.transforms = transforms

        if preprocess == True:
            self.__preprocess__()

        if train:
            self.train_data, self.train_labels = torch.load('train.pt')
        else:
            self.test_data, self.test_labels = torch.load('test.pt')

    def __preprocess__(self):
        print('Preprocessing...')

        if self.train:
            self.df_train = pd.read_csv(self.fp, skiprows=1, header=None,
                                sep=' |,', engine='python')

            self.train_labels = torch.from_numpy(self.df_train.iloc[:, 0].values)
            self.train_images = torch.from_numpy(self.df_train.iloc[:, 1:].values\
                                .astype('uint8')\
                                .reshape(-1, self.height, self.width))

            with open('train.pt', 'wb') as f:
                torch.save((self.train_images, self.train_labels), f)

        else:
            self.df_test = pd.read_csv(self.fp, skiprows=1, header=None,
                                    sep=' |,', engine='python')

            self.test_ids = torch.from_numpy(self.df_test.iloc[:, 0].values)
            self.test_images = torch.from_numpy(self.df_test.iloc[:, 1:].values\
                                .astype('uint8')\
                                .reshape(-1, self.height, self.width))

            with open('test.pt', 'wb') as f:
                torch.save((self.test_images, self.test_ids), f)

        print('Done!')

    
    def __getitem__(self, index):

        if self.train:
            img, label = self.train_data[index], self.train_labels[index]
        else:
            img, label = self.test_data[index], self.test_labels[index]

        img = Image.fromarray(img.numpy(), mode='L')

        if self.transforms is not None:
            img = self.transforms(img)

        return img, label

    def __len__(self):

        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)

def load_data(fp, train=True):

    if train:
        train_dataset = \
            SentimentDataset(fp, 48, 48, preprocess=True, train=True,
                transforms=transforms.Compose([
                    transforms.RandomResizedCrop(48, scale=(0.9, 1), ratio=(0.9, 1.1)),
                    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomCrop(size=46, padding=2),
                    transforms.ToTensor()
            ]))

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64,
                                                   shuffle=True, num_workers=2)

        return train_loader

    else:
        test_dataset = \
            SentimentDataset(fp, 48, 48, preprocess=True, train=False,
                transforms=transforms.Compose([
                    transforms.TenCrop(46),
                    transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
            ]))

        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=256,
                                                  shuffle=False, num_workers=2)

        return test_loader

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

def predict(models, dataloader):

    pred = []

    for model in models:
        model.eval()
    
    for data, target in dataloader:
        data = Variable(data, volatile=True).cuda()

        bs, ncrops, c, h, w = data.size()
        output_avg = None

        for model in models:
            output = model(data.view(-1, c, h, w))
            if output_avg is None:
                output_avg = output.view(bs, ncrops, -1).mean(1)
            else:
                output_avg += output.view(bs, ncrops, -1).mean(1)

        _, predict = torch.max(output_avg.data, 1)

        pred.append(predict.cpu().numpy())

    return np.concatenate(pred)

### CUDA environment ###
#os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
#os.environ['CUDA_VISIBLE_DEVICES']='0'

if __name__ == '__main__':

    if len(sys.argv) == 4:
        fp_train = sys.argv[1]
        fp_test = sys.argv[2]
        fp_pred = sys.argv[3]
    else:
        exit(0)

    ### Load data ###
    print('Loading data ...')

    train_loader = load_data(fp_train, True)
    test_loader = load_data(fp_test, False)

    print('Done!')

    ### Building model ###

    models = []
    for i in range(4):
        model = WideResNet()
        model.cuda()

        optimizer = optim.SGD(model.parameters(), lr=.1, momentum=.9, nesterov=True)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [60, 120, 180], 0.2)

        num_epochs = 180

        for epoch in range(num_epochs):
            scheduler.step()
            train(model, optimizer, epoch, train_loader)

        torch.save(model.state_dict(), fp_model + str(i) + '.' + str(epoch+1) + '.pt')

    ### Predict data ###
    print('Predicting ... ')

    pred = predict(models, test_loader)

    print('Done!')

    ### Write to file ###
    print('Writing to file ...')

    df_pred = pd.DataFrame()
    df_pred['id'] = np.arange(len(pred))
    df_pred['label'] = pred

    df_pred.to_csv(fp_pred, index=False)

    print('Done!')
