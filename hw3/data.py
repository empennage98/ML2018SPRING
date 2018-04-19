import os

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torchvision import transforms
from torch.autograd import Variable

from PIL import Image

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
                    #transforms.RandomRotation(degrees=30, resample=Image.BILINEAR),
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
