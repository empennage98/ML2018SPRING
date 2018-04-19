import torch
import torch.nn as nn

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

class Ensemble(nn.Module):

    def __init__(self, model1, model2, model3, model4):
        super(Ensemble, self).__init__()

        self.model1 = model1
        self.model2 = model2
        self.model3 = model3
        self.model4 = model4

    def forward(self, x):

        return x
