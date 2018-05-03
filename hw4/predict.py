import os
import sys

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import transforms

from torch.autograd import Variable
from PIL import Image
from model import AutoEncoder, Encoder

class Dataset(torch.utils.data.Dataset):
    def __init__(self, fp_data, height, width, transforms=None):

        self.fp_data = fp_data
        self.height = height
        self.width = width
        self.transforms = transforms

        data = np.load(fp_data)

        self.train_data = torch.from_numpy(data\
                            .astype('uint8')\
                            .reshape(-1, height, width))

    def __getitem__(self, index):

        img = self.train_data[index]
        img = Image.fromarray(img.numpy(), mode='L')
        img = self.transforms(img)

        return img

    def __len__(self):

        return len(self.train_data)

def load_data(fp_data):

    train_dataset = Dataset(fp_data, 28, 28,
                        transforms=transforms.Compose([
                            transforms.ToTensor(),
                    ]))

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=4096,
                        shuffle=False, num_workers=4)

    return train_loader

def predict(model, dataloader):

    model.eval()
    pred = []

    for i, data in enumerate(dataloader):
        data = data.view(data.size(0), -1)
        data = Variable(data, volatile=True).cuda()

        output = model(data)
        pred.append(output.data.cpu().numpy())

    return np.concatenate(pred)

#os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
#os.environ['CUDA_VISIBLE_DEVICES']='0'

if __name__ == '__main__':

    if len(sys.argv) != 4:
        print('Usage: python predict.py [image.npy] [test_case.csv] [ans.csv]')
        exit(0) 
    else:
        fp_data = sys.argv[1]
        fp_ind = sys.argv[2]
        fp_ans = sys.argv[3]
        
fp_model_fe='model6.fe.pt'

state_dict = torch.load(fp_model_fe)

model_enc = Encoder()
model_enc_dict = model_enc.state_dict()
model_enc_dict.update({k: v for k, v in state_dict.items() \
                            if k in model_enc_dict})
model_enc.load_state_dict(model_enc_dict)
model_enc.cuda()

test_loader = load_data(fp_data)
features = predict(model_enc, test_loader)

ind = (pd.read_csv(fp_ind, delimiter=',').values)[:,1:]

pred = []
for i in range(ind.shape[0]):
    if np.linalg.norm(features[ind[i][0]] - features[ind[i][1]]) > 10:
        pred.append(0)
    else:
        pred.append(1)

df_pred = pd.DataFrame()
df_pred['ID'] = np.arange(len(pred))
df_pred['Ans'] = pred

df_pred.to_csv(fp_ans, index=False)
