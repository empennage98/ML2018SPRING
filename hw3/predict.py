import os
import sys
#import urllib

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torchvision import transforms
from torch.autograd import Variable

from PIL import Image

from data import SentimentDataset, load_data
from model import WideResNet, Ensemble

import pandas as pd
import numpy as np

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

    if len(sys.argv) == 3:
        fp_test = sys.argv[1]
        fp_pred = sys.argv[2]
    else:
        print('Usage:')
        print('    python3 predict.py [testing data] [prediction file]')

    ### Download models ###
    #print('Donwloading models ...')

    #for i in range(1, 5):
    #    fp_model = 'model_ensemble.' + str(i) + '.pt'
    #    url_model = 'https://www.csie.ntu.edu.tw/~b05902042/model_ensemble_' + str(i) + '.pt'

    #    model =  urllib.request.urlopen(url_model)

    #    with open(fp_model, 'wb') as f:
    #        f.write(model.read())

    #print('Done!')

    ### Load models ###
    print('Loading models ...')

    ensemble = Ensemble(WideResNet(),WideResNet(),WideResNet(),WideResNet())
    ensemble.load_state_dict(torch.load('model_ensemble.pt'))
    ensemble_w = ensemble.state_dict()

    models = []
    for i in range(1, 5):
        model = WideResNet()
        model_w = model.state_dict()

        weight = {}
        for key, val in model_w.items():
            weight[key] = ensemble_w['model'+str(i)+'.'+key]

        model_w.update(weight)
        model.load_state_dict(model_w)

        model.cuda()
        models.append(model)

    print('Done!')

    ### Load data ###
    print('Loading data ...')

    test_loader = load_data(fp_test, train=False)

    print('Done!')

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
