import sys
import pickle

import numpy as np
import pandas as pd

import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader

from data import TextDataset
from model import LSTMClassifier

def predict(model, dataloader):

    model.eval()
    pred = []

    for idx, data in enumerate(dataloader):
        inputs = data
        inputs = Variable(inputs.cuda()).t()
 
        model.batch_size = data.size(0)
        model.hidden = model.init_hidden()

        output = model(inputs)

        _, predict = torch.max(output.data, 1)
        pred.append(predict.cpu().numpy())

    return np.concatenate(pred)

#os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
#os.environ['CUDA_VISIBLE_DEVICES']='0'

if __name__ == '__main__':

    if len(sys.argv) == 3:
        fp_test = sys.argv[1]
        fp_pred = sys.argv[2]
    else:
        print('Usage: python3 predict.py [testing data] [prediction file]')
        exit(0)

    fp_word2idx = 'word2idx.pkl'
    fp_model = 'model.pt'

    embedding_dim = 128
    hidden_dim = 128
    num_layers = 2
    batch_size = 128

    ### Load dictionary
    print('Loading Dictionary ... ', end='')

    word2idx = pickle.load(open(fp_word2idx, 'rb'))

    print('Done !')

    ### Load data
    print('Loading Data ... ', end='')

    d_test = TextDataset(word2idx, fp_test, train=False)
    test_loader = DataLoader(d_test, batch_size=batch_size, shuffle=False)

    print('Done !')

    ### Load model
    print('Loading Model ... ', end='')

    model = LSTMClassifier(embedding_dim, hidden_dim, num_layers, batch_size)
    model.cuda()
    model.load_state_dict(torch.load(fp_model))

    print('Done !')

    ### Predict
    print('Predict ... ', end='')

    pred = predict(model, test_loader)

    print('Done !')

    ### Write
    print('Write ... ', end='')

    df_pred = pd.DataFrame()
    df_pred['id'] = np.arange(len(pred)-1)
    df_pred['label'] = pred[1:]

    df_pred.to_csv(fp_pred, index=False)

    print('Done !')
