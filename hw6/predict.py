import sys

import numpy as np
import pandas as pd

import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader

from data import UserRating
from model import MatrixFactorization

def predict(model, dataloader):
    model.eval()

    preds = []
    for data in dataloader:
        users, items = data
        users = Variable(torch.LongTensor(users))
        items = Variable(torch.LongTensor(items))

        pred = model(users, items)
        pred = np.clip(pred.data.numpy() * 5 + 1, 1, 5)
        preds.append(pred)

    return np.concatenate(preds)

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print('Usage: python3 predict.py [test.csv] [ans.csv]')
        exit(1)
    else:
        fp_test = sys.argv[1]
        fp_ans = sys.argv[2]

    torch.manual_seed(42)

    n_users = 6041
    n_items = 3953
    n_factors = 256
    batch_size = 1024

    ### Load testing data
    print('Load testing data ... ', end='')

    pd_test = pd.read_csv(fp_test)
    test_loader = DataLoader(UserRating(pd_test, mode='test'),
        batch_size=batch_size, shuffle=False)

    print('Done!')

    ### Load model
    print('Load model ... ', end='')

    model = MatrixFactorization(n_users, n_items, n_factors)
    model.load_state_dict(torch.load('model.pt'))

    print('Done!')

    ### Produce prediction
    print('Produce prediction ... ', end='')

    pred = predict(model, test_loader)

    df_pred = pd.DataFrame()
    df_pred['TestDataID'] = np.arange(1,len(pred)+1)
    df_pred['Rating'] = pred

    df_pred.to_csv(fp_ans, index=False)

    print('Done!')
