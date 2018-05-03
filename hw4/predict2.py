import os
import sys

import numpy as np
import pandas as pd

if __name__ == '__main__':

    if len(sys.argv) != 4:
        print('Usage: python predict.py [image.npy] [test_case.csv] [ans.csv]')
        exit(0) 
    else:
        fp_data = sys.argv[1]
        fp_ind = sys.argv[2]
        fp_ans = sys.argv[3]
        
features = np.load('features.npy')

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
