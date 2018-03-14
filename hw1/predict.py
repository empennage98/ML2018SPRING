import sys

import numpy as np
import pandas as pd
from sklearn import linear_model

from GradientDescent import GradientDescent
 
def generateTestData(file_path, config, encoding='big5'):
    table = pd.read_csv(file_path, encoding=encoding, header=None)

    pm25 = table.loc[table[1] == 'PM2.5']
    pm10 = table.loc[table[1] == 'PM10']

    pm25 = pm25.iloc[:,11-config[0]:11].astype(float)
    pm10 = pm10.iloc[:,11-config[1]:11].astype(float)

    pm25[(pm25 <= 0) | (pm25 >= 200)] = np.nan
    pm10[(pm10 <= 0)] = np.nan

    pm25 = pm25.interpolate(method='linear', limit_direction='both', axis=1)
    pm10 = pm10.interpolate(method='linear', limit_direction='both', axis=1)
    
    pm25 = pm25.values
    pm10 = pm10.values
    
    return(np.concatenate([pm25, pm10], axis=1))

if __name__ == '__main__':
    if len(sys.argv) != 4:
        print('Usage: python3 predict.py [test.csv] [model.npy] [ans.csv]')
        exit(0)

    conf = (7,7)

    print("Loading data...")
    Xtest = generateTestData(sys.argv[1], conf, encoding=None)
    w = np.load(sys.argv[2])

    # Regress with gradient descent
    print("Predicting...")
    ypred = np.c_[np.ones((Xtest.shape[0], 1)), Xtest].dot(w)

    # Output results
    print("Saving results...")
    pred = pd.DataFrame()
    pred['id'] = ['id_'+str(x) for x in range(len(ypred))]
    pred['value'] = ypred

    pred.to_csv(sys.argv[3], index=False)
    print("Done!")
