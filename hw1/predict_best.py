import sys

import numpy as np
import pandas as pd

from sklearn import linear_model
from sklearn.preprocessing import QuantileTransformer
from sklearn.externals import joblib
 
def generateTestData(file_path, encoding, config):
    table = pd.read_csv(file_path, encoding=encoding, header=None)

    table_feature = dict()
    for key in config:
        table_feature[key] = table.loc[table[1] == key].\
                iloc[:,2:11].astype(float)

    for key in config:
        table_feature[key][table_feature[key] <= config[key][1]] = np.nan
        table_feature[key][table_feature[key] >= config[key][2]] = np.nan

        table_feature[key].interpolate(method='linear', inplace=True, axis=1)
        table_feature[key].fillna(method='bfill', inplace=True, axis=1)
        table_feature[key].fillna(method='ffill', inplace=True, axis=1)

        table_feature[key][np.isnan(table_feature[key])] = 0

    for key in config:
        table_feature[key] = np.delete(table_feature[key].as_matrix(), np.s_[:-config[key][0]:], 1)
    
    return(np.concatenate([table_feature[key] for key in config], axis=1))

config = {
    'PM2.5': [8, 0, 200],
    'PM10': [8, 0, np.inf],
    'CO': [4, 0, np.inf],
    'NOx': [7, 0, np.inf],
    'SO2': [6, 0, np.inf]
}

if __name__ == '__main__':
    if len(sys.argv) != 4:
        print('Usage: python3 predict_best.py [test.csv] [model_best.pkl] [ans.csv]')
        exit(1)

    print("Loading data...")
    model = joblib.load(sys.argv[2])

    Xtest = generateTestData(sys.argv[1], 'big5', config)
    test_scale = model[0].transform(Xtest[:,20:])
    Xtest_scale = np.c_[Xtest[:,:20], test_scale]

    print("Predicting...")
    ypred = model[1].predict(Xtest_scale)

    # Output results
    print("Saving results...")
    pred = pd.DataFrame()
    pred['id'] = ['id_'+str(x) for x in range(len(ypred))]
    pred['value'] = ypred

    pred.to_csv(sys.argv[3], index=False)
    print("Done!")
