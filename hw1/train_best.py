import pickle
import sys

import numpy as np
import pandas as pd

from sklearn import linear_model
from sklearn.preprocessing import QuantileTransformer
from sklearn.externals import joblib

def reformatData(file_path, encoding, config):
    table = pd.read_csv(file_path, encoding=encoding)
    table['日期'] = pd.to_datetime(table['日期'])

    table_feature = dict()
    for key in config:
        table_feature[key] = table.loc[table['測項'] == key]

    mat = dict()
    for key in config:
        mat[key] = table_feature[key].as_matrix(columns=[str(x) for x in range(24)]). \
                    astype(float).flatten()
    
    table_list = [pd.DataFrame( \
        {'Date': pd.date_range('2017-'+str(date)+'-01 00:00:00', '2017-'+str(date)+'-20 23:00:00', freq='H')} \
    ) \
        for date in range(1,13)]
    table_reform = pd.concat(table_list)

    for key in config:
        table_reform[key] = mat[key]
    
    return(table_reform)

def splitData(table, n):
    ubound, lbound = ((n+1)*4, n*4)
    
    train = []
    test = []
    
    if n == 10:
        for month in range(1,13):
            train.append(table[(table['Date'].dt.month == month)])
        return train, test

    for month in range(1,13):
        train.append( \
            table[(table['Date'].dt.month == month) & 
                  (table['Date'].dt.day <= lbound)]
        )
        train.append( \
            table[(table['Date'].dt.month == month) & 
                  (table['Date'].dt.day >  ubound)]
        )
        test.append( \
            table[(table['Date'].dt.month == month) & 
                  (table['Date'].dt.day > lbound) &
                  (table['Date'].dt.day <= ubound)]
        )
    
    return train, test

def preprocessData(train, test, config):
    train_rets = []
    test_rets = []
    
    for train_slice in train:
        train_ret = train_slice.copy()
        for key in config:
            train_ret[train_ret[key] <= config[key][1]] = np.nan
            train_ret[train_ret[key] >= config[key][2]] = np.nan
        train_rets.append(train_ret)
        
    for test_slice in test:
        test_ret = test_slice.copy()
        for key in config:
            test_ret[test_ret[key] <= config[key][1]] = np.nan
            test_ret[test_ret[key] >= config[key][2]] = np.nan
        test_rets.append(test_ret)

    return train_rets, test_rets

def generateTrainingData(train, config, err):
    features = []
    targets = []
    keys = list(config.keys())

    for train_slice in train:
        if train_slice.shape[0] == 0:
            continue

        train_mat = train_slice.as_matrix(columns=keys).transpose()
        t = [config[key][0] for key in config]
        t_max = max(t)
        
        if train_mat.shape[1] < t_max:
            continue
        
        # Index for sliding windows
        indexer_max = np.arange(t_max)[None, :] + np.arange(train_mat.shape[1]-t_max-1)[:, None]
        indexers = [indexer_max[:,-i:] for i in t]

        # Matrix of features and targets
        feature = np.concatenate([train_mat[i][indexers[i]] for i in range(len(t))], axis=1)
        target = train_mat[0][(indexer_max + 1)[:,-1]].reshape(-1,1)

        # NaN filter
        feature_filter = np.isnan(feature).sum(1)
        target_filter = np.isnan(target).sum(1)
        
        # Filtered Data
        feature_clean = feature[(feature_filter <= err) & (target_filter == 0)]
        target_clean = target[(feature_filter <= err) & (target_filter == 0)]

        # Interpolate Data
        if err != 0:
            beg = 0; end = 0
            for key in config:
                beg = end
                end += config[key][0]
                df = pd.DataFrame(feature_clean[:,beg:end], dtype=float)
                df.interpolate(method='linear', inplace=True, axis=1)
                df.fillna(method='bfill', inplace=True, axis=1)
                df.fillna(method='ffill', inplace=True, axis=1)
                feature_clean[:,beg:end] = df.values

        features.append(feature_clean)
        targets.append(target_clean)

    return np.concatenate(features, axis=0), np.concatenate(targets, axis=0).flatten()
 
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

# Note that PM2.5 must be the first one
config = { \
    'PM2.5': [8, 0, 200], \
    'PM10': [8, 0, np.inf], \
    'CO': [4, 0, np.inf], \
    'NOx': [7, 0, np.inf], \
    'SO2': [6, 0, np.inf], \
}

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print('Usage: python3 train_best.py [data.csv] [model_best.pkl]')
        exit(1)

    ### Load Dataset
    print('Loading Data...')
    data = reformatData(sys.argv[1], 'big5', config)

    ### Predict
    train, test = splitData(data, 10)
    train_scale, test_scale = preprocessData(train, test, config)

    Xtrain, ytrain = generateTrainingData(train_scale, config, err=0)

    scaler = QuantileTransformer(output_distribution='uniform')
    train_scale = scaler.fit_transform(Xtrain[:,20:])
    Xtrain_scale = np.c_[Xtrain[:,:20], train_scale]

    # with scikit-learn
    print('Training...')
    regr = linear_model.LinearRegression()
    regr.fit(Xtrain_scale, ytrain)

    ### Saving Model 
    print('Saving model...')
    model = (scaler, regr)
    joblib.dump(model, sys.argv[2])
    print('Done!')
