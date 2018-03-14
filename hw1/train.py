import sys

import numpy as np
import pandas as pd
from sklearn import linear_model

from GradientDescent import GradientDescent

def reformatData(file_path, encoding='big5'):
    table = pd.read_csv(file_path, encoding=encoding)
    table['日期'] = pd.to_datetime(table['日期'])
    
    pm25 = table.loc[table['測項'] == 'PM2.5']
    pm10 = table.loc[table['測項'] == 'PM10']

    pm25_mat = pm25.as_matrix(columns=[str(x) for x in range(24)]).astype(float).flatten()
    pm10_mat = pm10.as_matrix(columns=[str(x) for x in range(24)]).astype(float).flatten()
    
    table_list = [pd.DataFrame( \
        {'Date': pd.date_range('2017-'+str(date)+'-01 00:00:00', '2017-'+str(date)+'-20 23:00:00', freq='H')} \
    ) \
        for date in range(1,13)]
    table_reform = pd.concat(table_list)
     
    table_reform['PM2.5'] = pm25_mat
    table_reform['PM10'] = pm10_mat
    
    return(table_reform)

def splitData(table, n):
    ubound, lbound = ((n+1)*2, n*2)
    
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

def preprocessData(train, test):
    
    train_rets = []
    test_rets = []
    
    for train_slice in train:
        train_ret = train_slice.copy()
        train_ret[(train_ret['PM2.5'] >= 200) | (train_ret['PM2.5'] <= 0)] = np.nan
        train_ret[(train_ret['PM10'] <= 0)] = np.nan
        train_rets.append(train_ret)
        
    for test_slice in test:
        test_ret = test_slice.copy()
        test_ret[(test_ret['PM2.5'] >= 200) | (test_ret['PM2.5'] <= 0)] = np.nan
        test_ret[(test_ret['PM10'] <= 0)] = np.nan
        test_rets.append(test_ret)

    return train_rets, test_rets

'''
        avg = {'PM2.5': np.nanmean(train_ret['PM2.5'].as_matrix()),\
               'PM10': np.nanmean(train_ret['PM10'].as_matrix())}
        std = {'PM2.5': np.nanstd(train_ret['PM2.5'].as_matrix()),\
               'PM10': np.nanstd(train_ret['PM10'].as_matrix())}

        train_ret['PM2.5'] = (train_ret['PM2.5'] - avg['PM2.5']) / std['PM2.5']
        train_ret['PM10'] = (train_ret['PM10'] - avg['PM10']) / std['PM10']
        
        test_ret['PM2.5'] = (test_ret['PM2.5'] - avg['PM2.5']) / std['PM2.5']
        test_ret['PM10'] = (test_ret['PM10'] - avg['PM10']) / std['PM10']
'''

def generateTrainingData(train, t, err):
    features = []
    targets = []

    for train_slice in train:
        train_mat = train_slice.as_matrix(columns=['PM2.5','PM10']).transpose()
        
        t_max = max(t)
        
        if train_mat.shape[1] < t_max:
            continue
        
        # Index for sliding windows
        indexer_max = np.arange(t_max)[None, :] + np.arange(train_mat.shape[1]-t_max-1)[:, None]
        indexer = [ \
            indexer_max[:,-t[0]:], indexer_max[:,-t[1]:] \
        ]

        # Matrix of features and targets
        feature = np.concatenate([ \
            train_mat[0][indexer[0]],\
            train_mat[1][indexer[1]]
        ], axis=1)
        target = train_mat[0][(indexer_max + 1)[:,-1]].reshape(-1,1)

        # NaN filter
        feature_filter = np.isnan(feature).sum(1)
        target_filter = np.isnan(target).sum(1)
        
        # Filtered Data
        feature_clean = feature[(feature_filter <= err) & (target_filter == 0)]
        target_clean = target[(feature_filter <= err) & (target_filter == 0)]

        # Interpolate Data
        if err != 0:
            df = pd.DataFrame(feature_clean, dtype=float)
            df.interpolate(method='linear', limit_direction='both', inplace=True, axis=1)
            #df.fillna(method='bfill', inplace=True, axis=1)
            #df.fillna(method='ffill', inplace=True, axis=1)
            feature_clean = df.values

        features.append(feature_clean)
        targets.append(target_clean)

    return np.concatenate(features, axis=0), np.concatenate(targets, axis=0).flatten()
 
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
    
    #pm10.to_csv('out.csv', index=False)
    
    pm25 = pm25.values
    pm10 = pm10.values
    
    return(np.concatenate([pm25, pm10], axis=1))

'''
### Cross validation
for config in itertools.product(range(3,10),range(3,10)):
    scores = []
    for cv in range(10):
        train, test = splitData(data, cv)
        train_scale, test_scale = preprocessData(train, test)
        
        Xtrain, ytrain = generateTrainingData(train_scale, config, err=0)
        Xtest, ytest = generateTrainingData(test_scale, config, err=2)
        
        print(Xtest.shape)
        
        regr = linear_model.LinearRegression()
        regr.fit(Xtrain, ytrain)
        #print(regr.coef_, regr.intercept_)
        #print(regr.score(Xtrain, ytrain))
        ypred = regr.predict(Xtest)
        
        scores.append(np.sqrt(np.mean((ypred - ytest) ** 2)))
        
    print(config, np.mean(scores), np.std(scores))
    #print(scores)
'''

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print('Usage: python3 train.py [data.csv] [model.npy]')
        exit(0)

    ### Predict
    print("Loading data...")
    data = reformatData(sys.argv[1], encoding='big5')

    train, test = splitData(data, 10)
    train_scale, test_scale = preprocessData(train, test)

    conf = (7,7)

    Xtrain, ytrain = generateTrainingData(train_scale, conf, err=0)

    # Regress with gradient descent
    print("Training...")
    regr = GradientDescent()
    regr.fit(Xtrain, ytrain)

    print("Saving model...")
    np.save(sys.argv[2], regr.w)
    print("Done!")
'''
    Xtest = generateTestData('test.csv', conf, encoding=None)

    # with scikit-learn
    regr = linear_model.LinearRegression()
    regr.fit(Xtrain, ytrain)
    ypred = regr.predict(Xtest)

    #print(ypred)

    pred = pd.DataFrame()
    pred['id'] = ['id_'+str(x) for x in range(len(ypred))]
    pred['value'] = ypred

    #pred.to_csv('pred.csv', index=False)
'''
