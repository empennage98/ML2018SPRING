import sys

import numpy as np
import pandas as pd

from model_tree import RandomForest

def preprocess(df_train, df_test, features):
    df_train = df_train.copy()
    df_test = df_test.copy()

    ytrain = df_train['income'].values
    df_train = df_train.drop('income', axis=1)

    # Drop fnlwgt
    df_train.drop('fnlwgt', axis=1, inplace=True)
    df_test.drop('fnlwgt', axis=1, inplace=True)

    # Binarize capital loss since it is too sparse
    df_train.loc[df_train['capital_loss'] != 0, 'capital_loss'] = 1
    df_test.loc[df_test['capital_loss'] != 0, 'capital_loss'] = 1

    numericals = [f for f, t in features.items() if t == 'numerical']
    categoricals = [f for f, t in features.items() if t == 'categorical']

    # One-hot encode the categorical features
    for cat in categoricals:
        df_train[cat] = df_train[cat].astype('category')
        df_test[cat] = df_test[cat].astype('category')

    df = pd.concat(objs=[df_train, df_test], axis=0)
    df = pd.get_dummies(df, columns=categoricals, prefix=categoricals)
    split = len(df_train)
    df_train = df[:split].copy()
    df_test = df[split:].copy()

    # Scale the numerical features
    lbound = df_train[numericals].quantile(.05)
    ubound = df_train[numericals].quantile(.95)

    df_train[numericals] -= lbound
    df_train[numericals] /= ubound

    df_test[numericals] -= lbound
    df_test[numericals] /= ubound

    Xtrain = df_train.values
    Xtest = df_test.values

    return Xtrain, ytrain, Xtest

if __name__ == '__main__':
    if len(sys.argv) != 4:
        print('Usage:')
        print('  python train_logistic.py logistic [train.csv] [test.csv] [ans.csv]')
        exit(0)

    fp_train, fp_test, fp_ans = sys.argv[1:]
    
    features = {
        'age':              'numerical',
        'workclass':        'categorical',
    #   'fnlwgt':           'numerical',
        'education':        'categorical',
        'edu_num':          'numerical',
        'marital_status':   'categorical',
        'occupation':       'categorical',
        'relationship':     'categorical',
        'race':             'categorical',
        'sex':              'categorical',
        'capital_gain':     'numerical',
        'capital_loss':     'categorical',
        'hours_per_week':   'numerical',
        'native_country':   'categorical',
        'income':           'output'
    }

    ####################################
    # Read and preprocess data from files
    ####################################
    df_train = pd.read_csv(fp_train, skipinitialspace=True)
    df_test = pd.read_csv(fp_test, skipinitialspace=True)

    df_train['income'].replace('<=50K', 0, inplace=True)
    df_train['income'].replace('>50K', 1, inplace=True)

    Xtrain, ytrain, Xtest = preprocess(df_train, df_test, features)

    ####################################
    # Train the estimator and predict test data
    ####################################
    regr = RandomForest(n_estimators=60, min_samples_split=6, min_samples_leaf=2,
                max_features='auto', max_depth=80, tree_type='decision')
    regr.fit(Xtrain, ytrain)
    ypred = regr.predict(Xtest)

    ####################################
    # Write the result to file
    ####################################
    df_pred = pd.DataFrame()
    df_pred['id'] = np.arange(1,len(ypred)+1)
    df_pred['label'] = ypred
    df_pred.to_csv(fp_ans, index=False)
