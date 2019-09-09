# -*- coding:utf-8 -*-

import json
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold


INPUT_DIR = '../input/'
OUTPUT_DIR = '../output/'


def lgb_train(X, y):
    param = {'num_leaves': 8,  #
             'min_data_in_leaf': 2,  #
             'objective': 'binary',
             'max_depth': -1,  #
             'learning_rate': 0.0123,
             'boosting': 'gbdt',
             'bagging_freq': 5,
             'bagging_fraction': 0.8,
             'feature_fraction': 0.8201,
             'bagging_seed': 11,
             'random_state': 42,
             'metric': 'binary',
             'verbosity': -1,
             'num_threads': 8,
             }

    def print_report(true_label, p):
        pred = (p > 0.5).astype(int)
        print(classification_report(true_label, pred))

    # train
    RANDOM_STATE = 42
    NFOLDS = 5
    folds = StratifiedKFold(n_splits=NFOLDS, shuffle=True, random_state=RANDOM_STATE)
    for fold_, (trn_, val_) in enumerate(folds.split(y, y)):
        print('-' * 50, "Current Fold: {}".format(fold_))
        trn_x, trn_y = X[trn_, :], y[trn_]
        val_x, val_y = X[val_, :], y[val_]

        train_data = lgb.Dataset(trn_x, label=trn_y)
        valid_data = lgb.Dataset(val_x, label=val_y)
        clf = lgb.train(param, train_data, num_boost_round=10000, valid_sets=[train_data, valid_data],
                        verbose_eval=300, early_stopping_rounds=400)
        trn_pred = clf.predict(trn_x)
        val_pred = clf.predict(val_x)
        print('*' * 20, 'train report:')
        print_report(trn_y, trn_pred)
        print('*' * 20, 'valid report:')
        print_report(val_y, val_pred)
    clf.save_model(lgb_output + 'lgb_model.txt')


if __name__ == '__main__':
    print('execute lgb_train.py ...')
    lgb_output = OUTPUT_DIR + 'lgb_output/'
    conf = json.load(open(lgb_output + 'conference.json'))
    arxiv = json.load(open(lgb_output + 'arxiv.json'))

    # t[1-4]:figures, tables, formulas, count;
    # 1/0: label
    lst = [t[1] + t[2] + t[3] + [t[4], 1] for t in conf if len(t[1]) != 1]
    lst += [t[1] + t[2] + t[3] + [t[4], 0] for t in arxiv if len(t[1]) != 1]

    # make dataframe
    names = []
    for pre in ['figures', 'tables', 'formulas']:
        for i in range(8):
            names.append(pre + str(i))
    names += ['pagenum', 'label']
    df = pd.DataFrame(lst, columns=names)
    print(df.head())

    # train lightgbm
    y = df.label.values
    df = df.drop(['label'], axis=1)
    X = df.values
    lgb_train(X, y)
