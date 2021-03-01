'''
Author: xiaoyao jiang
LastEditors: Peixin Lin
Date: 2020-08-31 14:19:30
LastEditTime: 2021-01-03 21:36:09
FilePath: /JD_NLP1-text_classfication/model.py
Desciption:
'''
import json
import joblib
import xgboost as xgb
import pandas as pd
import numpy as np
import os
import sklearn.metrics as metrics

from embedding import Embedding
from features import (get_basic_feature, get_embedding_feature,
                      get_lda_features, get_tfidf)


class Classifier:
    def __init__(self, train_mode=False) -> None:
        self.stopWords = [
            x.strip() for x in open('./data/stopwords.txt').readlines()
        ]
        self.embedding = Embedding()
        self.embedding.load()
        self.labelToIndex = json.load(
            open('./data/label2id.json', encoding='utf-8'))
        self.ix2label = {v: k for k, v in self.labelToIndex.items()}
        if train_mode:
            self.train = pd.read_csv('./data/train.csv',
                                     sep='\t').dropna().reset_index(drop=True)
            #self.dev = pd.read_csv('./data/eval.csv',
            #                       sep='\t').dropna().reset_index(drop=True)
        else:
            self.test = pd.read_csv('./data/test.csv',
                                    sep='\t').dropna().reset_index(drop=True)
        self.exclusive_col = ['text', 'lda', 'bow', 'label']

    def feature_engineer(self, data, tag='train'):
        datapath = './'+tag+'_feature.csv'
        # 如果存在读取，不存在生成。
        if os.path.exists(datapath):
            return pd.read_csv(datapath, sep=',')
        else :
            print(tag + " tfidf")
            data = get_tfidf(self.embedding.tfidf, data)
            print(tag + " embedding")
            data = get_embedding_feature(data, self.embedding.w2v)
            #print(tag + " lda")
            #data = get_lda_features(data, self.embedding.lda)
            #print(tag + " basic")
            #data = get_basic_feature(data)
            #         print(data)
            data.to_csv(datapath, sep=',', index=False, header=True)
            return data

    def trainer(self):
        self.train = self.feature_engineer(self.train, 'train')
        # dev = self.feature_engineer(self.dev)
        cols = [x for x in self.train.columns if x not in self.exclusive_col]

        X_train = self.train[cols]
        y_train = self.train['label'].apply(lambda x: self.labelToIndex[x])
        #y_train = self.train['label']

        #X_dev = dev[cols]
        #         y_test = dev['label'].apply(lambda x: eval(x))
        #y_dev = dev['label']

        xgb_train = xgb.DMatrix(X_train, label=y_train)

        #######################################################################
        #  16        TODO:  lgb模型训练 #
        #######################################################################
        # 初始化多标签训练
        # train
        print('Start training...')
        params = {
            'objective': 'multi:softmax',
            'eta': 0.1,
            'max_depth': 5,
            'num_class': 11,
            'eval_metric':'auc',
            'silent':0
        }
        num_round = 5
        self.clf_BR = xgb.train(params, xgb_train, num_round)

        joblib.dump(self.clf_BR, './xgb.model')


        #######################################################################
        #          TODO:  lgb模型预测 #
        #######################################################################
        # 初始化训练参数，并进行fit
        #
        print('Start predicting...')


    def save(self):
        joblib.dump(self.clf_BR, './model/clf_BR')

    def load(self):
        self.model = joblib.load('./xgb.model')


    def predict(self):
        print('Start predicting...')
        test = self.feature_engineer(self.test, 'test')
        cols = [x for x in self.test.columns if x not in self.exclusive_col]
        X_test = test[cols]
        y_test = test['label'].apply(lambda x: self.labelToIndex[x])
        xgb_test = xgb.DMatrix(X_test, label=y_test)
        pred = self.model.predict(xgb_test)
        error_rate=np.sum(pred!=y_test)/y_test.shape[0]
        print('测试集错误率(softmax):{}'.format(error_rate))
        accuray=1-error_rate
        print('测试集准确率：%.4f' %accuray)



if __name__ == "__main__":
    bc = Classifier(train_mode=True)
    bc.trainer()
    bc.save()
    bc.load()
    bc.predict()