# -*- coding: utf-8 -*-

import os
import random
import chardet
import sys
import pickle
reload(sys)
sys.setdefaultencoding('utf-8')
import opt_ori_data as ood

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

def train(file_src):
    #split the data to train and test
    dataX, dataY = ood.run_train_data()
    trainX, testX, trainY, testY = train_test_split(dataX, dataY, test_size = 0.2, stratify = dataY)
    
    LR = LogisticRegression(penalty = 'l2')
    LR.fit(trainX, trainY)
    print 'mean accuracy: ' + str(LR.score(trainX, trainY))
    file_pickle = open(file_src, 'wb')
    p = pickle.dump(LR, file_pickle)
    return p

def predict(predictX, file_src):
    file_pickle = open(file_src, 'rb')
    LR = pickle.load(file_pickle)
    predictX = ood.run_predict_data(predictX)
    return LR.predict(predictX)

def predict_API(predictX):
    file_src = os.path.join(os.path.abspath('.'), 'LR_model.pkl')
    result =  predict(predictX, file_src)
    print result[0]
    return result[0]

def train_API():
    file_src = os.path.join(os.path.abspath('.'), 'LR_model.pkl')
    return train(file_src)

def run():
    train_API()
    #predict_API('今天好开心啊')

if __name__ == '__main__':
    run()

