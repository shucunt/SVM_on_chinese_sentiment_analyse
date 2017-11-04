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

from sklearn.svm import SVC
def train(file_src):
    #split the data to train and test
    dataX, dataY = ood.run_train_data() 
    trainX, testX, trainY, testY = train_test_split(dataX, dataY, test_size = 0.2, stratify = dataY)

    clf = SVC(kernel = 'linear')
    clf.fit(trainX, trainY)
    print 'mean accuracy: ' + str(clf.score(trainX, trainY))
    file_pickle = open(file_src, 'wb')
    p = pickle.dump(clf, file_pickle)
    return p

def predict(predictX, file_src):
    file_pickle = open(file_src, 'rb')
    clf = pickle.load(file_pickle)
    predictX = ood.run_predict_data(predictX)
    return clf.predict(predictX)

def predict_API(predictX):
    file_src = os.path.join(os.path.abspath('.'), 'svm_model.pkl')
    result =  predict(predictX, file_src)
    print result[0]
    return result[0]

def train_API():
    file_src = os.path.join(os.path.abspath('.'), 'svm_model.pkl')
    return train(file_src)

def run():
    #train_API()
    predict_API('本人是一名大一学生，大一的生活一直处于浑浑噩噩的状态，直到我看到了这本书。它对于我的意义远远大于一本书。《杜拉拉升职记》让我开始重新审视自己的生活，开始规划自己的未来，在今后，我希望我能向拉拉一样所向无敌。人际关系的交际技巧，与上司下属的巧妙沟通，善于利用学习机会，每一次的金玉良言，我想说，杜拉拉不仅仅给在职的白领以启迪，她势必将改变我的生活。真的谢谢作者。')

if __name__ == '__main__':
    run()

