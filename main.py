# -*- coding: utf-8 -*-
"""
Created on Mon Dec 18 16:45:14 2017

@author: amirli
"""

import scipy
import scipy.io as sio
import numpy as np
from logisticModel import LogisticModel
import matplotlib.pyplot as plt

#from Quantizer import quantizer

dataStruct = sio.loadmat('BreastCancerData.mat')
data = dataStruct['X']
dataSize = np.shape(data)
labels = dataStruct['y']

randIdx = np.random.permutation(dataSize[1])

trainSetSize = int(np.floor(dataSize[1] / 5))

max_data = np.amax(data)
min_data = np.amin(data)

# init train and test sets
normelized_data = (2*(data-min_data)/(max_data-min_data))-1

trainSet = []
for index in randIdx[:trainSetSize]:
    trainSet.append((normelized_data[:, index], labels[index]))

testSet = []
for index in randIdx[trainSetSize:]:
    testSet.append((normelized_data[:, index], labels[index]))


logistic_model = LogisticModel()
print("start training")
#train_loss,test_loss = logistic_model.train(trainSet,testSet,'batch')
train_loss,test_loss = logistic_model.train(trainSet,testSet,'serial')




