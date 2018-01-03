# -*- coding: utf-8 -*-
"""
Created on Wed Jan  3 19:26:55 2018

@author: amirli
"""

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
import Q4

# from Quantizer import quantizer

dataStruct = sio.loadmat('BreastCancerData.mat')
data = dataStruct['X']
dataSize = np.shape(data)
labels = dataStruct['y']

#randIdx = np.random.permutation(dataSize[1])
randIdx = [i for i in range(np.shape(data)[1])]

trainSetSize = int(np.floor(4 * dataSize[1] / 5))

train_data = data[:, randIdx[:trainSetSize]]
train_labels = labels[randIdx[:trainSetSize]]
test_data = data[:, randIdx[trainSetSize:]]
test_labels = labels[randIdx[trainSetSize:]]

tree = Q4.Tree(train_data, train_labels)

count_true = 0
for i in range(np.shape(test_data)[1]):
    label = tree.predict(test_data[:, i])
    if label == test_labels[i]:
        count_true += 1

print(count_true/np.shape(test_data)[1]*100)

