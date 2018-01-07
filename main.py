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

# from Quantizer import quantizer
from NaiveBayes import *

dataStruct = sio.loadmat('BreastCancerData.mat')
data = dataStruct['X']
dataSize = np.shape(data)
labels = dataStruct['y']

randIdx = np.random.permutation(dataSize[1])

trainSetSize = int(np.floor(dataSize[1] * 4 / 5))

max_data = np.amax(data)
min_data = np.amin(data)

# init train and test sets
normelized_data = (2 * (data - min_data) / (max_data - min_data)) - 1

trainSet = []
for index in randIdx[:trainSetSize]:
    trainSet.append((normelized_data[:, index], labels[index]))

train_data = [data for data, label in trainSet]
train_label = [label for data, label in trainSet]

testSet = []
for index in randIdx[trainSetSize:]:
    testSet.append((normelized_data[:, index], labels[index]))

test_data = [data for data, label in testSet]
test_label = [label for data, label in testSet]

# train
cluster_sizes, mu, sigma = trainNaiveBayes(train_data, 2, train_label)
# classification
our_classification_test = BinaryClassification(test_data, cluster_sizes, mu, sigma)

correct_test = np.sum(int(x) == int(y) for x, y in zip(test_label, our_classification_test)) / len(test_label)

print("Success Rate Test:", correct_test * 100)

our_classification_train = BinaryClassification(train_data, cluster_sizes, mu, sigma)

correct_train = np.sum(int(x) == int(y) for x, y in zip(train_label, our_classification_train)) / len(train_label)
print("Success Rate Train:", correct_train * 100)
