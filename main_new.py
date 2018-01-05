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
from logisticModel import logisticRegression
import matplotlib.pyplot as plt
import Tree
from Quantizer import quantizer

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

train_data_list = []
train_labels_list = []
test_data_list = []
test_labels_list = []



########################## Q4 - Tree ###########################
# Cross Validation
#modes = ['entropy','gini','error']
#max_depth = [4,5,6,7,8,10]
#results = np.zeros((3,6,10))
#for i in range(10):
#    train_data_list.append(np.delete(data,range(i*56,(i+1)*56),axis=1))
#    train_labels_list.append(np.delete(labels,range(i*56,(i+1)*56),axis=0))
#    test_data_list.append(data[:,range(i*56,(i+1)*56)])
#    test_labels_list.append(labels[range(i*56,(i+1)*56)])
#
#    
#for m,mode in enumerate(modes):
#    print("calculating for ",mode)
#    for d,depth in enumerate(max_depth):
#        for i in range(10):
#            tree = Q4.Tree(train_data_list[i], train_labels_list[i], mode, depth)
#            count_true = 0
#            for j in range(np.shape(test_data_list[1])[1]):
#                label = tree.predict(test_data_list[i][:, j])
#                if label == test_labels_list[i][j]:
#                    count_true += 1
#            results[m,d,i] = count_true/np.shape(test_data_list[1])[1]*100
#
#for m,mode in enumerate(modes):    
#    for d,depth in enumerate(max_depth):
#        print('mode: {}, depth: {}, std={}, avg accuracy={}'.format(mode, depth,np.std(results[m,d,:]),np.mean(results[m,d,:])))
#        
mode = 'entropy'
tree = Tree.Tree(train_data, train_labels, mode, 6)

count_true = 0
for i in range(np.shape(train_data)[1]):
    label = tree.predict(train_data[:, i])
    if label == train_labels[i]:
        count_true += 1

print('train accuracy: ', count_true/np.shape(train_data)[1]*100,'%')
    
count_true = 0
for i in range(np.shape(test_data)[1]):
    label = tree.predict(test_data[:, i])
    if label == test_labels[i]:
        count_true += 1

print('test accuracy: ', count_true/np.shape(test_data)[1]*100,'%')


########################## Q3 - Logistic Regression ###########################
# normelize the data
max_data = np.amax(data,axis=1)
min_data = np.amin(data,axis=1)
avg_data = np.average(data,axis=1)
means_expanded = np.outer(avg_data, np.ones(dataSize[1]))
max_expanded = np.outer(max_data, np.ones(dataSize[1]))

normelized_data = (data-means_expanded)/(max_expanded-means_expanded)
new_data = np.vstack([normelized_data,np.ones(dataSize[1])])

# init train and test sets
train_data = new_data[:,randIdx[:trainSetSize]]
train_labels = labels[randIdx[:trainSetSize]]
test_data = new_data[:,randIdx[trainSetSize:]]
test_labels = labels[randIdx[trainSetSize:]]

TrainLoss,TestLoss = logisticRegression(train_data,train_labels,test_data,test_labels,'serial')

print("Best accuracy measured over train set is: ", 100-min(TrainLoss),"%")
print("Best accuracy measured over test set is: ", 100-min(TestLoss),"%")        
fig = plt.figure()
plt.plot(TrainLoss,'-g')
plt.plot(TestLoss,'-b')
plt.title('Logistic Model Error (%)')
plt.ylabel('error (%)')
plt.xlabel('iteration #')
plt.legend(['Train Set','Test Set'])
plt.show()


