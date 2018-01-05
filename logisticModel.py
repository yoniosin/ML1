# -*- coding: utf-8 -*-
"""
Created on Thu Jan  4 21:50:42 2018

@author: amirli
"""
import numpy as np
import matplotlib.pyplot as plt

def logisticRegression(train_data,train_labels,test_data,test_labels,mode):
    # init lists of train and test loss
    TrainLoss = []
    TestLoss = []
    
    # init hyper parameters
    mu = 0.04
    stop_thresh = 0.00001
    
    images_num = np.shape(train_data)[1]
    features_len = np.shape(train_data)[0]
    
    # init random weights
    w = np.random.normal(0, 0.5, features_len)
    
    # mesure initial train and test loss (before training):
    train_loss,test_loss = check_loss(w,train_data,train_labels,test_data,test_labels)                
    TrainLoss.append(train_loss)
    TestLoss.append(test_loss) 
    
    run = True
    epoch = 0
    # train (until stop condition is activated, and "run" will be set to False)
    while run:
        index_list = np.random.permutation(images_num) # choose random order
        if mode == 'serial':
            for i in index_list:
                features = train_data[:,index_list[i]]
                label = train_labels[index_list[i]]
                v = np.dot(w,features)
                g = 1 / (1+np.exp(-v))
                d_g = np.exp(-v) / ((1+np.exp(-v))**2)
                dw = mu*(label-g)*d_g*features
                # update weights after each sample:
                w = w + dw
                #check stop condition:
                if np.all(dw**2 < stop_thresh) and epoch > 2: 
                    run = False
                    break
                # check loss after each update:
                train_loss,test_loss = check_loss(w,train_data,train_labels,test_data,test_labels)                
                TrainLoss.append(train_loss)
                TestLoss.append(test_loss)
            epoch += 1
            print('finished epoch ',epoch)               
         
        else: #if Batch mode   
            dw = 0
            for i in index_list:
                features = train_data[:,index_list[i]]
                label = train_labels[index_list[i]]
                v = np.dot(w,features)
                g = 1 / (1+np.exp(-v))
                d_g = np.exp(-v) / ((1+np.exp(-v))**2)
                step = mu*(label-g)*d_g*features
                dw += step
                #check stop condition:
                if np.all(step**2 < stop_thresh) and epoch > 2:
                    run = False
                    break
            # update weights after each epoch:
            w = w + dw
            epoch += 1
            print("finished epoch ",epoch)
            
            # check loss after each epoch:
            train_loss,test_loss = check_loss(w,train_data,train_labels,test_data,test_labels)                
            TrainLoss.append(train_loss)
            TestLoss.append(test_loss)      
    
    return (TrainLoss,TestLoss)
        
        

def check_loss(w,train_data,train_labels,test_data,test_labels):
    v = np.dot(w,train_data)
    classifications = 1*((1 / (1+np.exp(-v))) > 0.5)
    classifications = classifications.reshape(np.shape(train_labels))
    train_loss = 100 * np.sum(abs(classifications-train_labels)) / len(train_labels)

    v = np.dot(w,test_data)
    classifications = 1*((1 / (1+np.exp(-v))) > 0.5)
    classifications = classifications.reshape(np.shape(test_labels))
    test_loss = 100 * np.sum(abs(classifications-test_labels)) / len(test_labels)
    
    return (train_loss,test_loss)