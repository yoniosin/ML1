# -*- coding: utf-8 -*-
"""
Created on Mon Dec 18 14:33:53 2017

@author: amirli
"""
import numpy as np

import matplotlib.pyplot as plt

    
    

    
class LogisticModel:
    def __init__(self):
#        self.stop_threshold = 10**-5
#        self.mu = 0.001
        self.stop_threshold = 10**-6
        self.mu = 0.001        
        self.weights = 0
        self.loss_serial = []
        self.loss_batch = []
        self.count = 20

    def softmax(self,features):
        score = np.dot(features,self.weights)
        return 1 / (1 + np.exp(score))
    
    
    def checkStop(self,dw):
        #dw = vector of diffs in weights vector
        if self.count < 0:
            self.count = 5
        diff_sum = np.sum(dw**2)
        if diff_sum < self.stop_threshold:
            self.count -= 1
        else:
#            print(self.count)
            self.count = 5
        return (0 ==  self.count)
         
            
    def train(self,train_set,test_set,mode):
        
        # data = array of [parameters,data], were each column is an image
        # init weights vector in the size of the features + 1 for bias?
        TrainLoss = []
        TestLoss = []
        features_len = len(train_set[0][0])
        images_num = len(train_set)
        self.weights = np.ones(features_len, dtype=np.float64)
        index_list = np.random.permutation(images_num)
        first_ecpoch_done = False
        stop_flag = False
        image_done = 0
        epoch = 0
        #check initial loss precentage:
        loss = 0
        for features,label in train_set:
            g = self.softmax(-1*features) > 0.5
            update = abs(int(label) - int(g))
            loss += update
        loss = loss / len(train_set)
        TrainLoss.append(loss)
        
        loss = 0
        for features,label in test_set:
            g = self.softmax(-1*features) > 0.5
            update = abs(int(label) - int(g))
            loss += update
        loss = loss / len(test_set)
        TestLoss.append(loss)
        
        while not stop_flag:
            if 'serial' == mode:
                for i in index_list:
                    image_done += 1
                    sample_features = train_set[i][0]
                    data_label = train_set[i][1]
                    g = self.softmax(-1*sample_features)
                    diff = data_label - g
                    dw = self.mu*diff*sample_features
                    self.weights = self.weights + dw
    
                    if diff > 0:
                        stop_flag = self.checkStop(dw)
                        if stop_flag and first_ecpoch_done:
                            print("exiting!")
                            break
                        #check loss after each epoch:                                 
                    if image_done % 200 == 0:
                        loss = 0
                        for features,label in train_set:
                            g = self.softmax(-1*features) > 0.5
                            update = abs(int(label) - int(g))
                            loss += update
                        loss = loss / images_num
                        TrainLoss.append(loss)
                        
                        loss = 0
                        for features,label in test_set:
                            g = self.softmax(-1*features) > 0.5
                            update = abs(int(label) - int(g))
                            loss += update
                        loss = loss / len(test_set)
                        TestLoss.append(loss)
                        
                first_ecpoch_done = True
                epoch += 1
                print("finished epoch ",epoch)
                    #rearange data in new order:
                index_list = np.random.permutation(images_num)
            
            if 'batch' == mode:
                dw = 0
                for i in index_list:
                    image_done += 1
                    sample_features = train_set[i][0]
                    data_label = train_set[i][1]
                    g = self.softmax(-1*sample_features)
                    diff = data_label - g
                    dw += self.mu*diff*sample_features
                self.weights = self.weights + dw
                if dw.any() > 0:
                    stop_flag = self.checkStop(dw)
                    if stop_flag:
                        print("exiting!")
                        break
                #check loss after each epoch:                                 
                loss = 0
                for features,label in train_set:
                    g = self.softmax(-1*features) > 0.5
                    update = abs(int(label) - int(g))
                    loss += update
                loss = loss / len(train_set)
                TrainLoss.append(loss)
                
                loss = 0
                for features,label in test_set:
                    g = self.softmax(-1*features) > 0.5
                    update = abs(int(label) - int(g))
                    loss += update
                loss = loss / len(test_set)
                TestLoss.append(loss)
                
                epoch += 1
                print("finished epoch ",epoch)                    

        
        #batch
        print("Stopped after {} examples seen".format(image_done))
        print("Best accuracy measured over train set is: ", 1-min(TrainLoss),"%")
        print("Best accuracy measured over test set is: ", 1-min(TestLoss),"%")        
        fig = plt.figure()
        plt.plot(TrainLoss,'-g')
        plt.plot(TestLoss,'-b')
        plt.axhline(y=min(TrainLoss), xmin=0, xmax=TrainLoss.index(min(TrainLoss))/(len(TrainLoss)-1), linewidth=1, color = 'g', ls = '--', label = 'train')
        plt.axhline(y=min(TestLoss), xmin=0, xmax=TestLoss.index(min(TestLoss))/(len(TestLoss)-1), linewidth=1, color = 'b', ls = '--')

        plt.title('Logistic Model Loss - ' + mode + ' mode')
        plt.ylabel('error (%)')
        plt.legend(['Train Set Loss','Test Set Loss'])
        plt.show()

        return TrainLoss,TestLoss    
    

    
    
