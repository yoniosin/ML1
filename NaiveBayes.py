# Following is the implementation of a Naive Bayes Classifier, which includes two functions:
# 1) trainNaiveBayes- Splits the training data according to provided labels, and calculates mean and std for each cluster
# 2) BinaryClassification- Classifies test data, to one of two possible clusters, 
# assuming i.i.d normal distribution (with provided mu, sigma)

import numpy as np
import pylab as P


def trainNaiveBayes(samples, classes_num, real_labels):
    samples_amount = len(samples)
    mat_list = []
    cluster_sizes = np.zeros(classes_num)
    for i in range(classes_num):
        mat_list.append(np.zeros((len(samples[0]), 0)))

    for index in range(samples_amount): # associate every sample to a cluster
        curr_label = int(real_labels[index])
        mat_list[curr_label] = np.append(mat_list[curr_label], samples[index][:, None], axis=1)
        cluster_sizes[curr_label] += 1

    mu = [np.mean(mat, axis=1) for mat in mat_list]
    sigma = [np.std(mat, axis=1) for mat in mat_list]
    return cluster_sizes, mu, sigma


def BinaryClassification(samples, cluster_sizes, mu, sigma):
    pdf = []
    for i in range(2):
        pdf.append(np.transpose(P.normpdf(samples, mu[i], sigma[i])))

    our_classification = P.prod(pdf[0], axis=0) * cluster_sizes[0] < P.prod(pdf[1], axis=0) * cluster_sizes[1]
    return our_classification
