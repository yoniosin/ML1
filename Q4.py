import numpy as np


class Tree:
    def __init__(self, trainData, trainLabels):
        self.root = Node(trainData, trainLabels, [i for i in range(np.shape(trainData)[0])])

    def predict(self, sample):
        return self.root.predict(sample)


class Node:
    def __init__(self, data, labels, featureList):
        thresArray = threshCalc(data)
        self.featureIdx = chaosCalc(data, featureList, thresArray, labels)
        self.thresh = thresArray[self.featureIdx]
        featureList.remove(self.featureIdx)

        rightIdx = data[self.featureIdx, :] > self.thresh
        leftIdx = data[self.featureIdx, :] <= self.thresh

        rightLabels = labels[rightIdx]
        leftLabels = labels[leftIdx]

        if not isLeaf(rightLabels, featureList):
            self.right = Node(data[:, rightIdx], rightLabels, featureList[:])
        else:
            self.right = Leaf(rightLabels)

        if not isLeaf(leftLabels, featureList):
            self.left = Node(data[:, leftIdx], leftLabels, featureList[:])
        else:
            self.left = Leaf(leftLabels)

    def predict(self, sample):
        if sample[self.featureIdx] > self.thresh:
            return self.right.predict(sample)
        else:
            return self.left.predict(sample)


class Leaf:
    def __init__(self, labels):
        self.label = np.median(labels)

    def predict(self, sample):
        return self.label


def chaosCalc(data, featureList, thresharray, labels):
    bestFeature = -1
    bestChaos = np.inf

    for feature in featureList:
        posIdx = data[feature, :] > thresharray[feature]
        negIdx = data[feature, :] <= thresharray[feature]

        pos_pos_prob = sum(labels[posIdx]) / sum(posIdx) + 0.000001  # to avoid zero
        pos_neg_prob = 1 - pos_pos_prob + 0.000001
        pos_chaos = pos_pos_prob * np.log(pos_pos_prob) + pos_neg_prob * np.log(pos_neg_prob)

        neg_pos_prob = sum(labels[negIdx]) / sum(negIdx) + 0.000001
        neg_neg_prob = 1 - neg_pos_prob + 0.000001
        neg_chaos = neg_pos_prob * np.log(neg_pos_prob) + neg_neg_prob * np.log(neg_neg_prob)

        chaos = (pos_chaos * sum(posIdx) + neg_chaos * sum(negIdx)) / len(labels) * (-1)
        if chaos < bestChaos:
            bestChaos = chaos
            bestFeature = feature
    return bestFeature


def threshCalc(data):
    return np.mean(data, axis=1)


def isLeaf(labels, featureList):
    return np.std(labels) == 0 or len(featureList) == 0
