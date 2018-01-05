import numpy as np

max_depth = 30


# Following is an implementation of a Desicion Tree
# The tree builds itself recoursivley upon initialization.
# there are two main srtucts:
# Node - each node in the tree
# Leaf - a special node that holds the desicion of the label
# each node builds the following nodes or leafs, until the tree is done
# the tree can be built in one of 3 modes (entropy, gini, error)
# there is a limit to the tree depth, which sent as input upon initialization


class Tree:
    def __init__(self, trainData, trainLabels, mode, depth):
        global max_depth
        max_depth = depth
        self.root = Node(trainData, trainLabels, [i for i in range(np.shape(trainData)[0])], mode, 1)

    def predict(self, sample):
        return self.root.predict(sample)


class Node:
    def __init__(self, data, labels, featureList, mode, depth):
        # calculate thresholds for each feature:
        thresArray = threshCalc(data)
        # choose best feature and remove it from future features list:
        self.featureIdx = chaosCalc(data, featureList, thresArray, labels, mode)
        self.thresh = thresArray[self.featureIdx]
        featureList.remove(self.featureIdx)
        # split the data according to the chosen feature and threshold:
        rightIdx = data[self.featureIdx, :] > self.thresh
        leftIdx = data[self.featureIdx, :] <= self.thresh
        rightLabels = labels[rightIdx]
        leftLabels = labels[leftIdx]
        # check if need to build a leaf or children nodes, and do it
        self.right = self.buildSon(data[:, rightIdx], rightLabels, featureList[:], mode, depth)
        self.left = self.buildSon(data[:, leftIdx], leftLabels, featureList[:], mode, depth)

    def buildSon(self, data, labels, featureList, mode, depth):
        if isLeaf(labels, featureList, depth):
            return Leaf(labels)
        return Node(data, labels, featureList, mode, depth + 1)

    def predict(self, sample):
        if sample[self.featureIdx] > self.thresh:
            return self.right.predict(sample)
        return self.left.predict(sample)


class Leaf:
    def __init__(self, labels):
        self.label = np.median(labels)

    def predict(self, sample):
        return self.label


def chaosCalc(data, featureList, thresharray, labels, mode):
    bestFeature = -1
    bestChaos = np.inf

    for feature in featureList:
        posIdx = data[feature, :] > thresharray[feature]
        negIdx = data[feature, :] <= thresharray[feature]

        # calculate probebilities for feature:
        pos_pos_prob = sum(labels[posIdx]) / sum(posIdx) + 0.000001  # to avoid zero
        pos_neg_prob = 1 - pos_pos_prob + 0.000001
        neg_pos_prob = sum(labels[negIdx]) / sum(negIdx) + 0.000001  # to avoid zero
        neg_neg_prob = 1 - neg_pos_prob + 0.000001

        if mode == 'entropy':
            pos_chaos = (-1) * (pos_pos_prob * np.log(pos_pos_prob) + pos_neg_prob * np.log(pos_neg_prob))
            neg_chaos = (-1) * (neg_pos_prob * np.log(neg_pos_prob) + neg_neg_prob * np.log(neg_neg_prob))
        elif mode == 'gini':
            pos_chaos = pos_pos_prob * (1 - pos_pos_prob) + pos_neg_prob * (1 - pos_neg_prob)
            neg_chaos = neg_pos_prob * (1 - neg_pos_prob) + neg_neg_prob * (1 - neg_neg_prob)
        elif mode == 'error':
            pos_chaos = 1 - max(pos_pos_prob, pos_neg_prob, neg_pos_prob, neg_neg_prob)
            neg_chaos = 0

        chaos = (pos_chaos * sum(posIdx) + neg_chaos * sum(negIdx)) / len(labels)

        if chaos < bestChaos:
            bestChaos = chaos
            bestFeature = feature

    return bestFeature


def threshCalc(data):
    return np.mean(data, axis=1)


def isLeaf(labels, featureList, depth):
    global max_depth
    return np.std(labels) == 0 or len(featureList) == 0 or depth == max_depth
