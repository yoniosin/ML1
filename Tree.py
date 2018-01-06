import numpy as np


# Following is an implementation of a Decision Tree
# The tree builds itself recursively upon initialization.
# there are three main classes:
# Tree - defined according to it's root node, and two inputs (provided on tree initialization):
#   mode, in (entropy, gini, error)
#   limit to the tree depth
# Node - each node in the tree, contains:
#   Two Children (left, right)
#   Decision feature
#   Threshold Value
# Leaf - a special node that holds the decision of the label
# each node builds it's children nodes (or leafs), until the tree is done

class Tree:
    def __init__(self, trainData, trainLabels, mode, depth):
        self.max_depth = depth
        self.mode = mode
        self.root = Node(self, trainData, trainLabels, [i for i in range(np.shape(trainData)[0])], 1)

    # Create a new node (or leaf) in the tree
    def buildNode(self, data, labels, featureList, node_depth):
        if self.isLeaf(labels, featureList, node_depth):
            return Leaf(labels)
        return Node(self, data, labels, featureList, node_depth + 1)

    # Leaf is created if all labels are equal, feature list is empty, or if max depth achieved
    def isLeaf(self, labels, featureList, nodeDepth):
        return np.std(labels) == 0 or len(featureList) == 0 or nodeDepth == self.max_depth

    def predict(self, sample):
        return self.root.predict(sample)


class Node:
    def __init__(self, tree, data, labels, featureList, depth):
        # calculate thresholds for each feature:
        thresArray = self.threshCalc(data)
        # choose best feature and remove it from future features list:
        self.featureIdx = self.chaosCalc(data, featureList, labels, thresArray, tree)
        self.thresh = thresArray[self.featureIdx]
        featureList.remove(self.featureIdx)
        # split the data according to the chosen feature and threshold:
        rightIdx = data[self.featureIdx, :] > self.thresh
        leftIdx = data[self.featureIdx, :] <= self.thresh
        # check if need to build a leaf or children nodes, and do it
        self.right = tree.buildNode(data[:, rightIdx], labels[rightIdx], featureList[:], depth)
        self.left = tree.buildNode(data[:, leftIdx], labels[leftIdx], featureList[:], depth)

    @staticmethod
    def chaosCalc(data, featureList, labels, thresharray, tree):
        bestFeature = -1
        bestChaos = np.inf

        for feature in featureList:
            posIdx = data[feature, :] > thresharray[feature]
            negIdx = data[feature, :] <= thresharray[feature]

            # calculate probabilities for feature:
            pos_pos_prob = sum(labels[posIdx]) / sum(posIdx) + 0.000001  # to avoid zero
            pos_neg_prob = 1 - pos_pos_prob + 0.000001
            neg_pos_prob = sum(labels[negIdx]) / sum(negIdx) + 0.000001  # to avoid zero
            neg_neg_prob = 1 - neg_pos_prob + 0.000001

            if tree.mode == 'entropy':
                pos_chaos = (-1) * (pos_pos_prob * np.log(pos_pos_prob) + pos_neg_prob * np.log(pos_neg_prob))
                neg_chaos = (-1) * (neg_pos_prob * np.log(neg_pos_prob) + neg_neg_prob * np.log(neg_neg_prob))
            elif tree.mode == 'gini':
                pos_chaos = pos_pos_prob * (1 - pos_pos_prob) + pos_neg_prob * (1 - pos_neg_prob)
                neg_chaos = neg_pos_prob * (1 - neg_pos_prob) + neg_neg_prob * (1 - neg_neg_prob)
            else:  # tree.mode == 'error':
                pos_chaos = 1 - max(pos_pos_prob, pos_neg_prob, neg_pos_prob, neg_neg_prob)
                neg_chaos = 0

            chaos = (pos_chaos * sum(posIdx) + neg_chaos * sum(negIdx)) / len(labels)

            if chaos < bestChaos:
                bestChaos = chaos
                bestFeature = feature

        return bestFeature

    @staticmethod
    def threshCalc(data):
        return np.mean(data, axis=1)

    def predict(self, sample):
        if sample[self.featureIdx] > self.thresh:
            return self.right.predict(sample)
        return self.left.predict(sample)


class Leaf:
    def __init__(self, labels):
        self.label = np.median(labels)

    def predict(self, sample):
        return self.label
