import scipy
import scipy.io as sio
import numpy as np
import numpy.matlib


def quantizer(data, min_eps, max_iter):
    """
    :param data: m samples of n components each
    :param min_eps: minimum change between two iterations
    :param max_iter: alternative stopping condition, in case solution doesn't converge
    :return: clusters coordinates
    """
    (m, s) = np.shape(data)
    unique_levels = np.unique(data, axis=0)
    QL = unique_levels[:, :2]
    QL_mat = np.zeros((m, s, 2))
    data_rep = np.zeros((m, s, 2))
    distortion = [0]
    for j in range(20):
        for i in range(2):
            new_mat = np.matlib.repmat(np.transpose(QL[:, i]), s, 1)
            QL_mat[:, :, i] = np.transpose(new_mat)
            data_rep[:, :, i] = data
        diff = data_rep - QL_mat
        distance = np.matlib.sum(np.square(diff), axis=0)
        cluster = np.matlib.argmin(distance, axis=1)
        data_out = QL[:, cluster]
        distortion[j + 1] = np.mean(data_out - data)

        idx_mat = zeros(m, s + 1, 2)
        idx_mat[:, :s, :] = np.matlib.reshape(np.matlib.repmat(data, 1, s), m, s)


    return cluster
