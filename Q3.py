
import scipy
import scipy.io as sio
import numpy as np
from logisticModel import logisticRegression
import matplotlib.pyplot as plt

dataStruct = sio.loadmat('BreastCancerData.mat')
data = dataStruct['X']
dataSize = np.shape(data)
labels = dataStruct['y']

randIdx = np.random.permutation(dataSize[1])

trainSetSize = int(np.floor(4 * dataSize[1] / 5))

train_data = data[:, randIdx[:trainSetSize]]
train_labels = labels[randIdx[:trainSetSize]]
test_data = data[:, randIdx[trainSetSize:]]
test_labels = labels[randIdx[trainSetSize:]]

# normelize the data
max_data = np.amax(data, axis=1)
min_data = np.amin(data, axis=1)
avg_data = np.average(data, axis=1)
means_expanded = np.outer(avg_data, np.ones(dataSize[1]))
max_expanded = np.outer(max_data, np.ones(dataSize[1]))

normelized_data = (data - means_expanded) / (max_expanded - means_expanded)
new_data = np.vstack([normelized_data, np.ones(dataSize[1])])

# init train and test sets
train_data = new_data[:, randIdx[:trainSetSize]]
train_labels = labels[randIdx[:trainSetSize]]
test_data = new_data[:, randIdx[trainSetSize:]]
test_labels = labels[randIdx[trainSetSize:]]

TrainLoss, TestLoss = logisticRegression(train_data, train_labels, test_data, test_labels, 'serial')

print("Best accuracy measured over train set is: ", 100 - min(TrainLoss), "%")
print("Best accuracy measured over test set is: ", 100 - min(TestLoss), "%")
fig = plt.figure()
plt.plot(TrainLoss, '-g')
plt.plot(TestLoss, '-b')
plt.title('Logistic Model Error (%)')
plt.ylabel('error (%)')
plt.xlabel('iteration #')
plt.legend(['Train Set', 'Test Set'])
plt.show()
