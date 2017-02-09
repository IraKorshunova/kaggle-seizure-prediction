import numpy as np
from kaggle_auc import auc
import sklearn.metrics


def minmax_scaler(x):
    return (x - min(x)) / (max(x) - min(x))


targets1 = np.array([0, 0, 1, 1])
predictions1 = np.array([0.1, 0.2, 0.3, 0.4])
targets2 = np.array([0, 0, 1, 1])
predictions2 = np.array([0.6, 0.7, 0.8, 0.9])

targets = np.concatenate((targets1, targets2))

predictions1 = minmax_scaler(predictions1)
print predictions1
predictions2 = minmax_scaler(predictions2)
print predictions2

predictions = np.concatenate((predictions1, predictions2))

print
print 'AUC:', auc(targets, predictions), sklearn.metrics.roc_auc_score(targets, predictions)
