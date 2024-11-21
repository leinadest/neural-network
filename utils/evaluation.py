import numpy as np

def accuracy(y_pred, y_true):
    return np.mean(np.argmax(y_true, axis=1) == np.argmax(y_pred, axis=1))