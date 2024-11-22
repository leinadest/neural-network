import numpy as np

def accuracy(y_pred, y_true):
    """
    Compute the accuracy of the predictions.

    Parameters:
    y_pred (np.ndarray): The predicted probabilities.
    y_true (np.ndarray): The true labels.
    
    Returns:
    accuracy (float): The accuracy of the predictions.
    """
    return np.mean(np.argmax(y_true, axis=1) == np.argmax(y_pred, axis=1))