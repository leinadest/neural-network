import numpy as np

def one_hot_encode(labels, num_classes=10):
    """
    Converts labels to one-hot encoding.
    
    Parameters:
    labels (np.ndarray): Array of labels.
    num_classes (int): Number of classes.
    
    Returns:
    np.ndarray: One-hot encoded labels.
    """
    return np.eye(num_classes)[labels]