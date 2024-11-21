import numpy as np

# Convert labels to one-hot encoded format
def one_hot_encode(labels, num_classes=10):
    return np.eye(num_classes)[labels]