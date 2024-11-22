import numpy as np
import gzip

def load_mnist_images(filename):
    """
    Load the images from the MNIST dataset.

    Parameters:
    filename (str): The path to the MNIST images file.
    """
    with gzip.open(filename, 'rb') as f:
        f.read(16)  # Skip the magic number and dimensions
        # Read the image data
        images = np.frombuffer(f.read(), dtype=np.uint8).reshape(-1, 28, 28)
        # Flatten the images 20x20 images to 724x1
        images = images.reshape(images.shape[0], -1)
    return images

def load_mnist_labels(filename):
    """
    Load the labels from the MNIST dataset.

    Parameters:
    filename (str): The path to the MNIST labels file.
    """
    with gzip.open(filename, 'rb') as f:
        f.read(8)  # Skip the magic number and number of labels
        labels = np.frombuffer(f.read(), dtype=np.uint8)
    return labels