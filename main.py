import numpy as np

from utils.ingest import load_mnist_images, load_mnist_labels
from utils.features import one_hot_encode
from utils.neural_network import NeuralNetwork, Layer
from utils.evaluation import accuracy

### INGEST ###

train_images = load_mnist_images('data/train-images-idx3-ubyte.gz')
train_labels = load_mnist_labels('data/train-labels-idx1-ubyte.gz')
test_images = load_mnist_images('data/t10k-images-idx3-ubyte.gz')
test_labels = load_mnist_labels('data/t10k-labels-idx1-ubyte.gz')

### PREPROCESS ###

# Normalize the images to the range [0, 1]
train_images = train_images / 255.0
test_images = test_images / 255.0

# One-hot encode the labels for classification
train_labels = one_hot_encode(train_labels)
test_labels = one_hot_encode(test_labels)

print(train_images.shape, train_labels.shape)
print(test_images.shape, test_labels.shape)

### MODEL ###

np.random.seed(0)

nn = NeuralNetwork([
    Layer(784, 16, 'relu'),
    Layer(16, 16, 'relu'),
    Layer(16, 16, 'relu'),
    Layer(16, 10, 'softmax')
])

nn.train(train_images, train_labels, epochs=2000, learning_rate=0.01)

### EVALUATE ###

y_pred = nn.predict(test_images)
print(f'Accuracy: {accuracy(y_pred, np.argmax(test_labels, axis=1))}')

nn.save('model.json')