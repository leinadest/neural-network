import numpy as np

from utils.ingest import load_mnist_images, load_mnist_labels
from utils.features import one_hot_encode
from utils.cli import args
from utils.neural_network import NeuralNetwork, Layer, load_neural_network
from utils.evaluation import accuracy
from utils.math import cross_entropy_loss

### INGEST ###

train_images = load_mnist_images("data/train-images-idx3-ubyte.gz")
train_labels = load_mnist_labels("data/train-labels-idx1-ubyte.gz")
test_images = load_mnist_images("data/t10k-images-idx3-ubyte.gz")
test_labels = load_mnist_labels("data/t10k-labels-idx1-ubyte.gz")

### PREPROCESS ###

# Normalize the images to the range [0, 1]
train_images = train_images / 255.0
test_images = test_images / 255.0

# One-hot encode the labels for classification
train_labels = one_hot_encode(train_labels)
test_labels = one_hot_encode(test_labels)

### MODEL ###

def train():
    np.random.seed(0)
    nn = NeuralNetwork([
        Layer(784, 64, "relu"),
        Layer(64, 64, "relu"),
        Layer(64, 64, "relu"),
        Layer(64, 10, "softmax")
    ])
    nn.train(train_images, train_labels, 
             epochs=args.epochs, 
             learning_rate=args.lr, 
             minibatch_size=args.minibatch,
             lambda_reg=args.reg)
    nn.save(args.model)

def evaluate():
    nn = load_neural_network(args.model)
    y_pred = nn.predict(test_images)
    accuracy_score = accuracy(test_labels, y_pred)
    loss = cross_entropy_loss(test_labels, y_pred)
    return accuracy_score, loss

def predict():
    nn = load_neural_network(args.model)
    y_pred = nn.predict(test_images)
    np.savetxt(args.output, np.argmax(y_pred, axis=1), fmt="%d")

### CLI ###

if args.command == "train":
    train()
    print(f"Model saved to {args.model}")
elif args.command == "evaluate":
    accuracy_score, loss = evaluate()
    print(f"Accuracy: {accuracy_score:.4f}")
    print(f"Loss: {loss:.4f}")
elif args.command == "predict":
    predict()
    print(f"Predictions saved to {args.output}")
