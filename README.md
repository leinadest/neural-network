# Neural Network

## Description

A simple implementation of a neural network trained on the MNIST dataset, made from scratch using numpy. This project demonstrates basic deep learning concepts:

- Forward and backward propagation
- Activation functions (ReLU, softmax)
- Loss functions (cross-entropy)
- Mini-batch gradient descent
- L2 regularization

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Training](#training)
- [Testing](#testing)
- [Math](#math)

## Installation

```bash
git clone https://github.com/leinadest/neural-network.git
cd neural-network
pipenv install
```

## Usage

1. Run `pipenv shell` to spawn a shell within the virtual environment

2. Train the model \
   a. with default hyperparameters and save to models/model.json

   ```bash
   python main.py train
   ```

   b. with custom hyperparameters

   ```bash
   python main.py train \
       --epochs 1000 \
       --lr 0.01 \
       --minibatch 100 \
       --reg 0.001 \
       --model models/model.json
   ```

3. Evaluate the model \
   a. from models/model.json

   ```bash
   python main.py evaluate
   ```

   b. from a custom path

   ```bash
   python main.py evaluate --model models/model.json
   ```

4. Make predictions \
   a. using models/model.json to output predictions to predictions.csv
   ```bash
   python main.py predict
   ```
   b. using a custom path to output predictions to a custom path
   ```bash
   python main.py predict --model models/model.json --output predictions.csv
   ```

## Training

### Data Preprocessing

```python
from utils.ingest import load_mnist_images, load_mnist_labels
from utils.features import one_hot_encode

# Ingest the MNIST dataset
train_images = load_mnist_images("data/train-images-idx3-ubyte.gz")
train_labels = load_mnist_labels("data/train-labels-idx1-ubyte.gz")

# Normalize the images to the range [0, 1]
train_images = train_images / 255.0

# One-hot encode the labels for classification
train_labels = one_hot_encode(train_labels)
```

### Neural Network Setup

```python
from numpy as np
from utils.neural_network import NeuralNetwork, Layer

np.random.seed(0)  # For reproducibility

# Only "relu" and "softmax" activations are supported
nn = NeuralNetwork([
    Layer(784, 64, "relu"),  # Input layer
    Layer(64, 64, "relu"),  # Hidden layer
    Layer(64, 64, "relu"),  # Hidden layer
    # Hidden layers...
    Layer(64, 10, "softmax")  # Output layer
])
```

### Training

```python
nn.train(train_images, train_labels, epochs=100, learning_rate=0.01, minibatch_size=32, regularization=0.001)
```

### Hyperparameters

| Hyperparameter   | Description                                    | Default Value |
| ---------------- | ---------------------------------------------- | ------------- |
| `epochs`         | Number of complete passes through the dataset. | `100`         |
| `learning_rate`  | Controls the step size during weight updates.  | `0.01`        |
| `minibatch_size` | Number of training examples in each batch.     | `32`          |
| `regularization` | L2 regularization strength.                    | `0.001`       |

### Procedure

The following steps are performed under `nn.train`.

1. Repeat for each minibatch in the training dataset.

   ```python
   for epoch in range(epochs):
       for i in range(0, X.shape[0], minibatch_size):
           self.forward(X[i:i+minibatch_size])
           self.backward(y[i:i+minibatch_size], learning_rate, lambda_reg)
   ```

   1. **Forward Pass**: The training data is passed through the network, layer by layer.
      ```python
      prev_layer_output = X
      for layer in self.layers:
          layer.output = np.dot(prev_layer_output, layer.weights) + layer.bias
          layer.activated_output = layer.activation_fn(layer.output)
          prev_layer_output = layer.activated_output
      ```
   2. **Backpropagation**: Gradients are computed and used to update the weights and biases.

      ```python
      next_error = None  # Error term from next layer
      for layer_idx in reversed(range(1, len(self.layers))):
          layer = self.layers[layer_idx]

          # Compute the current layer's error term
          error_term = None
          if layer == self.layers[-1]:
              error_term = layer.activated_output - y
          else:
              error_term = np.dot(
                  next_error,
                  self.layers[layer_idx + 1].weights.T
              ) * layer.activation_fn_derivative(layer.activated_output)

          # Compute weight's loss gradient and average across the minibatch
          prev_activated_output = self.layers[layer_idx - 1] \
                                      .activated_output
          l_gradient_w = np.dot(prev_activated_output.T,
                                  error_term) / error_term.shape[0]

          # Compute bias' loss gradient and average across the minibatch
          l_gradient_b = np.sum(error_term,
                                  axis=0,
                                  keepdims=True) / error_term.shape[0]

          # Add L2 regularization term to the weight gradient
          l_gradient_w += lambda_reg * layer.weights

          # Update weights and biases using gradients
          layer.weights -= learning_rate * l_gradient_w
          layer.bias -= learning_rate * l_gradient_b

          # Pass on error term before propagating to previous layer
          next_error = error_term
      ```

2. Repeat for multiple epochs until convergence or maximum number of epochs.

## Testing

### Evaluation Procedure

1. **Data Preparation**: Ensure the test data is preprocessed in the same way as the training data.
2. **Forward Pass**: Pass the test data through the trained model.
3. **Compute Metrics**: Compute accuracy and other relevant metrics.

```python
from utils.ingest import load_mnist_images, load_mnist_labels
from utils.features import one_hot_encode

# Ingest the MNIST dataset
test_images = load_mnist_images("data/t10k-images-idx3-ubyte.gz")
test_labels = load_mnist_labels("data/t10k-labels-idx1-ubyte.gz")

# Normalize the images to the range [0, 1]
test_images = test_images / 255.0

# One-hot encode the labels for classification
test_labels = one_hot_encode(test_labels)

nn = load_neural_network("models/model.json")

# Compute predictions and output to predictions.csv
y_pred = nn.predict(test_images)
np.savetxt('predictions.csv', np.argmax(y_pred, axis=1), fmt="%d")

# Compute evaluation and print
accuracy_score = accuracy(test_labels, y_pred)
loss = cross_entropy_loss(test_labels, y_pred)
print(f"Accuracy: {accuracy_score:.4f}")
print(f"Loss: {loss:.4f}")
```

### Example Evaluation

`models/model.json`

- Training data: `data/train-images-idx3-ubyte.gz` and `data/train-labels-idx1-ubyte.gz`
- Experimental setup:
  - `np.random.seed(0)`
  - 1000 epochs
  - 0.01 learning rate
  - 100 mini-batch size
  - 0.001 L2 regularization.

Results on testing set `data/t10k-images-idx3-ubyte.gz`:

```bash
Accuracy: 0.9245
Loss: 0.2492
```

## Math

### Example Neural Network

<img src="https://i.imgur.com/SxrIzwZ.png" width="512">

### Definitions

$$
\begin{align}
    &\text{Let }m\text{ be the number of training examples.}\\
    &\text{Let }n\text{ be the number of features.}\\
    &\text{Let }A\text{ be the input data }(m\times n).\\
    &\text{Let }z^{[l]}\text{ be the output of layer }l.\\
    &\text{Let }a^{[l]}\text{ be the activated output of layer }l.\\
    &\text{Let }W^{[l]}\text{ be the weight matrix of layer }l.\\
    &\text{Let }b^{[l]}\text{ be the bias vector of layer }l.\\
    &\text{Let }y\text{ be the true labels }(m\times 10).\\
    &\text{Let }\hat{y}\text{ be the predicted labels }(m\times 10).\\
    &\text{Let }L\text{ be the number of layers.}\\
    &\text{Let }J\text{ be the loss function.}\\
    &\text{Let }\delta^{[l]}\text{ be the error term of layer }l.\\
    &\text{Let }\alpha\text{ be the learning rate.}\\
    &\text{Let }\lambda\text{ be the L2 regularization parameter.}\\
\end{align}
$$

### Forward Propagation

#### Input Layer:

$$
\begin{align}
    a^{[0]}=A&(m\times n)
\end{align}
$$

#### Hidden Layers:

1. Calculate the layer's pre-activation output

$$
\begin{align}
    z^{[l]}=a^{[l-1]}W^{[l]}+b^{[l]}&(m\times 64)
\end{align}
$$

2. Apply the activation function

$$
\begin{align}
    a^{[l]}=ReLU(z^{[l]})&(m\times 64)
\end{align}
$$

3. Propagate $a^{[l]}$ to the next layer

<br/>

#### Output Layer:

1. Calculate the layer's pre-activation output
   
$$
\begin{align}
    z^{[L-1]}=a^{[L-2]}W^{[L-1]}+b^{[L-1]}&(m\times 10)
\end{align}
$$
   
2. Apply the activation function

$$
\begin{align}
    a^{[L-1]}=Softmax(z^{[L-1]}_i)&(m\times 10)
\end{align}
$$

3. Calculate the loss (optional for logging; not needed for error term calculation when cross-entropy error is differentiated with softmax)

```math
\begin{align}
   L=-\frac{1}{m} \sum^m_{i=0} \sum^{10}_{j=0} y_{i,j}log(a^{[L-1]}_{i,j}) & (1\times 1)
\end{align}
```

### Backward Propagation

#### Output Layer:

1. Calculate the error term

$$
\begin{align}
    \delta^{[L-1]}=a^{[L-1]}-y&(m\times 10)
\end{align}
$$

2. Calculate the loss gradient with respect to the layer's weights

$$
\begin{align}
    \frac{dJ}{dW^{[L-1]}}=a^{[L-2]T}\delta^{[L-1]}&(10\times 10)
\end{align}
$$

3. The loss gradient with respect to the layer's biases equals the error term

$$
\begin{align}
    \frac{dJ}{db^{[L-1]}}=\delta^{[L-1]}&(m\times 10)
\end{align}
$$

4. Update the weights by gradient descent with L2 regularization (the gradient is already summed across the minibatch, only need to divide by m to average it)

$$
\begin{align}
    \Delta W^{[L-1]}=-\frac{\alpha}{m}\frac{dJ}{dW^{[L-1]}}+\lambda W^{[L-1]}&(10\times 10)
\end{align}
$$

5. Update the biases by averaging the gradient across the minibatch before descent

$$
\begin{align}
    \Delta b^{[L-1]}=-\frac{\alpha}{m}\sum^m_{i=0}\frac{dJ}{db^{[L-1]}_i}&(1\times 10)
\end{align}
$$

#### Hidden Layers:

1. Calculate the error term

$$
\begin{align}
    \delta^{[l]}=\delta^{[l+1]}W^{[l+1]T}\odot ReLU'(z^{[l]})&(m\times 64)
\end{align}
$$

2. Calculate the loss gradient with respect to the layer's weights

$$
\begin{align}
    \frac{dJ}{dW^{[l]}}=a^{[l-1]T}\delta^{[l]}&(64\times 64)
\end{align}
$$

3. The loss gradient with respect to the layer's biases equals the error term

$$
\begin{align}
    \frac{dJ}{db^{[l]}}=\delta^{[l]}&(m\times 64)
\end{align}
$$

4. Update the weights by gradient descent with L2 regularization (the gradient is already summed across the minibatch, only need to divide by m to average it)

$$
\begin{align}
    \Delta W^{[l]}=-\frac{\alpha}{m}\frac{dJ}{dW^{[l]}}+\lambda W^{[l]}&(64\times 64)
\end{align}
$$

5. Update the biases by averaging the gradient across the minibatch before descent

$$
\begin{align}
    \Delta b^{[l]}=-\frac{\alpha}{m}\sum^m_{i=0}\frac{dJ}{db^{[l]}_i}&(1\times 64)
\end{align}
$$
