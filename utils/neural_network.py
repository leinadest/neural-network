import numpy as np
import json

from utils.math import (softmax, softmax_derivative, relu, 
                        relu_derivative, l2_regularized_loss)

def he_initializer(input_size, output_size):
    """
    Initialize weights suitable for ReLU activation function using the He et al.
    initialization strategy. Initializes the weights scaled by the square root
    of 2 divided by the input size, so that the variance of the activations is 
    approximately 1.

    Parameters:
    input_size (int): The number of inputs in the layer.
    output_size (int): The number of outputs in the layer.

    Returns:
    weights (np.ndarray): The initialized weights.
    """
    return (np.random.randn(input_size, output_size) 
            * np.sqrt(2. / input_size))

def xavier_initializer(input_size, output_size):
    """
    Initialize weights suitable for softmax activation function using the Xavier
    initialization strategy. Initializes the weights scaled by the square root
    of 1 divided by the input size, so that the variance of the activations is 
    approximately 1.

    Parameters:
    input_size (int): The number of inputs in the layer.
    output_size (int): The number of outputs in the layer.

    Returns:
    weights (np.ndarray): The initialized weights.
    """
    return (np.random.randn(input_size, output_size) 
            * np.sqrt(1. / input_size))

def load_neural_network(filepath):
    """
    Load a neural network from a file.

    Parameters:
    filepath (str): The path to load the network from.
    """
    with open(filepath, 'r') as f:
        params = json.load(f)
    layers = []
    for layer_info in params["layers"]:
        layer = Layer(1, 1, layer_info["activation"])
        layer.weights = np.array(layer_info["weights"])
        layer.bias = np.array(layer_info["bias"])
        layers.append(layer)
    return NeuralNetwork(layers)

class Layer:
    """
    A single layer of the neural network.
    
    Attributes:
    input_size (int): The number of inputs in the layer.
    output_size (int): The number of outputs in the layer.
    activation (str): The activation function to use.
    weights (np.ndarray): The weights of the layer.
    bias (np.ndarray): The bias of the layer.
    output (np.ndarray): The output of the layer.
    activated_output (np.ndarray): The activated output of the layer.
    """
    def __init__(self, input_size, output_size, activation='relu'):
        """
        Initialize the layer.

        Parameters:
        input_size (int): The number of inputs in the layer.
        output_size (int): The number of outputs in the layer.
        activation (str): The activation function to use.
        """
        self.bias = np.zeros((1, output_size))
        self.activation = activation
        if activation == 'relu':
            self.weights = he_initializer(input_size, output_size)
            self.activation_fn = relu
            self.activation_fn_derivative = relu_derivative
        elif activation == 'softmax':
            self.weights = xavier_initializer(input_size, output_size)
            self.activation_fn = softmax
            self.activation_fn_derivative = softmax_derivative
        self.output = None
        self.activated_output = None

class NeuralNetwork:
    """
    A neural network with multiple layers.
    
    Attributes:
    layers (list): A list of Layer objects.

    Methods:
    forward(X): Forward propagate the input through the network.
    backward(y, learning_rate): Backpropagate the error through the network.
    train(X, y, epochs, learning_rate, minibatch_size, lambda_reg): Train the network.
    predict(X): Make predictions using the network.
    save(filename): Save the network to a file.
    """
    def __init__(self, layers):
        """
        Initialize the network.

        Parameters:
        layers (list): A list of Layer objects.
        """
        self.layers = layers

    def forward(self, X):
        """
        Forward propagate the input through the network.

        Parameters:
        X (np.ndarray): The input data.
        """
        prev_layer_output = X
        for layer in self.layers:
            layer.output = np.dot(prev_layer_output, layer.weights) + layer.bias
            layer.activated_output = layer.activation_fn(layer.output)
            prev_layer_output = layer.activated_output
    
    def backward(self, y, learning_rate, lambda_reg):
        """
        Backpropagate the error through the network.

        Parameters:
        y (np.ndarray): The true labels.
        learning_rate (float): The learning rate.
        lambda_reg (float): The L2 regularization parameter.
        """
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
    
    def train(self, X, y, epochs=1000, learning_rate=0.01, 
              minibatch_size=100, lambda_reg=0.001):
        """
        Train the network.

        Parameters:
        X (np.ndarray): The input data.
        y (np.ndarray): The true labels.
        epochs (int): The number of epochs to train for.
        learning_rate (float): The learning rate.
        minibatch_size (int): The size of the minibatches.
        lambda_reg (float): The L2 regularization parameter.
        """
        for epoch in range(epochs):
            for i in range(0, X.shape[0], minibatch_size):
                self.forward(X[i:i+minibatch_size])
                self.backward(y[i:i+minibatch_size], learning_rate, lambda_reg)
            if epoch % 10 == 0:
                y_pred = self.layers[-1].activated_output
                loss = l2_regularized_loss(y[i:i+minibatch_size], y_pred, self.layers, lambda_reg)
                print(f"Epoch {epoch}/{epochs}, Loss: {loss:.4f}")
    
    def predict(self, X):
        """
        Make predictions using the network.

        Parameters:
        X (np.ndarray): The input data.
        """
        self.forward(X)
        return self.layers[-1].activated_output

    def save(self, filepath):
        """
        Save the network to a file.

        Parameters:
        filepath (str): The path to save the network to.
        """
        params = {
            "layers": [{
                "weights": layer.weights.tolist(),
                "bias": layer.bias.tolist(),
                "activation": layer.activation
            } for layer in self.layers],
        }
        with open(filepath, 'w') as f:
            json.dump(params, f)
    