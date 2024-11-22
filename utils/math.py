import numpy as np

def softmax(X, epsilon=1e-12):
    """
    Compute the softmax function for each row of the input X.

    Parameters:
    X (np.ndarray): The input data.
    epsilon (float): A small number to avoid division by zero.
    
    Returns:
    softmax_X (np.ndarray): The softmax values for each row of X.
    """
    X_minus_max = X - np.max(X, axis=1, keepdims=True)
    exp_X = np.exp(X_minus_max)
    exp_X_sums = np.sum(exp_X, axis=1, keepdims=True)
    return exp_X / (exp_X_sums + epsilon)

def softmax_derivative(softmax_X):
    """
    Compute the derivative of the softmax function for each row of the input X.

    Parameters:
    softmax_X (np.ndarray): The softmax values for each row of X.
    
    Returns:
    jacobian (np.ndarray): The derivative of the softmax function for each row of X.
    """
    minibatch_size, layer_size = softmax_X.shape
    jacobian = np.zeros((minibatch_size, layer_size, layer_size))
    for i in range(minibatch_size):
        softmax_col = softmax_X[i].reshape(-1, 1)
        jacobian[i] = (np.diagflat(softmax_col) 
                       - np.dot(softmax_col, softmax_col.T))
    return jacobian

def relu(X):
    """
    Compute the ReLU activation function for each element of the input X.

    Parameters:
    X (np.ndarray): The input data.
    
    Returns:
    relu_X (np.ndarray): The ReLU values for each element of X.
    """
    return np.maximum(0, X)

def relu_derivative(X):
    """
    Compute the derivative of the ReLU activation function for each element of the input X.

    Parameters:
    X (np.ndarray): The input data.
    
    Returns:
    relu_X (np.ndarray): The ReLU values for each element of X.
    """
    return np.where(X > 0, 1, 0)

def cross_entropy_loss(y_true, y_pred, epsilon=1e-12):
    """
    Compute the cross-entropy loss between the true labels and the predicted probabilities.

    Parameters:
    y_true (np.ndarray): The true labels.
    y_pred (np.ndarray): The predicted probabilities.
    epsilon (float): A small number to avoid division by zero.
    
    Returns:
    loss (float): The cross-entropy loss.
    """
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    minibatch_size = y_true.shape[0]
    return -np.sum(y_true * np.log(y_pred)) / minibatch_size

def l2_regularized_loss(y_true, y_pred, layers, epsilon=1e-12, lambda_reg=0.001):
    """
    Compute the cross-entropy loss between the true labels and the predicted probabilities.

    Parameters:
    y_true (np.ndarray): The true labels.
    y_pred (np.ndarray): The predicted probabilities.
    layers (list): A list of Layer objects.
    epsilon (float): A small number to avoid division by zero.
    lambda_reg (float): The L2 regularization parameter.
    
    Returns:
    loss (float): The cross-entropy loss.
    """ 
    loss = cross_entropy_loss(y_true, y_pred, epsilon)
    l2_penalty = sum(np.sum(layer.weights**2) for layer in layers)
    return loss + (lambda_reg / 2) * l2_penalty