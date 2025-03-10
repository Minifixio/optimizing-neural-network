import numpy as np
cimport numpy as np

DTYPE = np.float64
ctypedef np.float64_t DTYPE_t

cpdef np.ndarray[np.float64_t, ndim=2] g(np.ndarray[np.float64_t, ndim=2] x):
    """ Sigmoid function applied element-wise to a NumPy array """
    return 1.0 / (1.0 + np.exp(-x))

cpdef np.ndarray[np.float64_t, ndim=2] grad_g(np.ndarray[np.float64_t, ndim=2] x):
    """ Gradient of the sigmoid function applied element-wise to a NumPy array """
    cdef np.ndarray[np.float64_t, ndim=2] gx = g(x)
    return gx * (1.0 - gx)

def reshape(np.ndarray[DTYPE_t, ndim=1] theta, int input_layer_size, int hidden_layer_size, int num_labels):
    """ Reshape theta into Theta1 and Theta2 """
    cdef int ncut = hidden_layer_size * (input_layer_size + 1)
    cdef np.ndarray[DTYPE_t, ndim=2] Theta1 = theta[:ncut].reshape(hidden_layer_size, input_layer_size + 1)
    cdef np.ndarray[DTYPE_t, ndim=2] Theta2 = theta[ncut:].reshape(num_labels, hidden_layer_size + 1)
    return Theta1, Theta2

def gradient(np.ndarray[DTYPE_t, ndim=1] theta, int input_layer_size, int hidden_layer_size, int num_labels,
             np.ndarray[DTYPE_t, ndim=2] X, np.ndarray[np.int64_t, ndim=2] y, double lmbda):
    """ Compute the gradient of the neural network cost function """
    # Unflatten theta
    cdef np.ndarray[DTYPE_t, ndim=2] Theta1, Theta2
    Theta1, Theta2 = reshape(theta, input_layer_size, hidden_layer_size, num_labels)

    # Number of training examples
    cdef int m = X.shape[0]

    # Add bias to input layer
    cdef np.ndarray[DTYPE_t, ndim=2] X_bias = np.hstack((np.ones((m, 1), dtype=DTYPE), X))

    # Forward propagation
    cdef np.ndarray[DTYPE_t, ndim=2] Z2 = X_bias @ Theta1.T
    cdef np.ndarray[DTYPE_t, ndim=2] A2 = g(Z2)
    
    # Add bias to hidden layer
    cdef np.ndarray[DTYPE_t, ndim=2] A2_bias = np.hstack((np.ones((m, 1), dtype=DTYPE), A2))

    # Output layer
    cdef np.ndarray[DTYPE_t, ndim=2] Z3 = A2_bias @ Theta2.T
    cdef np.ndarray[DTYPE_t, ndim=2] A3 = g(Z3)

    # One-hot encoding of y
    cdef np.ndarray[DTYPE_t, ndim=2] Y = np.zeros((m, num_labels), dtype=DTYPE)
    cdef int i
    for i in range(m):
        Y[i, int(y[i, 0])] = 1.0

    # Compute errors
    cdef np.ndarray[DTYPE_t, ndim=2] Delta3 = A3 - Y
    cdef np.ndarray[DTYPE_t, ndim=2] Delta2 = (Delta3 @ Theta2[:, 1:]) * grad_g(Z2)

    # Compute gradients
    cdef np.ndarray[DTYPE_t, ndim=2] Theta1_grad = (Delta2.T @ X_bias) / m
    cdef np.ndarray[DTYPE_t, ndim=2] Theta2_grad = (Delta3.T @ A2_bias) / m

    # Add regularization (excluding bias term)
    Theta1_grad[:, 1:] += (lmbda / m) * Theta1[:, 1:]
    Theta2_grad[:, 1:] += (lmbda / m) * Theta2[:, 1:]

    # Flatten gradients
    cdef np.ndarray[DTYPE_t, ndim=1] grad = np.concatenate((Theta1_grad.ravel(), Theta2_grad.ravel()))

    return grad