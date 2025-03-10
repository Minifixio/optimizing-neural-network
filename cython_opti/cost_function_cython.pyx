import numpy as np
cimport numpy as np
from timeit import default_timer as timer

DTYPE = np.float64
ctypedef np.float64_t DTYPE_t

cdef np.ndarray[np.float64_t, ndim=2] sigmoid(np.ndarray[np.float64_t, ndim=2] x):
    return 1.0 / (1.0 + np.exp(-x))

cdef tuple reshape(np.ndarray[DTYPE_t, ndim=1] theta, int input_layer_size, int hidden_layer_size, int num_labels):
    """ Reshape theta into Theta1 and Theta2 """
    cdef int ncut = hidden_layer_size * (input_layer_size + 1)
    cdef np.ndarray[DTYPE_t, ndim=2] Theta1 = theta[:ncut].reshape(hidden_layer_size, input_layer_size + 1)
    cdef np.ndarray[DTYPE_t, ndim=2] Theta2 = theta[ncut:].reshape(num_labels, hidden_layer_size + 1)
    return Theta1, Theta2

def cost_function(np.ndarray[DTYPE_t, ndim=1] theta, int input_layer_size, int hidden_layer_size, int num_labels,
                  np.ndarray[DTYPE_t, ndim=2] X, np.ndarray[np.int64_t, ndim=2] y, double lmbda):  # Specify y as int64
    """ Neural net cost function for a three layer classification network. """

    # Unflatten theta
    cdef np.ndarray[DTYPE_t, ndim=2] Theta1, Theta2
    Theta1, Theta2 = reshape(theta, input_layer_size, hidden_layer_size, num_labels)

    # Number of training values
    cdef int m = X.shape[0]

    # Feedforward: calculate the cost function J:
    cdef np.ndarray[DTYPE_t, ndim=2] a1 = np.hstack((np.ones((m, 1)), X))
    cdef np.ndarray[DTYPE_t, ndim=2] a2 = sigmoid(a1 @ Theta1.T)  # Applying the sigmoid function here
    a2 = np.hstack((np.ones((m, 1)), a2))
    cdef np.ndarray[DTYPE_t, ndim=2] a3 = sigmoid(a2 @ Theta2.T)  # Applying the sigmoid function here

    # One-hot encoding of y
    cdef np.ndarray[DTYPE_t, ndim=2] y_mtx = np.equal.outer(y.ravel(), np.arange(num_labels)).astype(DTYPE)

    # Cost function
    cdef double J = np.sum(-y_mtx * np.log(a3) - (1.0 - y_mtx) * np.log(1.0 - a3)) / m

    # Add regularization
    J += lmbda / (2.0 * m) * (np.sum(Theta1[:, 1:] ** 2) + np.sum(Theta2[:, 1:] ** 2))

    return J