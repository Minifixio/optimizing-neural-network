import numpy as np
cimport numpy as np

DTYPE = np.float64
ctypedef np.float64_t DTYPE_t

cdef np.ndarray[np.float64_t, ndim=2] sigmoid(np.ndarray[np.float64_t, ndim=2] x):
    return 1.0 / (1.0 + np.exp(-x))

cpdef np.ndarray[np.int64_t, ndim=2] predict(np.ndarray[DTYPE_t, ndim=2] Theta1,
                                        np.ndarray[DTYPE_t, ndim=2] Theta2,
                                        np.ndarray[DTYPE_t, ndim=2] X):
    """Predict labels in a trained three layer classification network."""

    cdef int m = X.shape[0]
    cdef int num_labels = Theta2.shape[0]

    cdef np.ndarray[DTYPE_t, ndim=2] a1 = np.hstack((np.ones((m, 1)), X))  # Add bias (input layer)
    cdef np.ndarray[DTYPE_t, ndim=2] z2 = a1 @ Theta1.T
    cdef np.ndarray[DTYPE_t, ndim=2] a2 = sigmoid(z2)  # Sigmoid function

    a2 = np.hstack((np.ones((m, 1)), a2))  # Add bias (hidden layer)
    cdef np.ndarray[DTYPE_t, ndim=2] z3 = a2 @ Theta2.T
    cdef np.ndarray[DTYPE_t, ndim=2] a3 = sigmoid(z3)  # Sigmoid function

    cdef np.ndarray[np.int64_t, ndim=1] prediction_1d = np.argmax(a3, axis=1)
    cdef np.ndarray[np.int64_t, ndim=2] prediction = prediction_1d.reshape((m, 1)).astype(np.int64)

    return prediction