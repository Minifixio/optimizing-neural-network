import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import optimize
from functools import partial
import jax
import jax.numpy as jnp
from jax import grad, jit, vmap, lax
from timeit import default_timer as timer

# enable 64-bit precision for JAX
jax.config.update("jax_enable_x64", True)

MAX_ITER = 300
PLOT = False

def g(x):
    return 1.0 / (1.0 + jnp.exp(-x))

def grad_g(x):
    gx = g(x)
    return gx * (1.0 - gx)

# predict function with JAX
@jit
def predict(Theta1, Theta2, X):
    """ Predict labels in a trained three layer classification network. """
    m = X.shape[0]
    a1 = jnp.hstack((jnp.ones((m, 1)), X))
    a2 = g(a1 @ Theta1.T)
    a2 = jnp.hstack((jnp.ones((m, 1)), a2))
    a3 = g(a2 @ Theta2.T)
    prediction = jnp.argmax(a3, axis=1).reshape((m, 1))
    return prediction

def reshape(theta, input_layer_size, hidden_layer_size, num_labels):
    """ Reshape theta into Theta1 and Theta2, the weights of our neural network. """
    ncut = hidden_layer_size * (input_layer_size + 1)
    Theta1 = lax.dynamic_slice(theta, (0,), (ncut,)).reshape(hidden_layer_size, input_layer_size + 1)
    Theta2 = lax.dynamic_slice(theta, (ncut,), (len(theta) - ncut,)).reshape(num_labels, hidden_layer_size + 1)
    return Theta1, Theta2

@partial(jax.jit, static_argnums=[1, 2, 3])
def cost_function(theta, input_layer_size, hidden_layer_size, num_labels, X, y, lmbda):
    """ Neural net cost function for a three layer classification network. """
    Theta1, Theta2 = reshape(theta, input_layer_size, hidden_layer_size, num_labels)
    m = len(y)
    a1 = jnp.hstack((jnp.ones((m, 1)), X))
    a2 = g(a1 @ Theta1.T)
    a2 = jnp.hstack((jnp.ones((m, 1)), a2))
    a3 = g(a2 @ Theta2.T)

    y_mtx = 1.*(y==0)
    for k in range(1,num_labels):
        y_mtx = jnp.hstack((y_mtx, 1.*(y==k)))
    J = jnp.sum(-y_mtx * jnp.log(a3) - (1.0 - y_mtx) * jnp.log(1.0 - a3)) / m
    J += lmbda / (2.0 * m) * (jnp.sum(Theta1[:, 1:] ** 2) + jnp.sum(Theta2[:, 1:] ** 2))
    return J

# gradient function using JAX's automatic differentiation
@partial(jax.jit, static_argnums=[1, 2, 3])
def gradient(theta, input_layer_size, hidden_layer_size, num_labels, X, y, lmbda):
    return grad(cost_function)(theta, input_layer_size, hidden_layer_size, num_labels, X, y, lmbda)

N_iter = 1
J_min = jnp.inf
theta_best = []
Js_train = jnp.array([])
Js_test = jnp.array([])

# redefine callbackF to work with static data and JAX
def callbackF(input_layer_size, hidden_layer_size, num_labels, X, y, lmbda, test, test_label, theta_k):
    global N_iter, J_min, theta_best, Js_train, Js_test

    Theta1, Theta2 = reshape(theta_k, input_layer_size, hidden_layer_size, num_labels)

    J = cost_function(theta_k, input_layer_size, hidden_layer_size, num_labels, X, y, lmbda)
    y_pred = predict(Theta1, Theta2, X)
    accuracy = jnp.sum(1.0 * (y_pred == y)) / len(y)

    J_test = cost_function(theta_k, input_layer_size, hidden_layer_size, num_labels, test, test_label, lmbda)
    test_pred = predict(Theta1, Theta2, test)
    accuracy_test = jnp.sum(1.0 * (test_pred == test_label)) / len(test_label)

    global Js_train, Js_test
    Js_train = jnp.append(Js_train, J)
    Js_test = jnp.append(Js_test, J_test)

    print('iter={:3d}:  Jtrain= {:0.4f} acc= {:0.2f}%  |  Jtest= {:0.4f} acc= {:0.2f}%'.format(N_iter, J, 100*accuracy, J_test, 100*accuracy_test))
    N_iter += 1

    if (J_test < J_min):
        theta_best = theta_k
        J_min = J_test

    return J_test


def main():
    np.random.seed(917)
    train = pd.read_csv('train.csv', delimiter=',').values
    test = pd.read_csv('test.csv', delimiter=',').values
    train_label = train[:, 0].reshape(len(train), 1)
    test_label = test[:, 0].reshape(len(test), 1)
    train = train[:, 1:] / 255.0
    test = test[:, 1:] / 255.0

    # convert data to JAX arrays
    X = jnp.array(train[:,1:], dtype=jnp.float64) # use float64 
    y = jnp.array(train_label, dtype=jnp.int64) # use int64 to be used as indexes
    test_data = jnp.array(test[:,1:], dtype=jnp.float64)
    test_label = jnp.array(test_label, dtype=jnp.int64)

    m, input_layer_size = X.shape
    hidden_layer_size = 8
    num_labels = 3
    lmbda = 1.0
    Theta1 = np.random.rand(hidden_layer_size, input_layer_size + 1) * 0.4 - 0.2
    Theta2 = np.random.rand(num_labels, hidden_layer_size + 1) * 0.4 - 0.2
    theta0 = jnp.concatenate((jnp.array(Theta1.flatten()), jnp.array(Theta2.flatten())))
    J = cost_function(theta0, input_layer_size, hidden_layer_size, num_labels, X, y, lmbda)
    print('Initial cost function J =', J)

    Theta1_start, Theta2_start = reshape(theta0, input_layer_size, hidden_layer_size, num_labels)
    train_pred = predict(Theta1_start, Theta2_start, X)
    print('Initial accuracy on training set =', jnp.sum(1.0 * (train_pred == y)) / len(y))
    global Js_train, Js_test
    Js_train = jnp.array([J])
    J_test = cost_function(theta0, input_layer_size, hidden_layer_size, num_labels, test_data, test_label, lmbda)
    Js_test = jnp.array([J_test])

    args = (input_layer_size, hidden_layer_size, num_labels, X, y, lmbda)
    cbf = partial(callbackF, input_layer_size, hidden_layer_size, num_labels, X, y, lmbda, test_data, test_label)
    theta = optimize.fmin_cg(cost_function, theta0, fprime=gradient, args=args, callback=cbf, maxiter=MAX_ITER)

    Theta1, Theta2 = reshape(theta_best, input_layer_size, hidden_layer_size, num_labels)
    train_pred = predict(Theta1, Theta2, X)
    test_pred = predict(Theta1, Theta2, test_data)
    print('Accuracy on training set =', jnp.sum(1.0 * (train_pred == y)) / len(y))
    print('Accuracy on test set =', jnp.sum(1.0 * (test_pred == test_label)) / len(test_label))

    return 0

def stdTime(n_runs=10):
    """Compute the standard deviation of execution times over multiple runs."""
    times = []
    global N_iter, J_min, theta_best, Js_train, Js_test
    for _ in range(n_runs):
        N_iter = 1
        J_min = jnp.inf
        theta_best = []
        Js_train = jnp.array([])
        Js_test = jnp.array([])
        start_time = timer()
        main()
        end_time = timer()
        times.append(end_time - start_time)
    
    mean_time = np.mean(times)
    std_dev = np.std(times)
    
    print(f"\nMean execution time: {mean_time:.6f} seconds")
    print(f"Standard deviation: {std_dev:.6f} seconds")
    
    return mean_time, std_dev

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Artificial Neural Network for classifying galaxies')
    parser.add_argument('--max_iter', type=int, default=300, help='Maximum number of iterations')
    parser.add_argument('--plot', type=bool, default=False, help='Plot the results')
    args = parser.parse_args()
    MAX_ITER = args.max_iter
    PLOT = args.plot

    mean_time, std_dev = stdTime(5)
    
    # start_time = timer()
    # main()
    # end_time = timer()
    # print(f"\nTotal execution time: {end_time - start_time:.6f} seconds")