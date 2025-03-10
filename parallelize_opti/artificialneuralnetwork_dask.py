import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import optimize
from functools import partial
from timeit import default_timer as timer
import numexpr as ne
import statistics
from functools import wraps
from multiprocessing import Pool, cpu_count

# ---------------------------
# Timing utilities (unchanged)
# ---------------------------
class TimerStats:
    def __init__(self):
        self.times = []
    def add_time(self, t):
        self.times.append(t)
    def compute_stats(self):
        if not self.times:
            return None, None
        avg_time = sum(self.times) / len(self.times)
        std_time = statistics.stdev(self.times) if len(self.times) > 1 else 0
        return avg_time, std_time
    def report(self, function_name):
        avg, std = self.compute_stats()
        print(f"\nExecution time statistics for {function_name}:")
        print(f"Average execution time: {avg:.6f} seconds")
        print(f"Standard deviation: {std:.6f} seconds")

def timefn(fn, timer_stats=None):
    if timer_stats is None:
        timer_stats = TimerStats()
    @wraps(fn)
    def measure_time(*args, **kwargs):
        t1 = timer()
        result = fn(*args, **kwargs)
        t2 = timer()
        elapsed_time = t2 - t1
        timer_stats.add_time(elapsed_time)
        return result
    measure_time.stats = timer_stats
    return measure_time

# ---------------------------
# Global parameters
# ---------------------------
MAX_ITER = 300
PLOT = False

def g(x):
    """ sigmoid function """
    return 1.0 / (1.0 + np.exp(-x))

def grad_g(x):
    """ gradient of sigmoid function """
    gx = g(x)
    return gx * (1.0 - gx)

@timefn
def predict(Theta1, Theta2, X):
    """ Predict labels in a trained three layer classification network. """
    m = np.shape(X)[0]
    num_labels = np.shape(Theta2)[0]
    a1 = np.hstack((np.ones((m,1)), X))
    a2 = g(a1 @ Theta1.T)
    a2 = np.hstack((np.ones((m,1)), a2))
    a3 = g(a2 @ Theta2.T)
    prediction = np.argmax(a3, 1).reshape((m,1))
    return prediction

def reshape(theta, input_layer_size, hidden_layer_size, num_labels):
    """ reshape theta into Theta1 and Theta2 """
    ncut = hidden_layer_size * (input_layer_size+1)
    Theta1 = theta[0:ncut].reshape(hidden_layer_size, input_layer_size+1)
    Theta2 = theta[ncut:].reshape(num_labels, hidden_layer_size+1)
    return Theta1, Theta2

@timefn    
def cost_function(theta, input_layer_size, hidden_layer_size, num_labels, X, y, lmbda):
    """ Neural net cost function for a three layer classification network. """
    Theta1, Theta2 = reshape(theta, input_layer_size, hidden_layer_size, num_labels)
    m = len(y)
    a1 = np.hstack((np.ones((m,1)), X))
    a2 = g(a1 @ Theta1.T)
    a2 = np.hstack((np.ones((m,1)), a2))
    a3 = g(a2 @ Theta2.T)
    y_mtx = np.equal.outer(y.ravel(), np.arange(num_labels)).astype(float)
    J = np.sum(-y_mtx * np.log(a3) - (1.0-y_mtx) * np.log(1.0-a3)) / m
    J += lmbda/(2.*m) * (np.sum(Theta1[:,1:]**2)  + np.sum(Theta2[:,1:]**2))
    return J

@timefn
def gradient(theta, input_layer_size, hidden_layer_size, num_labels, X, y, lmbda):
    """ Neural net gradient for a three layer classification network. """
    Theta1, Theta2 = reshape(theta, input_layer_size, hidden_layer_size, num_labels)
    m = X.shape[0]
    X_bias = np.hstack((np.ones((m,1)), X))
    Z2 = X_bias @ Theta1.T
    A2 = g(Z2)
    A2_bias = np.hstack((np.ones((m,1)), A2))
    Z3 = A2_bias @ Theta2.T
    A3 = g(Z3)
    Y = np.zeros((m, num_labels))
    Y[np.arange(m), y.flatten().astype(int)] = 1
    Delta3 = A3 - Y
    Delta2 = (Delta3 @ Theta2[:,1:]) * grad_g(Z2)
    Theta1_grad = (Delta2.T @ X_bias) / m
    Theta2_grad = (Delta3.T @ A2_bias) / m
    Theta1_grad[:,1:] += (lmbda/m)*Theta1[:,1:]
    Theta2_grad[:,1:] += (lmbda/m)*Theta2[:,1:]
    grad_flat = np.concatenate((Theta1_grad.ravel(), Theta2_grad.ravel()))
    return grad_flat

N_iter = 1
J_min = np.Inf
theta_best = []
Js_train = np.array([])
Js_test = np.array([])

def callbackF(input_layer_size, hidden_layer_size, num_labels, X, y, lmbda, test, test_label, theta_k):
    """ Calculate some stats per iteration and update plot """
    global N_iter, J_min, theta_best, Js_train, Js_test
    Theta1, Theta2 = reshape(theta_k, input_layer_size, hidden_layer_size, num_labels)
    J = cost_function(theta_k, input_layer_size, hidden_layer_size, num_labels, X, y, lmbda)
    y_pred = predict(Theta1, Theta2, X)
    accuracy = np.sum(1.*(y_pred==y)) / len(y)
    Js_train = np.append(Js_train, J)
    J_test = cost_function(theta_k, input_layer_size, hidden_layer_size, num_labels, test, test_label, lmbda)
    test_pred = predict(Theta1, Theta2, test)
    accuracy_test = np.sum(1.*(test_pred==test_label)) / len(test_label)
    Js_test = np.append(Js_test, J_test)
    print('iter={:3d}:  Jtrain= {:0.4f} acc= {:0.2f}%  |  Jtest= {:0.4f} acc= {:0.2f}%'
          .format(N_iter, J, 100*accuracy, J_test, 100*accuracy_test))
    N_iter += 1
    if (J_test < J_min):
        theta_best = theta_k
        J_min = J_test
    # Plot update omitted for parallel version
    return

def optimize_ann(initial_guess, args, cbf, max_iter):
    """ Run optimization from an initial guess """
    res = optimize.fmin_cg(cost_function, initial_guess, fprime=gradient, args=args, callback=cbf, maxiter=max_iter, disp=False)
    final_cost = cost_function(res, *args)
    return res, final_cost

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Artificial Neural Network for classifying galaxies')
    parser.add_argument('--max_iter', type=int, default=300, help='Maximum number of iterations')
    parser.add_argument('--plot', type=bool, default=False, help='Plot the results')
    args = parser.parse_args()
    MAX_ITER = args.max_iter
    PLOT = args.plot

    start_time = timer()

    # Load datasets
    train = pd.read_csv('train.csv', delimiter=',').values
    test = pd.read_csv('test.csv', delimiter=',').values
    train_label = train[:,0].reshape(len(train),1)
    test_label = test[:,0].reshape(len(test),1)
    train = train[:,1:] / 255.
    test = test[:,1:] / 255.
    X = train
    y = train_label

    m = np.shape(X)[0]
    input_layer_size = np.shape(X)[1]
    hidden_layer_size = 8
    num_labels = 3
    lmbda = 1.0

    Theta1 = np.random.rand(hidden_layer_size, input_layer_size+1) * 0.4 - 0.2
    Theta2 = np.random.rand(num_labels, hidden_layer_size+1) * 0.4 - 0.2
    theta0 = np.concatenate((Theta1.flatten(), Theta2.flatten()))

    J = cost_function(theta0, input_layer_size, hidden_layer_size, num_labels, X, y, lmbda)
    print('initial cost function J =', J)
    train_pred = predict(Theta1, Theta2, train)
    print('initial accuracy on training set =', np.sum(1.*(train_pred==train_label))/len(train_label))
    Js_train = np.array([J])
    J_test = cost_function(theta0, input_layer_size, hidden_layer_size, num_labels, test, test_label, lmbda)
    Js_test = np.array([J_test])

    num_initializations = 2
    initial_guesses = [np.random.rand(theta0.shape[0]) for _ in range(num_initializations)]
    args_opt = (input_layer_size, hidden_layer_size, num_labels, X, y, lmbda)
    cbf_partial = partial(callbackF, input_layer_size, hidden_layer_size, num_labels, X, y, lmbda, test, test_label)
    
    with Pool(processes=cpu_count()) as pool:
        results = pool.starmap(optimize_ann, [(init, args_opt, cbf_partial, MAX_ITER) for init in initial_guesses])
    
    best_result = min(results, key=lambda r: r[1])
    theta_opt, best_cost = best_result
    print("Best final cost:", best_cost)

    Theta1, Theta2 = reshape(theta_opt, input_layer_size, hidden_layer_size, num_labels)
    train_pred = predict(Theta1, Theta2, train)
    test_pred = predict(Theta1, Theta2, test)
    print('accuracy on training set =', np.sum(1.*(train_pred==train_label))/len(train_label))
    print('accuracy on test set =', np.sum(1.*(test_pred==test_label))/len(test_label))

    end_time = timer()
    print(f"\nTotal execution time: {end_time - start_time:.6f} seconds")
