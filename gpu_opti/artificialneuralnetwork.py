import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

PLOT = False

import argparse
import matplotlib.pyplot as plt
import pandas as pd
from scipy import optimize
from functools import partial
from timeit import default_timer as timer

"""
Create Your Own Artificial Neural Network for Multi-class Classification (With Python)
Philip Mocz (2023), @PMocz

Create and train your own artificial neural network to classify images of galaxies from SDSS/the Galaxy Zoo project.

"""

def g(x):
    """ sigmoid function """
    return 1.0 / (1.0 + torch.exp(-x))

def grad_g(x):
    """ gradient of sigmoid function """
    gx = g(x)
    return gx * (1.0 - gx)

def predict(Theta1, Theta2, X):
    """ Predict labels in a trained three layer classification network.
    Input:
      Theta1       trained weights applied to 1st layer (hidden_layer_size x input_layer_size+1)
      Theta2       trained weights applied to 2nd layer (num_labels x hidden_layer_size+1)
      X            matrix of training data      (m x input_layer_size)
    Output:
      prediction   label prediction
    """

    m = X.shape[0]
    num_labels = Theta2.shape[0]

    a1 = torch.cat((torch.ones((m,1), device=device), X), dim=1)
    a2 = g(a1 @ Theta1.T)
    a2 = torch.cat((torch.ones((m,1), device=device), a2), dim=1)
    a3 = g(a2 @ Theta2.T)

    prediction = torch.argmax(a3, dim=1).reshape((m,1))
    return prediction

def reshape(theta, input_layer_size, hidden_layer_size, num_labels):
    """ reshape theta into Theta1 and Theta2, the weights of our neural network """
    ncut = hidden_layer_size * (input_layer_size+1)
    Theta1 = theta[0:ncut].reshape(hidden_layer_size, input_layer_size+1)
    Theta2 = theta[ncut:].reshape(num_labels, hidden_layer_size+1)
    return Theta1, Theta2

def cost_function(theta, input_layer_size, hidden_layer_size, num_labels, X, y, lmbda):
    """ Neural net cost function for a three layer classification network. """

    # convert theta to tensor if it's not already the case
    if not isinstance(theta, torch.Tensor):
        theta = torch.tensor(theta, dtype=torch.float32, device=device)

    Theta1, Theta2 = reshape(theta, input_layer_size, hidden_layer_size, num_labels)

    m = len(y)

    a1 = torch.cat((torch.ones((m,1), device=device), X), dim=1)
    a2 = g(a1 @ Theta1.T)
    a2 = torch.cat((torch.ones((m,1), device=device), a2), dim=1)
    a3 = g(a2 @ Theta2.T)

    # Convert to one-hot encoding
    y_mtx = torch.zeros((m, num_labels), device=device)
    y_mtx[torch.arange(m, device=device), y.flatten().long()] = 1.0

    J = torch.sum(-y_mtx * torch.log(a3) - (1.0-y_mtx) * torch.log(1.0-a3)) / m

    J += lmbda/(2.*m) * (torch.sum(Theta1[:,1:]**2) + torch.sum(Theta2[:,1:]**2))

    return J.item()  # convert to scalar for scipy optimizer

def gradient(theta, input_layer_size, hidden_layer_size, num_labels, X, y, lmbda):
    """ Neural net cost function gradient for a three layer classification network. """

    # convert theta to tensor if it's not already the case
    if not isinstance(theta, torch.Tensor):
        theta = torch.tensor(theta, dtype=torch.float32, device=device)

    Theta1, Theta2 = reshape(theta, input_layer_size, hidden_layer_size, num_labels)

    m = X.shape[0]

    X_bias = torch.cat((torch.ones((m, 1), device=device), X), dim=1)
    Z2 = X_bias @ Theta1.T
    A2 = g(Z2)
    A2_bias = torch.cat((torch.ones((m, 1), device=device), A2), dim=1)
    Z3 = A2_bias @ Theta2.T
    A3 = g(Z3)

    Y = torch.zeros((m, num_labels), device=device)
    Y[torch.arange(m, device=device), y.flatten().long()] = 1

    Delta3 = A3 - Y
    Delta2 = (Delta3 @ Theta2[:, 1:]) * grad_g(Z2)

    Theta1_grad = (Delta2.T @ X_bias) / m
    Theta2_grad = (Delta3.T @ A2_bias) / m

    Theta1_grad[:, 1:] += (lmbda / m) * Theta1[:, 1:]
    Theta2_grad[:, 1:] += (lmbda / m) * Theta2[:, 1:]

    grad = torch.cat((Theta1_grad.flatten(), Theta2_grad.flatten()))

    return grad.cpu().numpy()  # convert to numpy array for scipy optimizer

def callbackF(input_layer_size, hidden_layer_size, num_labels, X, y, lmbda, test, test_label, theta_k, max_iter):
    """ Calculate some stats per iteration and update plot """
    global N_iter, J_min, theta_best, Js_train, Js_test

    # convert theta_k to tensor if it's not already the case
    if not isinstance(theta_k, torch.Tensor):
        theta_k = torch.tensor(theta_k, dtype=torch.float32, device=device)

    Theta1, Theta2 = reshape(theta_k, input_layer_size, hidden_layer_size, num_labels)

    J = cost_function(theta_k, input_layer_size, hidden_layer_size, num_labels, X, y, lmbda)
    y_pred = predict(Theta1, Theta2, X)
    accuracy = torch.sum(y_pred == y).float() / len(y)
    Js_train = torch.cat((Js_train, torch.tensor([J], device=device)))

    J_test = cost_function(theta_k, input_layer_size, hidden_layer_size, num_labels, test, test_label, lmbda)
    test_pred = predict(Theta1, Theta2, test)
    accuracy_test = torch.sum(test_pred == test_label).float() / len(test_label)
    Js_test = torch.cat((Js_test, torch.tensor([J_test], device=device)))

    print(f'iter={N_iter:3d}:  Jtrain= {J:0.4f} acc= {100*accuracy:0.2f}%  |  Jtest= {J_test:0.4f} acc= {100*accuracy_test:0.2f}%')
    N_iter += 1

    if (J_test < J_min):
        theta_best = theta_k
        J_min = J_test

    if PLOT:
        plt.clf()
        plt.subplot(2,1,1)

        im_size = 32
        pad = 4
        galaxies_image = torch.zeros((3*im_size, 6*im_size+2*pad), device=device) + 255

        for i in range(3):
            for j in range(6):
                idx = 3*j + i + 900*(j>1) + 900*(j>3) + (N_iter % max_iter)
                shift = 0 + pad*(j>1) + pad*(j>3)
                ii = i * im_size
                jj = j * im_size + shift
                galaxies_image[ii:ii+im_size,jj:jj+im_size] = X[idx].reshape(im_size,im_size) * 255
                my_label = 'E' if y_pred[idx]==0 else 'S' if y_pred[idx]==1 else 'I'
                my_color = 'blue' if (y_pred[idx]==y[idx]) else 'red'
                plt.text(jj+2, ii+10, my_label, color=my_color)
                if (y_pred[idx]==y[idx]):
                    plt.text(jj+4, ii+25, "✓", color='blue', fontsize=50)

        plt.imshow(galaxies_image.cpu().numpy(), cmap='gray')
        plt.gca().axis('off')
        plt.subplot(2,1,2)
        plt.plot(Js_test.cpu().numpy(), 'r', label='test')
        plt.plot(Js_train.cpu().numpy(), 'b', label='train')
        plt.xlabel("iteration")
        plt.ylabel("cost")
        plt.xlim(0,max_iter)
        plt.ylim(1,2.1)
        plt.gca().legend()
        plt.pause(0.001)

def main(max_iter):
    """ Artificial Neural Network for classifying galaxies """

    torch.manual_seed(917)

    train = pd.read_csv('train.csv', delimiter=',').values
    test = pd.read_csv('test.csv', delimiter=',').values

    # convert to tensors and move to GPU
    train_label = torch.tensor(train[:,0], device=device).reshape(len(train),1)
    test_label = torch.tensor(test[:,0], device=device).reshape(len(test),1)

    # normalize image data to [0,1]
    train = torch.tensor(train[:,1:], dtype=torch.float32, device=device) / 255.
    test = torch.tensor(test[:,1:], dtype=torch.float32, device=device) / 255.

    X = train
    y = train_label

    m = X.shape[0]
    input_layer_size = X.shape[1]
    hidden_layer_size = 8
    num_labels = 3
    lmbda = 1.0

    Theta1 = (torch.rand(hidden_layer_size, input_layer_size+1, device=device) * 0.4 - 0.2)
    Theta2 = (torch.rand(num_labels, hidden_layer_size+1, device=device) * 0.4 - 0.2)

    theta0 = torch.cat((Theta1.flatten(), Theta2.flatten()))

    J = cost_function(theta0, input_layer_size, hidden_layer_size, num_labels, X, y, lmbda)
    print('initial cost function J =', J)

    train_pred = predict(Theta1, Theta2, train)
    print('initial accuracy on training set =',
          torch.sum(train_pred == train_label).float()/len(train_label))

    global Js_train, Js_test
    Js_train = torch.tensor([J], device=device)
    J_test = cost_function(theta0, input_layer_size, hidden_layer_size, num_labels,
                          test, test_label, lmbda)
    Js_test = torch.tensor([J_test], device=device)

    if PLOT:
        plt.figure(figsize=(6,6), dpi=80)

    args = (input_layer_size, hidden_layer_size, num_labels, X, y, lmbda)
    cbf = partial(callbackF, input_layer_size, hidden_layer_size, num_labels,
                 X, y, lmbda, test, test_label, max_iter=max_iter)

    theta = optimize.fmin_cg(cost_function, theta0.cpu().numpy(), fprime=gradient,
                           args=args, callback=cbf, maxiter=max_iter)

    # convert best theta back to tensor
    theta_best_tensor = torch.tensor(theta_best, device=device)

    Theta1, Theta2 = reshape(theta_best_tensor, input_layer_size, hidden_layer_size, num_labels)

    train_pred = predict(Theta1, Theta2, train)
    test_pred = predict(Theta1, Theta2, test)

    print('accuracy on training set =',
          torch.sum(train_pred == train_label).float()/len(train_label))
    print('accuracy on test set =',
          torch.sum(test_pred == test_label).float()/len(test_label))

    if PLOT:
        plt.savefig('artificialneuralnetwork.png',dpi=240)
        plt.show()

    return 0

max_iters = [10, 20, 30, 40, 50, 100, 200, 300, 400, 500, 600]
iter_times = []

for max_iter in max_iters:
  N_iter = 1
  J_min = float('inf')
  theta_best = []
  Js_train = torch.tensor([], device=device)
  Js_test = torch.tensor([], device=device)
  start_time = timer()
  main(max_iter)
  end_time = timer()
  iter_times.append(end_time - start_time)
  print(f"\nTotal execution time for ${max_iter} iterations: {end_time - start_time:.6f} seconds")

for i, t in enumerate(iter_times):
  print(f"{max_iters[i]} {t}")