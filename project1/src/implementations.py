import numpy as np
from helpers import *

def least_squares_GD(y, tx, initial_w, max_iters, gamma):
  """Linear regression using gradient descent"""
  N = len(y)
  w = initial_w

  for n_iter in range(max_iters):
    gradient = -1/N * tx.T @ (y - tx @ w)
    w = w - gamma * gradient
    # print("Gradient Descent({bi}/{ti}): loss={l}".format( bi=n_iter, ti=max_iters - 1, l=(1/(2*N) * (y - tx.T @ w).T @ (y - tx.T @ w))))

  loss = compute_mse_loss(y, tx, w)
  return (w, loss)

def least_squares_SGD(y, tx, initial_w, max_iters, gamma):
  """Linear regression using stochastic gradient descent"""
  return (w, loss)

def least_squares(y, tx):
  """Least squares regression using normal equations"""
  return (w, loss)

def ridge_regression(y, tx, lambda_):
  """Ridge regression using normal equations"""
  return (w, loss)

def logistic_regression(y, tx, initial_w, max_iters, gamma):
  """Logistic regression using gradient descent or SGD"""
  return (w, loss)

def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
  """Regularized logistic regression using gradient descent or SGD"""
  return (w, loss)

