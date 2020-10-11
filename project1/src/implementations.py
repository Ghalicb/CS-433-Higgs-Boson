import numpy as np
from proj1_helpers import *

loss_kinds = { 
  "LEAST_SQUARE" : (compute_mse_loss, compute_mse_gradient),
  "LOGISTIC_REGRESSION" : (compute_logistic_loss, compute_logistic_gradient)
}

#TODO Implement minibatch

def SGD(y, tx, initial_w, max_iters, gamma, loss_kind, mini_batch):
  """Linear regression using Stochastic Gradient Descent

  Parameters
  ----------
  y : numpy array
    Targets vector (N,) or (N,1)
  tx : numpy array
    Feature matrix (N,D)
  initial_w : numpy array
    weights vector (D,) or (D,1)
  max_iters : int
    number of iteration to run SGD
  gamma : float
    learning rate
  loss_kind : string
    can take value in { "LEAST_SQUARE" , "LOGISTIC_REGRESSION" }
  
  Returns
  -------
  (w, loss) : (numpy array (D,1), float)
      weights and loss after max_iters iterations of SGD
  """
  y, tx = prepare_dimensions(y, tx)
  N = len(y)
  w = initial_w.reshape(-1,1)
  loss_function, gradient_function = loss_kinds[loss_kind]
    
  indexes = np.arange(N)

  for n_iter in range(max_iters):
    if mini_batch != N and n_iter%N==0:
      #shuffle data before new pass on data, inplace
      np.random.shuffle(indexes) 
      
    n = indexes[n_iter%N]
    x_n = tx[n].reshape(1,-1)
    sg = gradient_function(y, x_n, w)
    
    w = w - gamma * sg
    
  loss = loss_function(y, tx, w)
  return (w, loss)

def least_squares_GD(y, tx, initial_w, max_iters, gamma):
  """Linear regression using gradient descent.
  A special case of SGD with a MSE loss and a mini-batch size of N

  Parameters
  ----------
  y : numpy array
    Targets vector (N,) or (N,1)
  tx : numpy array
    Feature matrix (N,D)
  initial_w : numpy array
    weights vector (D,) or (D,1)
  max_iters : int
    number of iteration to run SGD
  gamma : float
    learning rate
  
  Returns
  -------
  (w, loss) : (numpy array (D,1), float)
      weights and loss after max_iters iterations of GD
  """
  return SGD(y, tx, initial_w, max_iters, gamma, "LEAST_SQUARE", len(y))

def least_squares_SGD(y, tx, initial_w, max_iters, gamma):
  """Linear regression using Stochastic Gradient Descent.
  A special case of SGD with a MSE loss and a mini-batch size of 1

  Parameters
  ----------
  y : numpy array
    Targets vector (N,) or (N,1)
  tx : numpy array
    Feature matrix (N,D)
  initial_w : numpy array
    weights vector (D,) or (D,1)
  max_iters : int
    number of iteration to run SGD
  gamma : float
    learning rate
  loss_kind : string
    can take value in { "LEAST_SQUARE" , "LOGISTIC_REGRESSION" }
  
  Returns
  -------
  (w, loss) : (numpy array (D,1), float)
      weights and loss after max_iters iterations of SGD
  """
  return SGD(y, tx, initial_w, max_iters, gamma, "LEAST_SQUARE", 1)
    

def least_squares(y, tx):
  """Least squares regression using normal equations"""
  return ridge_regression(y, tx, 0)


def ridge_regression(y, tx, lambda_):
  """Ridge regression using normal equations"""
  y, tx = prepare_dimensions(y, tx)
  N, D = np.shape(tx)
  w, _, _, _ = np.linalg.lstsq(tx.T @ tx + 2 * N * lambda_ * np.eye(D), tx.T @ y, rcond=None)
  loss = compute_mse_loss(y, tx, w)
  return (w, loss)


def logistic_regression(y, tx, initial_w, max_iters, gamma):
  """Logistic regression using gradient descent or SGD"""
  return None#SGD(y, tx, 0, initial_w, max_iters, gamma, "LOGISTIC_REGRESSION", 1)


def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
  """Regularized logistic regression using gradient descent or SGD"""
  return None#SGD(y, tx, 0, initial_w, max_iters, gamma, "REGULARIZED_LOGISTIC_REGRESSION", 1)
