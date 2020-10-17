import numpy as np
from proj1_helpers import *


def SGD(y, tx, initial_w, max_iters, gamma, loss_kind, batch_size):
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
  batch_size : int 
    size of a minibatch
  
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
  n_iter = 0
  start_id = 0
  end_id = batch_size
  
  while n_iter < max_iters:
    # Reshuffle indexes at the beginning of epoch if minibatches are used
    if start_id == 0 and batch_size != N:
      np.random.shuffle(indexes) 
      
    x_n = tx[indexes[start_id:end_id]].reshape(batch_size,-1)
    y_n = y[indexes[start_id:end_id]]
    sg = gradient_function(y_n, x_n, w)
    
    w = w - gamma * sg
    
    n_iter += 1
    start_id = start_id + batch_size
    if start_id >= N:
      start_id = 0
    
    end_id = start_id + batch_size
    # Taking care of potentially smaller last minibatch
    if end_id > N:
      end_id = N
    
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
  """Ridge regression using normal equations

  Parameters
  ----------
  y : numpy array
    Targets vector (N,) or (N,1)
  tx : numpy array
    Feature matrix (N,D)
  lambda_ : float
    Regularisation parameter
  
  Returns
  -------
  (w, loss) : (numpy array (D,1), float)
    weights and loss after ridge regression
  """
  y, tx = prepare_dimensions(y, tx)
  N, D = np.shape(tx)
  w, _, _, _ = np.linalg.lstsq(tx.T @ tx + 2 * N * lambda_ * np.eye(D), tx.T @ y, rcond=None)
  loss = compute_mse_loss(y, tx, w)
  return (w, loss)


def logistic_regression(y, tx, initial_w, max_iters, gamma):
  """Logistic regression using gradient descent or SGD"""
  return SGD(y, tx, initial_w, max_iters, gamma, "LOGISTIC_REGRESSION", 1)


def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
  """Regularized logistic regression using gradient descent or SGD"""
  return SGD(y, tx, initial_w, max_iters, gamma, "REGULARIZED_LOGISTIC_REGRESSION", 1)
