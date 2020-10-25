import numpy as np
from helpers import prepare_dimensions
from losses import *


loss_kinds = { 
  "LEAST_SQUARE" : (compute_mse_loss, compute_mse_gradient),
  "LOGISTIC_REGRESSION" : (compute_logistic_loss, compute_logistic_gradient),
  "REGULARIZED_LOGISTIC_REGRESSION" : (compute_regularized_logistic_loss, compute_regularized_logistic_gradient)
}


def SGD(y, tx, initial_w, max_iters, gamma, loss_kind, batch_size, lambda_ = 0, verbose = False, validation_y = None, validation_tx = None):
  """Linear regression using Stochastic Gradient Descent

  Parameters
  ----------
  y : numpy array
    Targets vector (N,) or (N,1)
  tx : numpy array
    Feature matrix (N,D)
  initial_w : numpy array
    Weights vector (D,) or (D,1)
  max_iters : int
    Number of iteration to run SGD
  gamma : float
    Learning rate
  loss_kind : string
    Can take value in { "LEAST_SQUARE" , "LOGISTIC_REGRESSION", "REGULARIZED_LOGISTIC_REGRESSION"}
  batch_size : int
    Size of a minibatch
  lambda_ : float, optional
    Regularization parameter to use if loss_kind = "REGULARIZED_LOGISTIC_REGRESSION". The default is 0
  verbose : bool, optional 
    Whether to return lists of train errors and validation errors. The default is False
  validation_y : numpy array, optional
    Validation target vecor (N*, 1). The default is None. Specify if verbose is True.
  validation_tx : numpy vector, optional
    Validation feature matrix (N*, D). The default is None. Specify if verbose is True.

   Returns
  -------
  (w, loss) : (numpy array, float)
      weights (D,1) and loss after max_iters iterations of SGD
  """
  y, tx = prepare_dimensions(y, tx)
  N = len(y)
  w = initial_w.reshape(-1,1)
  loss_function, gradient_function = loss_kinds[loss_kind]
    
  indexes = np.arange(N)
  n_iter = 0
  start_id = 0
  end_id = batch_size
  train_errors = []
  validation_errors = []
  while n_iter < max_iters:
    # Reshuffle indexes at the beginning of epoch if minibatches is not whole dataset
    if start_id == 0 and batch_size != N:
      np.random.shuffle(indexes) 
      
    x_n = tx[indexes[start_id:end_id]]
    y_n = y[indexes[start_id:end_id]]
    if lambda_ != 0:
      sg = gradient_function(y_n, x_n, w, lambda_)
    else:
      sg = gradient_function(y_n, x_n, w)
    w = w - gamma * sg
    
    if verbose:
      if lambda_:
        train_loss = loss_function(y, tx, w, lambda_)
        validation_loss = loss_function(validation_y, validation_tx, w, lambda_)
      else:
        train_loss = loss_function(y, tx, w)
        validation_loss = loss_function(validation_y, validation_tx, w)
      train_errors.append(train_loss)
      validation_errors.append(validation_loss)
    
    n_iter += 1
    start_id = start_id + batch_size
    if start_id >= N:
      start_id = 0
    
    end_id = start_id + batch_size
    # Taking care of potentially smaller last minibatch
    if end_id > N:
      end_id = N
  if lambda_:  
    loss = loss_function(y, tx, w, lambda_)
  else:
    loss = loss_function(y, tx, w)
  if verbose:
    return (w, loss, train_errors, validation_errors)
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
  (w, loss) : (numpy array, float)
      weights (D,1) and loss after max_iters iterations of GD
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
  
  Returns
  -------
  (w, loss) : (numpy array, float)
    weights (D,1) and loss after max_iters iterations of SGD
  """
  return SGD(y, tx, initial_w, max_iters, gamma, "LEAST_SQUARE", 1)
    

def least_squares(y, tx):
  """Least squares regression using normal equations

  Parameters
  ----------
  y : numpy array
    Targets vector (N,) or (N,1)
  tx : numpy array
    Feature matrix (N,D)
  
  Returns
  -------
  (w, loss) : (numpy array, float)
    weights (D,1) and loss after ridge regression
  """
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
  (w, loss) : (numpy array, float)
    weights (D,1) and loss after ridge regression
  """
  y, tx = prepare_dimensions(y, tx)
  N, D = np.shape(tx)
  w, _, _, _ = np.linalg.lstsq(
    tx.T @ tx + 2 * N * lambda_ * np.eye(D),
    tx.T @ y,
    rcond=None
  )
  loss = compute_mse_loss_regularized(y, tx, w, lambda_)
  return (w, loss)


def logistic_regression(y, tx, initial_w, max_iters, gamma):
  """Logistic regression using gradient descent or SGD
  
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
  (w, loss) : (numpy array, float)
    weights (D,1) and loss after max_iters iterations of SGD  
  """
  
  return SGD(y, tx, initial_w, max_iters, gamma, "LOGISTIC_REGRESSION", 1)


def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
  """Regularized logistic regression using gradient descent or SGD
  
  Parameters
  ----------
  y : numpy array
    Targets vector (N,) or (N,1)
  tx : numpy array
    Feature matrix (N,D)
  lambda_ : float
    Regularization constant
  initial_w : numpy array
    weights vector (D,) or (D,1)
  max_iters : int
    number of iteration to run SGD
  gamma : float
    learning rate
  
  Returns
  -------
  (w, loss) : (numpy array, float)
    weights (D,1) and loss after max_iters iterations of SGD
  """
  return SGD(y, tx, initial_w, max_iters, gamma, "REGULARIZED_LOGISTIC_REGRESSION", 1, lambda_)
