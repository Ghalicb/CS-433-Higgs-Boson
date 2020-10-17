# -*- coding: utf-8 -*-
"""Some helper functions."""
import csv
import numpy as np


def compute_mse_loss(y, tx, w):
  """Mean Square Error loss

  Parameters
  ----------
  y : numpy array
    Targets vector (N,1)
  tx : numpy array
    Feature matrix (N,D)
  w : numpy array
    weights vector (D,1)
  
  Returns
  -------
  loss : float
  """  
  e = y - tx @ w
  return (1./(2*len(y)) * e.T @ e).item() 


def compute_mse_gradient(y, tx, w):
  """Mean Square Error gradient for a mini-batch of B points

  Parameters
  ----------
  y : numpy array
    Targets vector (B,1)
  tx : numpy array
    Feature matrix (B,D)
  w : numpy array
    weights vector (D,1)
  
  Returns
  -------
  sg : numpy array 
    gradient of mse loss (D,1)
  """ 
  B = len(y)
  e = y - tx @ w
  sg = -1/B * tx.T @ (y - tx @ w)
  return sg


def sigmoid(t):
    """apply the sigmoid function on t.
    
    Parameters
    ----------
    t: numpy array (B,1)
    
    Returns
    -------
    sig(t): numpy array (B,1)
    
    """
    return np.where(t <- 700, np.exp(t)/(1+np.exp(t)), 1/(1+np.exp(-t)))


def compute_logistic_loss(y, tx, w):
  """Logistic loss. 

  Parameters
  ----------
  y : numpy array
    Targets vector (B,1)
  tx : numpy array
    Feature matrix (B,D)
  w : numpy array
    weights vector (D,1)
  
  Returns
  -------
  loss : float
    negative log-likelihood
  """  
  prediction = tx @ w
  log_sum = np.sum(np.log(1 + np.exp(prediction)))
  return -y.T @ prediction + log_sum


def compute_logistic_gradient(y, tx, w):
  """Logistic gradient for a mini-batch of B points.

  Parameters
  ----------
  y : numpy array
    Targets vector (B,1)
  tx : numpy array
    Feature matrix (B,D)
  w : numpy array
    weights vector (D,1)

  Returns
  -------
  gradient : numpy array 
    Gradient of logistic loss (D,1)
  """  
  return tx.T @ (sigmoid(tx@w)-y)


def prepare_dimensions(y, tx):
  """Reshape input data to two dimensions, if the dimensions are already correct, they stay the same

  Parameters
  ----------
  y : numpy array
    Targets vector (N,) or (N,1)
  tx : numpy array
    Feature matrix (N,D)
  
  Returns
  -------
  y_reshaped, tx_reshaped : (numpy array (N,1), numpy array (N,D))
      two dimensional numpy arrays with correct dimensions
  """
  y_reshaped = y.reshape(-1, 1)
  tx_reshaped = tx.reshape(len(y_reshaped), -1)
  return y_reshaped, tx_reshaped


def build_k_indices(y, K, seed):
  """Build k indices for k-fold cross-validation.

  Parameters
  ----------
  y : numpy array
    Targets vector (N,1)
  K : int
    Number of folds
  seed : int
    Seed for index shuffling
  Returns
  -------
  res : numpy array
    2-dimensional array with shuffled indices arranged in K rows
  """
  num_row = y.shape[0]
  interval = int(num_row / K)
  np.random.seed(seed)
  # Shuffle (1, ..., num_row)
  indices = np.random.permutation(num_row)
  # Arrange indices into K lists
  k_indices = [indices[
    k * interval: (k + 1) * interval
  ] for k in range(K)]
  res = np.array(k_indices)
  return res


def cross_validation_SGD(y, tx, K, initial_w, max_iters, gamma, B, loss_kind, seed):
  """K-fold cross validation for stochastic gradient descent.

  Parameters
  ----------
  y : numpy array
    Targets vector (N,1)
  tx : numpy array
    Feature matrix (N,D)
  K : int
    Number of folds
  initial_w : numpy array
    weights vector (D,) or (D,1)
  max_iters : int
    number of iteration to run SGD
  gamma : float
    Learning rate for SGD
  B : int
    Size of mini-batches
  loss_kind : string
    can take value in { "LEAST_SQUARE" , "LOGISTIC_REGRESSION" }
  seed : int
    Seed for index shuffling
  Returns
  -------
  w_best : numpy array
    Weight vector (D,1) with smallest validation error
  training_errors : list
    Training error for each fold (K elements)
  validation_errors : list
    Validation error for each fold (K elements)
  """
  k_indices = build_k_indices(y, K, seed)

  training_errors = []
  validation_errors = []
  min_error = np.inf

  for k in range(K):
    # Take the k-th row of tx and y
    f = lambda a: a[k_indices[k]][:]
    tx_train, y_train = map(f, (tx, y))

    # Take all but the k-th row of tx and y
    f = lambda a: a[ np.concatenate( [
      k_indices[i] for i in range(len(k_indices)) if i != k
    ] )][:]
    tx_test, y_test = map(f, (tx, y))

    # Train
    w, loss_tr = SGD(
      y_train,
      tx_train,
      initial_w,
      max_iters,
      gamma,
      loss_kind,
      batch_size
    )
    # Test
    algo_loss = loss_kinds[loss_kind][0]
    loss_te = algo_loss(y_test, tx_test, w)

    training_errors.append(loss_tr)
    validation_errors.append(loss_te)

    # Keep the weights that give the lowest loss_te
    if loss_te < min_error:
      min_error = loss_te
      w_best = w

  return w_best, training_errors, validation_errors


def load_csv_data(data_path, sub_sample=False):
  """Loads data and returns y (class labels), tX (features) and ids (event ids)"""
  y = np.genfromtxt(data_path, delimiter=",", skip_header=1, dtype=str, usecols=1)
  x = np.genfromtxt(data_path, delimiter=",", skip_header=1)
  ids = x[:, 0].astype(np.int)
  input_data = x[:, 2:]

  # convert class labels from strings to binary (-1,1)
  yb = np.ones(len(y))
  yb[np.where(y=='b')] = -1

  # sub-sample
  if sub_sample:
    yb = yb[::50]
    input_data = input_data[::50]
    ids = ids[::50]

  return yb, input_data, ids


def predict_labels(weights, data):
  """Generates class predictions given weights, and a test data matrix"""
  y_pred = np.dot(data, weights)
  y_pred[np.where(y_pred <= 0)] = -1
  y_pred[np.where(y_pred > 0)] = 1

  return y_pred


def create_csv_submission(ids, y_pred, name):
  """
  Creates an output file in csv format for submission to kaggle

  Arguments:
    ids (event ids associated with each prediction)
    y_pred (predicted class labels)
    name (string name of .csv output file to be created)
  """
  with open(name, 'w') as csvfile:
    fieldnames = ['Id', 'Prediction']
    writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=fieldnames)
    writer.writeheader()
    for r1, r2 in zip(ids, y_pred):
      writer.writerow({'Id':int(r1),'Prediction':int(r2)})
