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
  sg = -1./B * tx.T @ (y - tx @ w)
  return sg


def sigmoid(t):
  """apply the sigmoid function on t.

  Parameters
  ----------
  t: numpy array (B,1)

  Returns
  -------
  sig(t): numpy array (B,1)
  
  Warning: this method can return values of 0. and 1.
  """
  #numerically stable version without useless overflow warnings
  
  pos_mask = (t >= 0)
  neg_mask = (t < 0)
  z = np.zeros_like(t)
  z[pos_mask] = np.exp(-t[pos_mask])
  z[neg_mask] = np.exp(t[neg_mask])
  top = np.ones_like(t)
  top[neg_mask] = z[neg_mask]
  return top / (1 + z)

def compute_regularized_logistic_loss(y, tx, w, lambda_):
  """Regularized Logistic loss. 

  Parameters
  ----------
  y : numpy array
    Targets vector (B,1)
  tx : numpy array
    Feature matrix (B,D)
  w : numpy array
    weights vector (D,1)
  lambda_ : float
    regularization constant
  
  Returns
  -------
  loss : float
    regularized negative log-likelihood   
  """  

  #smallest positive value we can have
  min_value = np.nextafter(0,1)
  
  #avoiding numerical unstability i.e making pred in ]0,1[
  pred = sigmoid(tx@w)
  pred[pred<min_value] = min_value
  
  one_minus_pred = 1-pred
  one_minus_pred[one_minus_pred<min_value] = min_value
  
  reg_term = lambda_*np.sum(w**2)
  loss = -(y.T @ (np.log(pred)) + (1 - y).T @ (np.log(one_minus_pred))) + reg_term
  return loss.item() 

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

  return compute_regularized_logistic_loss(y, tx, w, lambda_=0)
  
def compute_regularized_logistic_gradient(y, tx, w, lambda_):
  """Regularized Logistic gradient for a mini-batch of B points.

  Parameters
  ----------
  y : numpy array
    Targets vector (B,1)
  tx : numpy array
    Feature matrix (B,D)
  w : numpy array
    weights vector (D,1)
  lambda_ : float
    regularization constant  

  Returns
  -------
  gradient : numpy array 
    Gradient of logistic loss (D,1)
  """  
  reg_term = 2*lambda_*np.sum(w)
  return tx.T @ (sigmoid(tx@w)-y)+reg_term

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
  return compute_regularized_logistic_gradient(y, tx, w, lambda_=0)


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

def build_poly(x, degree):
    """polynomial basis functions for input data x, for j=0 up to j=degree."""
    # ***************************************************
    # polynomial basis function
    # this function should return the matrix formed
    # by applying the polynomial basis to the input data
    # ***************************************************
    out = np.ones((x.shape[0],1))
    for d in range(degree):
        out = np.hstack((out,(x**(d+1))))
    return out

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
