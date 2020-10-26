# -*- coding: utf-8 -*-
"""Some helper functions."""
import csv
import numpy as np

def partition(y, tx, ids, fraction = 0.8, seed=1):
  """Partition data by a given fraction.
  
  Parameters
  ----------
  y : numpy array
    (N,1)
  tx : numpy array
    (N,D)
  ids : numpy array
    (N,1)
  fraction : float, optional
    Fraction to use in test set. The default is 0.8
  seed : nt, optional
    Seed for permutations

  Returns
  -------
  y_train, tx_train, ids_train, y_test, tx_test, ids_test : numpy arrays
    
  """
  np.random.seed(seed)
  indices = np.random.permutation(len(y))
  cutoff = int(fraction * len(y))

  y_train = y[indices[:cutoff_idx]]
  tX_train = tX[indices[:cutoff_idx]]
  ids_train = ids[indices[:cutoff_idx]]
  y_test = y[indices[cutoff_idx:]]
  tX_test = tX[indices[cutoff_idx:]]
  ids_test = ids[indices[cutoff_idx:]]

  return y_train, tx_train, ids_train, y_test, tx_test, ids_test

def standardize(x, notFirst=True):
  """Normalize x, column-by-column

  Parameters
  ----------
  tx : numpy array
    (N,D)
  notFirst : bool, optional
    whether to leave the first column (usually 1's). The default is True
    
  Returns
  -------
  x : numpy array
    two dimensional numpy arrays with correct dimensions
    (N,D)
  mean_x : numpy array
    mean of each column of x (1,D)
  std_x : numpy array
    standard_deviation of x (1, D)
  """
  if notFirst:
    mean_x = np.mean(x[:,1:], axis=0)
    x[:,1:] = x[:,1:] - mean_x
    std_x = np.std(x[:,1:], axis=0)
    x[:,1:] = x[:,1:] / std_x
  else:
    mean_x = np.mean(x, axis=0)
    x = x - mean_x
    std_x = np.std(x, axis=0)
    x = x / std_x
  return x, mean_x, std_x

def standardize_test_data(x, mean, std, notFirst=True):
  """Normalize x, column-by-column

  Parameters
  ----------
  tx : numpy array
    (N,D)
  mean : numpy array
    mean to use for each column (1,D)
  std : numpy array
    standard_deviation to use for each column (1, D)
  notFirst : bool, optional
    whether to leave the first column (usually 1's). The default is True
    
  Returns
  -------
  x : numpy array
    two dimensional numpy arrays with correct dimensions
    (N,D)
  """
  if notFirst:
    x[:,1:] = x[:,1:] - mean
    x[:,1:] = x[:,1:] / std
  else:
    x = x - mean
    x = x / std
  return x

def get_accuracy(y, tx, w, threshold):
  """Calculates accuracy on given set.

  Parameters
  ----------
  y : numpy array
    targets (N,1)
  tx : numpy array
    features (N,D)
  w : numpy array
    weights to use (1,D)
  threshold : float
    threshold to use between classes
    
  Returns
  -------
  accuracy : float
    accuracy on the set
  """    
  pred = tx@w
  pred_class = [0 if p<threshold else 1 for p in pred]
  n = len(y)
  correct=0
  for i in range(n):
      if pred_class[i] == y[i]:
          correct+=1
  return correct/n
  
def build_poly(x, degree, interactions=False):
  """Polynomial basis functions for input data x, for 0 up to degree degrees and 
  all 2nd degree interactions of first degree features.

  Parameters
  ----------
  x : numpy array
    Feature matrix (N,D)
  degree : int
    max degree of polynomial expansion
  interactions : bool, optional
    whether to put interactions in the expanded polynomial. The default is False
    
  Returns
  -------
  accu_expanded : numpy array
      matrix (N, 1+D*degree+(D choose 2)) formed by applying the polynomial basis to the input data
  """
  n, d = x.shape
  accu_expanded = np.ones((n,1))
  for deg in range(degree):
    accu_expanded = np.c_[accu_expanded, x**(deg+1)]
 
  
  if interactions:
    feature_pairs = []
    for i in range(d):
      for j in range(i+1,d):
        feature_pairs.append([i,j])
    pair_columns = np.zeros((n, len(feature_pairs)))

    for i, pair in enumerate(feature_pairs):
      pair_columns[:, i] = x[:, pair[0]] * x[:, pair[1]]

    accu_expanded =  np.c_[accu_expanded, pair_columns]

  return accu_expanded


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
