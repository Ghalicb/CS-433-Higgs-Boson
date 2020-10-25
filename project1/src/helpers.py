# -*- coding: utf-8 -*-
"""Some helper functions."""
import csv
import numpy as np


def standardize(x):
  """Normalize x, column-by-column

  Parameters
  ----------
  tx : numpy array
    (N,D)
  
  Returns
  -------
  x : numpy array
    two dimensional numpy arrays with correct dimensions
    (N,D)
  mean_x : mean of each column of x
  std_x : standard_deviation of x
  """
  mean_x = np.mean(x, axis=0)
  x = x - mean_x
  std_x = np.std(x, axis=0)
  x = x / std_x
  return x, mean_x, std_x


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


def build_poly(x, degree):
  """polynomial basis functions for input data x, for 0 up to degree degrees and 
  all 2nd degree interactions of first degree features.

  Parameters
  ----------
  x : numpy array
    Feature matrix (N,D)
  degree : int
    max degree of polynomial expansion
    
  Returns
  -------
  accu_expanded : numpy array
      matrix (N, 1+D*degree+(D choose 2)) formed by applying the polynomial basis to the input data
  
  Example
  -------
  input = x,y,z degree=3
  output = 1,x,y,z,x^2,y^2,z^2,x^3,y^3,z^3,xy,xz,yz 
  """
  n, d = x.shape
  accu_expanded = np.ones((n,1))

  for deg in range(degree):
    accu_expanded = np.c_[accu_expanded, x**(deg+1)]
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
