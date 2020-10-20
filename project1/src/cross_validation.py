from helpers import *
from implementations import *

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
  k_indices = [indices[k * interval: (k + 1) * interval] for k in range(K)]
  res = np.array(k_indices)
  return res


def cross_validation_SGD(y, tx, K, initial_w, max_iters, gamma, batch_size, loss_kind, seed, lambda_=None):
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
  batch_size : int
    Size of mini-batches
  loss_kind : string
    can take value in { "LEAST_SQUARE" , "LOGISTIC_REGRESSION", "REGULARIZED_LOGISTIC_REGRESSION" }
  seed : int
    Seed for index shuffling
  lambda_ : float, optional
    Regularization constant, default None
  
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
    # Take all but the k-th row of tx and y
    tx_train, y_train = map(lambda a: a[np.delete(k_indices, k).flatten()], (tx, y))
    # Take the k-th row of tx and y
    tx_test, y_test = map(lambda a: a[k_indices[k]], (tx, y))

    # Train
    w, loss_tr = SGD(y_train, tx_train, initial_w, max_iters, gamma, loss_kind, batch_size, lambda_)
    # Test
    algo_loss = loss_kinds[loss_kind][0]

    if lambda_:
      loss_te = algo_loss(y_test, tx_test, w, lambda_)
    else:
      loss_te = algo_loss(y_test, tx_test, w)

    training_errors.append(loss_tr)
    validation_errors.append(loss_te)

    # Keep the weights that give the lowest loss_te
    if loss_te < min_error:
      min_error = loss_te
      w_best = w

  return w_best, training_errors, validation_errors


def cross_validation_ridge(y, tx, K, seed, lambda_=0):
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
  seed : int
    Seed for index shuffling
  lambda_ : float, optional
    Regularization constant, default None
  
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
    # Take all but the k-th row of tx and y
    tx_train, y_train = map(lambda a: a[np.delete(k_indices, k).flatten()], (tx, y))
    # Take the k-th row of tx and y
    tx_test, y_test = map(lambda a: a[k_indices[k]], (tx, y))

    # Train
    w, loss_tr = ridge_regression(y_train, tx_train, lambda_)
    
    # Test
    loss_te = compute_mse_loss(y_test, tx_test, w)

    training_errors.append(loss_tr)
    validation_errors.append(loss_te)

    # Keep the weights that give the lowest loss_te
    if loss_te < min_error:
      min_error = loss_te
      w_best = w

  return w_best, training_errors, validation_errors
