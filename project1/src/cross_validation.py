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


def cross_validation_SGD(y, tx, K, initial_w, max_iters, gamma, batch_size, loss_kind, seed, lambda_ = 0):
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
    Regularization constant, default 0
  
  Returns
  -------
  w_best : numpy array
    Weight vector (D,1) with smallest validation error
  final_training_errors : list
    Training error after training for each fold (K elements)
  final_validation_errors : list
    Validation error after training for each fold (K elements)
  training_errors_during_training : list of list
    Training errors during training for each fold (K lists of max_iters elements)
  validation_errors_during_training : list of list
    Validation errors during training for each fold (K lists of max_iters elements)
  """
  k_indices = build_k_indices(y, K, seed)
  final_training_errors = []
  final_validation_errors = []
  training_errors_during_training = []
  validation_errors_during_training = []
  min_error = np.inf

  for k in range(K):
    # Take all but the k-th row of tx and y
    tx_train, y_train = map(lambda a: a[np.delete(k_indices, k).flatten()], (tx, y))
    # Take the k-th row of tx and y
    tx_test, y_test = map(lambda a: a[k_indices[k]], (tx, y))

    # Train
    (w, loss, train_errors, validation_errors) = SGD(y_train, tx_train, initial_w, max_iters,
                                                   gamma, loss_kind, batch_size, lambda_,
                                                   verbose=True, validation_y = y_test, validation_tx=tx_test)
    training_errors_during_training.append(train_errors)
    validation_errors_during_training.append(validation_errors)
    # Test
    algo_loss = loss_kinds[loss_kind][0]

    if lambda_ != 0:
      loss_te = algo_loss(y_test, tx_test, w, lambda_)
    else:
      loss_te = algo_loss(y_test, tx_test, w)

    final_training_errors.append(loss_tr)
    final_validation_errors.append(loss_te)

    # Keep the weights that give the lowest loss_te
    if loss_te < min_error:
      min_error = loss_te
      w_best = w

  return w_best, final_training_errors, final_validation_errors, training_errors_during_training, validation_errors_during_training


def cross_validation_ridge(y, tx, K, seed, lambda_ = 0):
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
    Regularization constant, default 0
  
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

def lambda_degree_ridge_cv(y, tx, lambdas, degrees, K):
  """Do K-fold cross-validation for each value in lambdas and gammas and each degree of polynomial expansion, at every iteration.
  
  Inputs:
  ========
  y : numpy array
    Targets vector (N,1)
  tx : numpy array
    Feature matrix (N,D)
  lambdas : iterable
    Regularisation parameters for cost function
  degrees : iterable
    The polynomial degrees
  K : int
    Number of folds
  
  Outputs:
  training_errors : np array
    (K, len(degrees), len(lambdas))
    Training loss for each fold, for each degree and for each lambda
  validation_errors : np array
    (K, len(degrees), len(lambdas))
    Validation loss for each fold, for each degree and for each lambda
  """
  y, tx = prepare_dimensions(y, tx)
  

  N = len(y)
  len_degrees = len(degrees)
  len_lambdas = len(lambdas)

  training_errors = np.zeros((K, len_degrees, len_lambdas))
  validation_errors = np.zeros((K, len_degrees, len_lambdas))

  k_indices = build_k_indices(y, K, seed)

  for d, degree in enumerate(degrees):
    print("Degree = {}".format(degree))
    tx_poly = build_poly(tx, degree)
    tx_poly, *_ = standardize(tx_poly)
    initial_w = np.ones((tx_poly.shape[1], 1))
    print(tx_poly.shape)
    for k in range(K):
      print("Fold = {}".format(k+1))
      # Take all but the k-th row of tx and y
      tx_train, y_train = map(lambda a: a[np.delete(k_indices, k).flatten()], (tx_poly, y))
      # Take the k-th row of tx and y
      tx_test, y_test = map(lambda a: a[k_indices[k]], (tx_poly, y))

      for i, lambda_ in enumerate(lambdas):
        # Train
        w, loss_tr = ridge_regression(y_train, tx_train, lambda_)
        # Test
        loss_te = compute_mse_loss(y_test, tx_test, w)

        training_errors[k, d, i] = loss_tr
        validation_errors[k, d, i] = loss_te

  return training_errors, validation_errors

def lambda_gamma_degree_sgd_cv(y, tx, algorithm, lambdas, gammas, degrees, K, max_iters, batch_size):
  """Do K-fold cross-validation for each value in lambdas and gammas and each degree of polynomial expansion, at every iteration.
  
  Inputs:
  ========
  y : numpy array
    Targets vector (N,1)
  tx : numpy array
    Feature matrix (N,D)
  algorithm : string
    The algorithm to use for training
    Can take any value in { "LEAST_SQUARE" , "LOGISTIC_REGRESSION", "REGULARIZED_LOGISTIC_REGRESSION"}
  lambdas : iterable
    Regularisation parameters for cost function
  gammas : iterable
    Learning rates for SGD
  degrees : int
    The polynomial degree
  K : int
    Number of folds
  max_iters : int
    Maxium number of iterations for SGD
  batch_size : int
    Size of mini-batches
  
  Outputs:
  ========
  training_errors : np array
    (K, len(degrees), len(lambdas), len(gammas))
    Training loss for each fold, for each degree, for each lambda and gamma
  validation_errors : np array
    (K, len(degrees), len(lambdas), len(gammas))
    Validation loss for each fold, for each degree, for each lambda and gamma
  """
  y, tx = prepare_dimensions(y, tx)
  

  N = len(y)
  len_degrees = len(degrees)
  len_lambdas = len(lambdas)
  len_gammas = len(gammas)

  training_errors = np.zeros((K, len_degrees, len_lambdas, len_gammas))
  validation_errors = np.zeros((K, len_degrees, len_lambdas, len_gammas))

  k_indices = build_k_indices(y, K, seed)

  for d, degree in enumerate(degrees):
    print("Degree = {}".format(degree))
    tx_poly = build_poly(tx, degree)
    tx_poly, *_ = standardize(tx_poly)
    initial_w = np.ones((tx_poly.shape[1], 1))
    print(tx_poly.shape)
    for k in range(K):
      print("Fold = {}".format(k+1))
      # Take all but the k-th row of tx and y
      tx_train, y_train = map(lambda a: a[np.delete(k_indices, k).flatten()], (tx_poly, y))
      # Take the k-th row of tx and y
      tx_test, y_test = map(lambda a: a[k_indices[k]], (tx_poly, y))

      for i, lambda_ in enumerate(lambdas):
        for j, gamma in enumerate(gammas):
          # Train
          w, loss_tr = SGD(y_train, tx_train, initial_w, max_iters, gamma, algorithm, batch_size, lambda_)
          # Test
          loss_te = compute_mse_loss(y_test, tx_test, w)

          training_errors[k, d, i, j] = loss_tr
          validation_errors[k, d, i, j] = loss_te

  return training_errors, validation_errors
