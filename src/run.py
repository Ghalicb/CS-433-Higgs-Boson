# Here we deal with another data model which is the separation
# of the data set according to the only categorical variable
# (PRI_jet_num, which has 4 categories)


import numpy as np
from implementations import *
from cross_validation import *
from helpers import *


def predict_4sets(list_tX, list_ids, weights):
    """Make predictions using a list of data sets and weights (one
    per category).

    Parameters
    ----------
    list_tX : iterable
      The datasets on which to perform predictions, as
      numpy arrays of shape (N, D) (D may differ depending
      on the dataset)
    list_ids : iterable
      Data ids as provided
    weigths : iterable
      The weights to use for each dataset in list_tX,
      as numpy arrays (D, 1) (D may differ depending
      on the dataset)

    Returns:
    --------
    all_pred: numpy array
      The predicted response, ready to output to csv
    """
    all_pred = []

    for i in range(len(list_tX)):
        curr_pred = list_tX[i] @ weights[i]

        for j, curr_pred in enumerate(curr_pred):
            # Make a decision 
            if curr_pred < 0.5:
                decision = -1
            else:
                decision = 1

            all_pred.append([list_ids[i][j], decision])

    all_pred = np.array(all_pred)
    all_pred = all_pred[all_pred[:,0].argsort()]
    return all_pred
  
def prepare_test(tX_test, y_test, means, stds, ids_test, degrees):
    """Put the data in a form that can be used to make predictions.

    Parameters
    ----------
    tX_test : numpy array
      The test data matrix, as provided in test.csv
    tX_test : numpy array
      The test labels, as provided in test.csv
    means : numpy array
      The column means for the training data, which are
        used to 'normalize' the test data accordingly
    stds : numpy array
      The column standard deviations for the training data,
        which are used to 'normalize' the test data accordingly
    ids_test : iterable
      Test data ids, as provided in test.csv

    Returns:
    --------
    list_tX : iterable
      The different data matrices according to each category,
      expanded into polynomials and normalized.
    list_y : iterable
      The different test labels according to each category.
    list_ids : iterable
      The different test data ids according to each category.
    """
    tX_0_test, tX_1_test, tX_2_test, tX_3_test, y_0_test, y_1_test, y_2_test, y_3_test, ids_0, ids_1, ids_2, ids_3 =\
        separate_data(tX_test, y_test, ids_test)
    
    tX_0_test_expanded = build_poly(tX_0_test, degrees[0])
    tX_0_test_expanded = standardize_test_data(tX_0_test_expanded, means[0], stds[0])

    tX_1_test_expanded = build_poly(tX_1_test, degrees[1])
    tX_1_test_expanded = standardize_test_data(tX_1_test_expanded, means[1], stds[1])

    tX_2_test_expanded = build_poly(tX_2_test, degrees[2])
    tX_2_test_expanded = standardize_test_data(tX_2_test_expanded, means[2], stds[2])

    tX_3_test_expanded = build_poly(tX_3_test, degrees[3])
    tX_3_test_expanded = standardize_test_data(tX_3_test_expanded, means[3], stds[3])
    list_tX = [tX_0_test_expanded, tX_1_test_expanded, tX_2_test_expanded, tX_3_test_expanded]
    list_y = [y_0_test, y_1_test, y_2_test, y_3_test]
    list_ids = [ids_0, ids_1, ids_2, ids_3]
    return list_tX, list_y, list_ids
  

def standardize_train(tX_0, tX_1, tX_2, tX_3, degrees):
    """Perform feature expansion and normalisation on each
    data set (corresponding to a different category in the variable).

    Parameters
    ----------
    The data matrices for each category

    Returns:
    --------
    The expanded data matrices, along with with the
    column means and standard deviations which will be
    used to 'normalize' the test data.
    """
    tx_0_expanded = build_poly(tX_0, degrees[0])
    tx_0_expanded, mean_0, std_0 = standardize(tx_0_expanded)

    tx_1_expanded = build_poly(tX_1, degrees[1])
    tx_1_expanded, mean_1, std_1 = standardize(tx_1_expanded)

    tx_2_expanded = build_poly(tX_2, degrees[2])
    tx_2_expanded, mean_2, std_2 = standardize(tx_2_expanded)

    tx_3_expanded = build_poly(tX_3, degrees[3])
    tx_3_expanded, mean_3, std_3 = standardize(tx_3_expanded)
    means = [mean_0, mean_1, mean_2, mean_3]
    stds = [std_0, std_1, std_2, std_3]
    return tx_0_expanded, tx_1_expanded, tx_2_expanded, tx_3_expanded, means, stds


def separate_data(tx, y, ids):
    """Generate 4 datasets according to the categorical variable.

    Parameters:
    -----------
    The data matrix, response and ids

    Outputs:
    --------
    The 4 data matrices, reponse and id vectors corresponding
    to each category
    """

    tX_0 = tx.copy()
    tX_0 = tX_0[tx[:,-2]==0]
    tX_0 = np.delete(tX_0, [-2, -1], axis=1)
    y_0 = y[tx[:,-2]==0]
    ids_0 = ids[tx[:,-2]==0]
    
    tX_1 = tx.copy()
    tX_1 = tX_1[tx[:,-2]==1]
    tX_1 = np.delete(tX_1, [-2], axis=1)
    y_1 = y[tx[:,-2]==1]
    ids_1 = ids[tx[:,-2]==1]
    
    tX_2 = tx.copy()
    tX_2 = tX_2[tx[:,-2]==2]
    tX_2 = np.delete(tX_2, [-2], axis=1)
    y_2 = y[tx[:,-2]==2]
    ids_2 = ids[tx[:,-2]==2]
    
    tX_3 = tx.copy()
    tX_3 = tX_3[tx[:,-2]==3]
    tX_3 = np.delete(tX_3, [-2], axis=1)
    y_3 = y[tx[:,-2]==3]
    ids_3 = ids[tx[:,-2]==3]
    return tX_0, tX_1, tX_2, tX_3, y_0, y_1, y_2, y_3, ids_0, ids_1, ids_2, ids_3

def get_best_hyperparameters_ridge(y, tX, lambdas=np.logspace(-8, -1, 8),
                                   degrees=np.array([1,2,3,4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]), K=4):
  """
  Calculate the best hyperparameters for ridge regression by using K-cross validation and grid search
  
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
   
  Returns:
  =======
  degree, lambda: int, float
    best hyperparameters
  """
  
  training_errors, validation_errors =\
  lambda_degree_ridge_cv(y, tX, lambdas,degrees, K, 1)

  val_error_deg_lambda = np.mean(validation_errors, axis=0)
  np.nanargmin(val_error_deg_lambda)
  idx_min = np.unravel_index(np.nanargmin(val_error_deg_lambda), val_error_deg_lambda.shape)
  print("Smalles validation loss {}".format(val_error_deg_lambda[idx_min]))
  print("Best degree {}".format(degrees[idx_min[0]]))
  print("Best lambda {}".format(lambdas[idx_min[1]]))
  return degrees[idx_min[0]], lambdas[idx_min[1]]


def get_best_hyperparameters_SGD(y, tX, algo, lambdas=np.logspace(-8, -1, 8), gammas = np.logspace(-8, -1, 8),
                                   degrees=np.array([1,2,3,4]), K=4, max_iters = 10000):
  """
  Calculate the best hyperparameters for a SGD algorithm by using K-cross validation and grid search
  
  Inputs:
  ========
  y : numpy array
    Targets vector (N,1)
  tx : numpy array
    Feature matrix (N,D)
  algo : string
    The algorithm to use for training
    Can take any value in { "LEAST_SQUARE" , "LOGISTIC_REGRESSION", "REGULARIZED_LOGISTIC_REGRESSION"}
  lambdas : iterable
    Regularisation parameters for cost function
  gammas : iterable
    Learning rates for SGD
  degrees : iterable
    The polynomial degrees
  K : int
    Number of folds
  max_iters : int
    Maxium number of iterations for SGD
   
  Returns:
  =======
  degree, lambda, gamma: int, float, float
    best hyperparameters
  """

  training_errors, validation_errors =\
  lambda_gamma_degree_sgd_cv(y, tX, algo, lambdas, gammas, degrees, K, max_iters,1,1)

  val_error_deg_lambda = np.mean(validation_errors, axis=0)
  np.nanargmin(val_error_deg_lambda)
  idx_min = np.unravel_index(np.nanargmin(val_error_deg_lambda), val_error_deg_lambda.shape)
  print("Smalles validation loss {}".format(val_error_deg_lambda[idx_min]))
  print("Best degree {}".format(degrees[idx_min[0]]))
  print("Best lambda {}".format(lambdas[idx_min[1]]))
  print("Best gamma {}".format(gammas[idx_min[2]]))
  return degrees[idx_min[0]], lambdas[idx_min[1]],gammas[idx_min[2]] 

def main(submission_name, seeCrossValidationExample = False):
  """
  Get a submission for AICrowd
  
  Inputs:
  ========
  submission_name : str
  
  seeCrossValidationExample : bool, optional
    Feature matrix (N,D)
  """
  DATA_TRAIN_PATH = '../data/train.csv'
  y, tX, ids = load_csv_data(DATA_TRAIN_PATH, sub_sample=False)
  print("Train set loaded")
  seed = 1
  train_percentage = 0.8

  # Take 20% out for testing
  y_train, tX_train, ids_train, y_test, tX_test, ids_test = partition(y, tX, ids, train_percentage, seed)
  
  # Preparing train set.
  # First change y vector from {-1, 1} to {0,1}
  y_train_0 =y_train.copy()
  y_train_0[y_train_0 == -1] = 0

  # Drop columns with nans
  tX_train_drop_invalid = tX_train.copy()
  tX_train_drop_invalid = tX_train_drop_invalid[:, ~np.any(tX_train_drop_invalid == -999., axis=0)]

  ##############################################################################
  
  # Separate into 4 train datasets according to categorical feature
  tX_0, tX_1, tX_2, tX_3, y_0, y_1, y_2, y_3, ids_0, ids_1, ids_2, ids_3 =\
      separate_data(tX_train_drop_invalid, y_train_0, ids_train)
  
  if seeCrossValidationExample:
    best_degree_0, best_lambda_0 = get_best_hyperparameters_ridge(y_0, tX_0)
  
  best_degrees = [14,15,12,15]
  # Expand and standardize the 4 train sets
  tx_0_expanded, tx_1_expanded, tx_2_expanded, tx_3_expanded, means, stds =\
  standardize_train(tX_0, tX_1, tX_2, tX_3, best_degrees)
  
  (w_0, loss_0) = ridge_regression(y_0, tx_0_expanded, 1e-8)
  (w_1, loss_1) = ridge_regression(y_1, tx_1_expanded, 1e-8)
  (w_2, loss_2) = ridge_regression(y_2, tx_2_expanded, 1e-8)
  (w_3, loss_3) = ridge_regression(y_3, tx_3_expanded, 1e-8)
  best_weights = [w_0, w_1, w_2, w_3]
    
  DATA_TEST_PATH = '../data/test.csv'
  y_test_aiCrowd, tX_test_aiCrowd, ids_test_aiCrowd = load_csv_data(DATA_TEST_PATH, sub_sample=False)
  print("Test set loaded")
  tX_test_aiCrowd_drop = tX_test_aiCrowd.copy()
  tX_test_aiCrowd_drop = tX_test_aiCrowd_drop[:, ~np.any(tX_train == -999., axis=0)]
  
  #Separate into 4 test datasets and standardize them with the means and variances of the train sets
  list_tX_test_ai, list_y_test_ai, list_ids_test_ai =\
    prepare_test(tX_test_aiCrowd_drop, y_test_aiCrowd, means, stds, ids_test_aiCrowd, best_degrees)
  
  ai_predictions = predict_4sets(list_tX_test_ai, list_ids_test_ai, best_weights)
  create_csv_submission(ai_predictions[:,0], ai_predictions[:,1], submission_name)


if __name__ == '__main__':
  main("best_submission")
