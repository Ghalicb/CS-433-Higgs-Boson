# Here we deal with another data model which is the separation
# of the data set according to the only categorical variable
# (PRI_jet_num, which has 4 categories)


import numpy as np
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
  
def prepare_test(tX_test, y_test, means, stds, ids_test):
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
    
    tX_0_test_expanded = build_poly(tX_0_test, 14)
    tX_0_test_expanded = standardize_test_data(tX_0_test_expanded, means[0], stds[0])

    tX_1_test_expanded = build_poly(tX_1_test, 15)
    tX_1_test_expanded = standardize_test_data(tX_1_test_expanded, means[1], stds[1])

    tX_2_test_expanded = build_poly(tX_2_test, 12)
    tX_2_test_expanded = standardize_test_data(tX_2_test_expanded, means[2], stds[2])

    tX_3_test_expanded = build_poly(tX_3_test, 15)
    tX_3_test_expanded = standardize_test_data(tX_3_test_expanded, means[3], stds[3])
    list_tX = [tX_0_test_expanded, tX_1_test_expanded, tX_2_test_expanded, tX_3_test_expanded]
    list_y = [y_0_test, y_1_test, y_2_test, y_3_test]
    list_ids = [ids_0, ids_1, ids_2, ids_3]
    return list_tX, list_y, list_ids
  

def standardize_train(tX_0, tX_1, tX_2, tX_3):
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
    tx_0_expanded = build_poly(tX_0, 14)
    tx_0_expanded, mean_0, std_0 = standardize(tx_0_expanded)

    tx_1_expanded = build_poly(tX_1, 15)
    tx_1_expanded, mean_1, std_1 = standardize(tx_1_expanded)

    tx_2_expanded = build_poly(tX_2, 12)
    tx_2_expanded, mean_2, std_2 = standardize(tx_2_expanded)

    tx_3_expanded = build_poly(tX_3, 15)
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
    # tX = np.zeros((len(y), tx.shape[1], 4))
    # y_out = np.zeros((len(y), 1, 4))
    # ids_out = np.zeros((len(y), 1, 4))

    # for i in range(4):
    #   tX_i = tx.copy()
    #   tX_i = tX_i[tx[:,-2] == i]
    #   if i == 0:
    #     tX_i = np.delete(tX_i, [-2, -1], axis=1)
    #   else:
    #     tX_i = np.delete(tX_i, [-2], axis=1)
    #   tX[:,:,1] = tx_i
    #   y_out[:,:,i] = y[tx[:,-2] == i]
    #   ids_out[:,:,i] = ids[tx[:,-2] == i]

    # return tX, y_out, ids_out
