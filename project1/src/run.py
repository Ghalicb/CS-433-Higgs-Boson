import numpy as np
from helpers import *

def predict_4sets(list_tX, list_y, list_ids, weights):
    all_pred = []
    for i in range(len(list_tX)):
        curr_pred = list_tX[i]@weights[i]
        for j, curr_pred in enumerate(curr_pred):
            if curr_pred<0.5:
                all_pred.append([list_ids[i][j], -1])
            else:
                all_pred.append([list_ids[i][j], 1])
    all_pred = np.array(all_pred)
    all_pred = all_pred[all_pred[:,0].argsort()]
    return all_pred
  
def prepare_test(tX_test, y_test, means, stds, ids_test):
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

def remove_columns(tx, cols):
    return np.delete(tx, cols , axis=1)
  
def separate_data(tx, y, ids):
    tX_0 = tx.copy()
    tX_0 = tX_0[tx[:,-2]==0]
    tX_0 = remove_columns(tX_0, [-2, -1])
    y_0 = y[tx[:,-2]==0]
    ids_0 = ids[tx[:,-2]==0]
    
    tX_1 = tx.copy()
    tX_1 = tX_1[tx[:,-2]==1]
    tX_1 = remove_columns(tX_1, [-2])
    y_1 = y[tx[:,-2]==1]
    ids_1 = ids[tx[:,-2]==1]
    
    tX_2 = tx.copy()
    tX_2 = tX_2[tx[:,-2]==2]
    tX_2 = remove_columns(tX_2, [-2])
    y_2 = y[tx[:,-2]==2]
    ids_2 = ids[tx[:,-2]==2]
    
    tX_3 = tx.copy()
    tX_3 = tX_3[tx[:,-2]==3]
    tX_3 = remove_columns(tX_3, [-2])
    y_3 = y[tx[:,-2]==3]
    ids_3 = ids[tx[:,-2]==2]
    return tX_0, tX_1, tX_2, tX_3, y_0, y_1, y_2, y_3, ids_0, ids_1, ids_2, ids_3