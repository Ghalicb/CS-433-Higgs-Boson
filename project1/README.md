# CS-433 Machine Learning: Project 1 (Higgs boson data)
=========================

## What is this?
In this project, we build a classification model using the data
provided by EPFL which can be found at [this link](https://www.aicrowd.com/challenges/epfl-machine-learning-higgs).

<<<<<<< HEAD
The test.csv and train.csv files should be located in ./data

=======
To see the best results we obtained, run the `run.py` script.
>>>>>>> 8a5024fe3ce63f5b2e14e25c9aeedafdc1e121d7

## What is in there?
The `run.py` script will:
1. Load the training data and clean it (including normalising it);
2. Train the best model we found on the clean training data;
3. Use the resulting model for prediction on the test data
and output the predictions in a csv file.

The algorithms implemented in this project are in `implementations.py`.
All of them are either based on closed-form solutions or based on
the stochastic gradient descent (or SGD) method. Many of the functions
are actually defined as special cases of the `SGD` function.

All functions relating to cross-validation (including those used for
grid-searching hyperparameters) can be found in `cross_validation.py`.

`helpers.py` contains various helper functions, such as the `build_poly`
function which performs feature expansion into polynomials, or `standardize`
which normalises the data column-wise.

## Why is it like this?
- Data preparation
An important issue with this dataset is the number of non-retrieved
values, with are referred to as -999 in the dataset. About a third
of the features suffer from this, and for many of those up to 70%
of the points is unretrieved.
We explore 2 major strategies to address this, which are explained
in detail in the accompanying report.
- Feature generation
Generating a more complex model (e.g. with polynomial expansion)
can help performance. We explain our course of action in detail
in the report.
- Cross-validation
Cross-validation allows for a more precise characterisation of the
error during training, while using the available data to its full
potential.
At each stage, the dataset is separated into 2 parts (the training set and the 
validation set) using the `build_k_indices` function written
during Lab 4 of the course. The size of each set is determined
by the number of folds.
We then proceed to train a given model on the training set
and then evaluate it on the validation set.
We iterate over the number of folds, so that after the whole
process, every data point has been used for validation just
once.

