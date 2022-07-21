from typing import Tuple

import numpy as np
import pandas as pd


def split_train_test(X: pd.DataFrame, y: pd.Series,
                     train_proportion: float = .25) \
        -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Split given sample to a training- and testing sample

    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_features)
        Data frame of samples and feature values.

    y : Series of shape (n_samples, )
        Responses corresponding samples in data frame.

    train_proportion: Fraction of samples to be split as training set

    Returns
    -------
    train_X : DataFrame of shape (ceil(train_proportion * n_samples), n_features)
        Design matrix of train set

    train_y : Series of shape (ceil(train_proportion * n_samples), )
        Responses of training samples

    test_X : DataFrame of shape (floor((1-train_proportion) * n_samples), n_features)
        Design matrix of test set

    test_y : Series of shape (floor((1-train_proportion) * n_samples), )
        Responses of test samples

    """
    # X = X.join(y)
    # train = X.sample(frac=train_proportion)
    # test = X.drop(train.index)
    # train_y = train.iloc[:, -1]
    # train_x = train.iloc[:, :-1]
    # test_y = test.iloc[:, -1]
    # test_x = test.iloc[:, :-1]
    # X = X.iloc[:, :-1]
    # return train_x, train_y, test_x, test_y
    X.insert(0, "responses", y, True)
    train_sample = X.sample(frac=train_proportion)
    test_sample = X.drop(train_sample.index)
    train_y = train_sample['responses']
    train_X = train_sample.drop(columns='responses')
    test_y = test_sample['responses']
    test_X = test_sample.drop(columns='responses')
    X.drop(columns='responses', inplace=True)
    return train_X, train_y, test_X, test_y


def confusion_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Compute a confusion matrix between two sets of integer vectors

    Parameters
    ----------
    a: ndarray of shape (n_samples,)
        First vector of integers

    b: ndarray of shape (n_samples,)
        Second vector of integers

    Returns
    -------
    confusion_matrix: ndarray of shape (a_unique_values, b_unique_values)
        A confusion matrix where the value of the i,j index shows the number of times value `i` was found in vector `a`
        while value `j` vas found in vector `b`
    """
    values = np.unique(a)
    mat = np.zeros(len(values), len(values))
    for i in range(len(values)):
        for j in range(len(values)):
            mat[i][j] = np.sum((a == values[i]) & (b == values[j]))
    return mat
