from __future__ import annotations

from typing import Tuple, Callable

import numpy as np

from IMLearn import BaseEstimator


def cross_validate(estimator: BaseEstimator, X: np.ndarray, y: np.ndarray,
                   scoring: Callable[[np.ndarray, np.ndarray, ...], float],
                   cv: int = 5) -> Tuple[float, float]:
    """
    Evaluate metric by cross-validation for given estimator

    Parameters
    ----------
    estimator: BaseEstimator
        Initialized estimator to use for fitting the data

    X: ndarray of shape (n_samples, n_features)
       Input data to fit

    y: ndarray of shape (n_samples, )
       Responses of input data to fit to

    scoring: Callable[[np.ndarray, np.ndarray, ...], float]
        Callable to use for evaluating the performance of the cross-validated model.
        When called, the scoring function receives the true- and predicted values for each sample
        and potentially additional arguments. The function returns the score for given input.

    cv: int
        Specify the number of folds.

    Returns
    -------
    train_score: float
        Average train score over folds

    validation_score: float
        Average validation score over folds
    """
    joined_data = np.vstack((X, y)).T
    folds = [i for i in np.array_split(joined_data, cv)]
    train_score, validation_score = [], []
    for i in range(len(folds)):
        train_folds = folds[:]
        del train_folds[i]
        training = np.vstack([train_folds[j] for j in
                              range(len(train_folds))])
        train_x = training[:, :-1]
        train_y = training[:, -1]
        validate_x = folds[i][:, :-1]
        validate_y = folds[i][:, -1]
        estimator.fit(train_x, train_y)
        train_score.append(
            scoring(train_y, estimator.predict(train_x)))
        validation_score.append(scoring(validate_y,
                                        estimator.predict(validate_x)))
    return float(np.mean(train_score)), float(np.mean(validation_score))
