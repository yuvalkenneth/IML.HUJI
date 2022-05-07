from __future__ import annotations

from typing import Tuple, NoReturn

import numpy as np

from IMLearn.metrics.loss_functions import misclassification_error
from ...base import BaseEstimator


class DecisionStump(BaseEstimator):
    """
    A decision stump classifier for {-1,1} labels according to the CART algorithm

    Attributes
    ----------
    self.threshold_ : float
        The threshold by which the data is split

    self.j_ : int
        The index of the feature by which to split the data

    self.sign_: int
        The label to predict for samples where the value of the j'th feature is about the threshold
    """

    def __init__(self) -> DecisionStump:
        """
        Instantiate a Decision stump classifier
        """
        super().__init__()
        self.threshold_, self.j_, self.sign_ = None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits a decision stump to the given data

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        current_best = ()
        for j in range(np.shape(X)[1]):
            for i in [-1, 1]:
                if not current_best:
                    self.j_ = j
                    current_best = self._find_threshold(X[:, j], y, i)
                    self.sign_ = i
                    self.threshold_ = current_best[0]
                else:
                    best = self._find_threshold(X[:, j], y, i)
                    if best[1] < current_best[1]:
                        current_best = best
                        self.sign_ = i
                        self.j_ = j
                        self.threshold_ = best[0]

    def _predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict responses for given samples using fitted estimator

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples

        Notes
        -----
        Feature values strictly below threshold are predicted as `-sign` whereas values which equal
        to or above the threshold are predicted as `sign`
        """
        y = np.array([self.sign_ if X[k][self.j_] >= self.threshold_ else
                      -self.sign_ for k in range(np.shape(X)[0])])
        return y

    def _find_threshold(self, values: np.ndarray, labels: np.ndarray,
                        sign: int) -> Tuple[float | Any, ndarray]:
        """
        Given a feature vector and labels, find a threshold by which to perform a split
        The threshold is found according to the value minimizing the misclassification
        error along this feature

        Parameters
        ----------
        values: ndarray of shape (n_samples,)
            A feature vector to find a splitting threshold for

        labels: ndarray of shape (n_samples,)
            The labels to compare against

        sign: int
            Predicted label assigned to values equal to or above threshold

        Returns
        -------
        thr: float
            Threshold by which to perform split

        thr_err: float between 0 and 1
            Misclassificaiton error of returned threshold

        Notes
        -----
        For every tested threshold, values strictly below threshold are predicted as `-sign` whereas values
        which equal to or above the threshold are predicted as `sign`
        """
        i = np.argsort(values)
        values = values[i]
        labels = labels[i]
        pred_labels = np.array([sign] * len(values))
        err = np.sum(abs(labels[np.sign(labels) != pred_labels]))
        best_err = (values[0] - 1, err)
        for j in range(len(values) - 1): # o(n)
            thresh = ((values[j] + values[j + 1]) / 2)
            # pred_labels = np.array([-sign] * (j + 1) + [sign] * (len(values)
            #                                                      - j - 1))
            pred_labels[j] = -sign
            if pred_labels[j] == np.sign(labels[j]): #if the current index
                # is labeled properly after thresh change, it is no longer
                # part of the error
                err -= abs(labels[j])
            else:
                err += abs(labels[j])
            # err = np.sum(abs(labels[np.sign(labels) != pred_labels])) # o(1)
            if err < best_err[1]:
                best_err = (thresh, err)
        pred_labels = np.array([sign] * len(values))
        err = np.sum(abs(labels[np.sign(labels) != pred_labels]))
        if err < best_err[1]:
            best_err = (values[len(values) - 1] + 1, err)
        return best_err

    def _loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluate performance under misclassification loss function

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples

        y : ndarray of shape (n_samples, )
            True labels of test samples

        Returns
        -------
        loss : float
            Performance under missclassification loss function
        """
        y_pred = self._predict(X)
        return misclassification_error(y, y_pred)


