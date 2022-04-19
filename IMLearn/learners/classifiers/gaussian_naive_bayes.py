from typing import NoReturn

import numpy as np
import scipy.stats

from IMLearn.metrics.loss_functions import misclassification_error
from ...base import BaseEstimator


class GaussianNaiveBayes(BaseEstimator):
    """
    Gaussian Naive-Bayes classifier
    """

    def __init__(self):
        """
        Instantiate a Gaussian Naive Bayes classifier

        Attributes
        ----------
        self.classes_ : np.ndarray of shape (n_classes,)
            The different labels classes. To be set in `GaussianNaiveBayes.fit`

        self.mu_ : np.ndarray of shape (n_classes,n_features)
            The estimated features means for each class. To be set in `GaussianNaiveBayes.fit`

        self.vars_ : np.ndarray of shape (n_classes, n_features)
            The estimated features variances for each class. To be set in `GaussianNaiveBayes.fit`

        self.pi_: np.ndarray of shape (n_classes)
            The estimated class probabilities. To be set in `GaussianNaiveBayes.fit`
        """
        super().__init__()
        self.classes_, self.mu_, self.vars_, self.pi_ = None, None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits a gaussian naive bayes model

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        self.classes_ = np.unique(y).astype(int)
        samples_by_class = {}
        for i in range(len(y)):
            if y[i] not in samples_by_class:
                samples_by_class[y[i]] = np.array([X[i]])
            else:
                samples_by_class[y[i]] = np.vstack([samples_by_class[y[i]],
                                                    np.array(X[i])])
        m = len(y)
        self.pi_ = np.array([len(samples_by_class[label]) / m for label in
                             self.classes_])
        self.mu_ = None
        self.vars_ = None
        self.gauss_param_calc(samples_by_class)

    def gauss_param_calc(self, samples_by_class):
        for i in self.classes_:
            if self.mu_ is None:
                self.mu_ = np.array(samples_by_class[i].mean(axis=0))
            else:
                self.mu_ = np.vstack(
                    [self.mu_, samples_by_class[i].mean(axis=0)])
            if self.vars_ is None:
                self.vars_ = np.var(samples_by_class[i], axis=0, ddof=1)
            else:
                self.vars_ = np.vstack(
                    [self.vars_, np.var(samples_by_class[i], axis=0, ddof=1)])

    def _predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict responses for given samples using fitted estimator

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples
        """
        likelihoods = self.likelihood(X)
        y_pred = np.array([])
        for sample in likelihoods:
            # max_likelihood = 0
            # pred = None
            # for label in self.classes_:
            #     ind = int(label)
            #     if sample[ind] > max_likelihood:
            #         max_likelihood = sample[ind]
            #         pred = label
            pred = np.argmax(sample)
            y_pred = np.append(y_pred, pred)
        return y_pred

    def likelihood(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate the likelihood of a given data over the estimated model

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input data to calculate its likelihood over the different classes.

        Returns
        -------
        likelihoods : np.ndarray of shape (n_samples, n_classes)
            The likelihood for each sample under each of the classes

        """
        if not self.fitted_:
            raise ValueError(
                "Estimator must first be fitted before calling `likelihood` function")

        likelihood = None
        for sample in X:
            sample_likelihood = np.array([])
            for y in self.classes_:
                ind = int(y)
                likelihood_y = scipy.stats.multivariate_normal.pdf(
                    sample, self.mu_[ind], np.diag(self.vars_[ind])) * \
                               self.pi_[ind]
                sample_likelihood = np.append(sample_likelihood, likelihood_y)
            if likelihood is None:
                likelihood = sample_likelihood
            else:
                likelihood = np.vstack([likelihood, sample_likelihood])
        return likelihood

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
