from typing import NoReturn
from IMLearn.metrics.loss_functions import misclassification_error
import numpy as np

from ...base import BaseEstimator
import scipy.stats

class LDA(BaseEstimator):
    """
    Linear Discriminant Analysis (LDA) classifier

    Attributes
    ----------
    self.classes_ : np.ndarray of shape (n_classes,)
        The different labels classes. To be set in `LDA.fit`

    self.mu_ : np.ndarray of shape (n_classes,n_features)
        The estimated features means for each class. To be set in `LDA.fit`

    self.cov_ : np.ndarray of shape (n_features,n_features)
        The estimated features covariance. To be set in `LDA.fit`

    self._cov_inv : np.ndarray of shape (n_features,n_features)
        The inverse of the estimated features covariance. To be set in `LDA.fit`

    self.pi_: np.ndarray of shape (n_classes)
        The estimated class probabilities. To be set in `GaussianNaiveBayes.fit`
    """

    def __init__(self):
        """
        Instantiate an LDA classifier
        """
        super().__init__()
        self.classes_, self.mu_, self.cov_, self._cov_inv, self.pi_ = None, None, None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits an LDA model.
        Estimates gaussian for each label class - Different mean vector, same covariance
        matrix with dependent features.

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
        self.pi_ = np.array([])
        self.mu_ = np.array([])
        m = len(y)
        cov_ = None
        for i in self.classes_:
            self.pi_ = np.append(self.pi_, len(samples_by_class[i]) / m)
            if len(self.mu_) == 0:
                self.mu_ = np.append(self.mu_, samples_by_class[i].mean(
                    axis=0))
            else:
                self.mu_ = np.vstack([self.mu_, samples_by_class[i].mean(
                    axis=0)])
        for j in range(len(y)):
            if cov_ is None:
                cov_ = np.outer(X[j] - self.mu_[int(y[j])], X[j] - self.mu_[
                    int(y[j])])
            else:
                cov_ += np.outer(X[j] - self.mu_[int(y[j])], X[j] - self.mu_[
                    int(y[j])])
        self.cov_ = cov_ / (m - len(self.classes_))
        self._cov_inv = np.linalg.inv(self.cov_)

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
                    sample, self.mu_[ind], self.cov_) * \
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



