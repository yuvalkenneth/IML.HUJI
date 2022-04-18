import sys
from typing import Tuple

import numpy as np
import plotly.express as px
import plotly.io as pio

from IMLearn.learners.classifiers import Perceptron, GaussianNaiveBayes,LDA

pio.templates.default = "simple_white"
sys.path.append(r"C:\Users\yuval\Desktop\github\IML.HUJI\datasets")


class loss_report():
    def __init__(self):
        self.loss = []

    def loss_callback(self, fit: Perceptron, samples, responses):
        self.loss.append(fit.loss(samples, responses))


def load_dataset(filename: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load dataset for comparing the Gaussian Naive Bayes and LDA classifiers. File is assumed to be an
    ndarray of shape (n_samples, 3) where the first 2 columns represent features and the third column the class

    Parameters
    ----------
    filename: str
        Path to .npy data file

    Returns
    -------
    X: ndarray of shape (n_samples, 2)
        Design matrix to be used

    y: ndarray of shape (n_samples,)
        Class vector specifying for each sample its class

    """
    data = np.load(filename)
    return data[:, 0:-1], data[:, -1]


def run_perceptron():
    """
    Fit and plot fit progression of the Perceptron algorithm over both the linearly separable and inseparable datasets

    Create a line plot that shows the perceptron algorithm's training loss values (y-axis)
    as a function of the training iterations (x-axis).
    """
    for n, f in [("Linearly Separable", "linearly_separable.npy"),
                 ("Linearly Inseparable", "linearly_inseparable.npy")]:
        # Load dataset
        X, y = load_dataset(f)
        # Fit Perceptron and record loss in each fit iteration
        report = loss_report()
        model = Perceptron()
        model.callback_ = report.loss_callback
        model.fit(X, y)
        loss = report.loss

        # Plot figure
        px.line(y=loss).show()


def compare_gaussian_classifiers():
    """
    Fit both Gaussian Naive Bayes and LDA classifiers on both gaussians1 and gaussians2 datasets
    """
    for f in ["gaussian1.npy", "gaussian2.npy"]:
        # Load dataset
        raise NotImplementedError()

        # Fit models and predict over training set
        raise NotImplementedError()

        # Plot a figure with two suplots, showing the Gaussian Naive Bayes predictions on the left and LDA predictions
        # on the right. Plot title should specify dataset used and subplot titles should specify algorithm and accuracy
        raise NotImplementedError()


if __name__ == '__main__':
    np.random.seed(0)
    # run_perceptron()
    # compare_gaussian_classifiers()
    new = LDA()
    data = np.load(r"gaussian1.npy")
    new.fit(data[:,0:-1], data[:,-1])
    y = new._predict(data[:,0:-1])
    # z = lda()
    # new.loss(data[:,0:-1], data[:,-1])
    # z.fit(data[:,0:-1], data[:,-1])
    # k = z.predict_proba(data[:,0:-1])
    # test = GaussianNaiveBayes()
    # test.fit(data[:, 0:-1], data[:, -1])
    # y = test.likelihood(data[:, 0:-1])

    print(5)
