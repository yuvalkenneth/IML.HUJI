import sys
from typing import Tuple

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

from IMLearn.learners.classifiers import Perceptron, GaussianNaiveBayes, LDA
from IMLearn.metrics import accuracy

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
        data = np.load(f)
        X = data[:, 0:-1]
        y = data[:, -1]
        # Fit models and predict over training set
        gaussian = GaussianNaiveBayes()
        gaussian.fit(X, y)
        gaussian_pred = gaussian._predict(X)
        gaussian_accuracy = accuracy(y, gaussian_pred)
        linear_discriminant = LDA()
        linear_discriminant.fit(X, y)
        lda_pred = linear_discriminant._predict(X)
        lda_accuracy = accuracy(y, lda_pred)

        # Plot a figure with two suplots, showing the Gaussian Naive Bayes predictions on the left and LDA predictions
        # on the right. Plot title should specify dataset used and subplot titles should specify algorithm and accuracy
        plot_models(X, y, gaussian_pred, lda_pred, gaussian_accuracy,
                    lda_accuracy)


def plot_models(X, y, gauss_pred, lda_pred, gauss_acc, lda_acc):
    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=(f"gaussian naive bayes\naccuracy:"
                                        f"{gauss_acc}",
                                        f"LDA\naccuracy: {lda_acc}"))
    fig.add_trace(
        go.Scatter(x=X[:, 0], y=X[:, 1], mode='markers', marker=dict(
            color=gauss_pred, symbol=y, size=7)),
        row=1, col=1)
    fig.add_trace(
        go.Scatter(x=X[:, 0], y=X[:, 1], mode='markers', marker=dict(
            color=lda_pred, symbol=y, size=7)), row=1, col=2)
    fig.update_layout(height=800, width=1500,
                      title_text="Naive gaussian bayes VS LDA")
    fig.show()


if __name__ == '__main__':
    np.random.seed(0)
    # run_perceptron()
    compare_gaussian_classifiers()
