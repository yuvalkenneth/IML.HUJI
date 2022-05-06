from typing import Tuple

from IMLearn.learners.classifiers import DecisionStump
from IMLearn.metalearners.adaboost import AdaBoost
from utils import *


def generate_data(n: int, noise_ratio: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a dataset in R^2 of specified size

    Parameters
    ----------
    n: int
        Number of samples to generate

    noise_ratio: float
        Ratio of labels to invert

    Returns
    -------
    X: np.ndarray of shape (n_samples,2)
        Design matrix of samples

    y: np.ndarray of shape (n_samples,)
        Labels of samples
    """
    '''
    generate samples X with shape: (num_samples, 2) and labels y with shape (num_samples).
    num_samples: the number of samples to generate
    noise_ratio: invert the label for this ratio of the samples
    '''
    X, y = np.random.rand(n, 2) * 2 - 1, np.ones(n)
    y[np.sum(X ** 2, axis=1) < 0.5 ** 2] = -1
    y[np.random.choice(n, int(noise_ratio * n))] *= -1
    return X, y


def fit_and_evaluate_adaboost(noise, n_learners=250, train_size=5000,
                              test_size=500):
    (train_X, train_y), (test_X, test_y) = generate_data(train_size,
                                                         noise), generate_data(
        test_size, noise)

    # Question 1: Train- and test errors of AdaBoost in noiseless case
    ada_model = AdaBoost(wl=lambda: DecisionStump(), iterations=n_learners)
    ada_model.fit(train_X, train_y)

    train_loss = []
    test_loss = []
    for i in range(1, n_learners + 1):
        train_loss.append(ada_model.partial_loss(train_X, train_y, i))
        test_loss.append(ada_model.partial_loss(test_X, test_y, i))
    fig1 = go.Figure()
    fig1.add_trace(
        go.Line(x=list(range(1, n_learners)), y=train_loss[:n_learners],
                name="train loss"))
    fig1.add_trace(
        go.Line(x=list(range(1, n_learners)), y=test_loss[:n_learners],
                name="test loss"))
    fig1.show()

    # Question 2: Plotting decision surfaces
    T = [5, 50, 100, 250]
    lims = np.array([np.r_[train_X, test_X].min(axis=0),
                     np.r_[train_X, test_X].max(axis=0)]).T + np.array(
        [-.1, .1])
    model-predictions = [ada_model.partial_predict(test_X, t) for t in T]
    # raise NotImplementedError()
    #
    # # Question 3: Decision surface of best performing ensemble
    # raise NotImplementedError()
    #
    # # Question 4: Decision surface with weighted samples
    # raise NotImplementedError()
    #


if __name__ == '__main__':
    np.random.seed(0)
    fit_and_evaluate_adaboost(0, 50,2500,250)
