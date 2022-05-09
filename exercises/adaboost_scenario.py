from typing import Tuple

from IMLearn.learners.classifiers import DecisionStump
from IMLearn.metalearners.adaboost import AdaBoost
from IMLearn.metrics import accuracy
from utils import *

WEIGHTD_DATA_TITLE = "train data prediction proportional to point" \
    "weight"

DECISION_BOUNDARIES_TITLE = rf"$\text{{Decision Boundaries Of Models with " \
                            "[5, 50, 100," + "250] learners}}$ "

LOSSES_OF_LEARNERS = "train and test loss as a function of fitted " \
                     "learners"

TEST_LOSS = "test loss"

TRAIN_LOSS = "train loss"

pio.renderers.default = "browser"


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
                name=TRAIN_LOSS))
    fig1.add_trace(
        go.Line(x=list(range(1, n_learners)), y=test_loss[:n_learners],
                name=TEST_LOSS))
    fig1.update_layout(title=LOSSES_OF_LEARNERS)
    fig1.show()

    # Question 2: Plotting decision surfaces
    T = [5, 50, 100, 250]
    lims = np.array([np.r_[train_X, test_X].min(axis=0),
                     np.r_[train_X, test_X].max(axis=0)]).T + np.array(
        [-.1, .1])
    model_predictions = [ada_model.partial_predict(test_X, t) for t in T]
    fig2 = make_subplots(rows=2, cols=3,
                         subplot_titles=[rf"$\text{{{t} learners}}$" for
                                         t in T],
                         horizontal_spacing=0.01, vertical_spacing=.03)
    symbols = np.array(["circle", "x"])
    y = np.where(test_y == 1, 1, 0)
    for i, m in enumerate(T):
        fig2.add_traces(
            [decision_surface(lambda x: ada_model.partial_predict(x,
                                                                  T=m),
                              lims[0], lims[1], showscale=False),
             go.Scatter(x=test_X[:, 0], y=test_X[:, 1],
                        mode="markers",
                        showlegend=False,
                        marker=dict(color=y, symbol=symbols[y],
                                    colorscale=[custom[0],
                                                custom[-1]],
                                    line=dict(color="black",
                                              width=1)))],
            rows=(i // 3) + 1, cols=(i % 3) + 1)

    fig2.update_layout(
        title=DECISION_BOUNDARIES_TITLE
        , margin=dict(t=100)).update_xaxes(visible=False).update_yaxes(
        visible=False)
    fig2.show()

    # # Question 3: Decision surface of best performing
    best_ensemble = int(np.argmin(test_loss))
    acc = accuracy(ada_model.partial_predict(test_X, best_ensemble + 1),
                   test_y)
    fig3 = go.Figure().add_traces([decision_surface(
        lambda x: ada_model.partial_predict(x, T=best_ensemble + 1),
        lims[0], lims[1], showscale=False),
        go.Scatter(x=test_X[:, 0], y=test_X[:, 1],
                   mode="markers",
                   showlegend=False,
                   marker=dict(color=y,
                               symbol=symbols[y],
                               colorscale=[
                                   custom[0],
                                   custom[-1]],
                               line=dict(
                                   color="black",
                                   width=1)))])
    fig3.update_layout(title=f"best ensemble of {best_ensemble} "
                             f"learners, with {acc} accuracy")
    fig3.show()

    # # Question 4: Decision surface with weighted samples
    # raise NotImplementedError()
    normal = ada_model.D_ / np.max(ada_model.D_) * 5
    fig4 = go.Figure().add_traces([decision_surface(
        lambda x: ada_model.partial_predict(x, T=250),
        lims[0], lims[1], showscale=False),
        go.Scatter(x=train_X[:, 0], y=train_X[:, 1],
                   mode="markers",
                   showlegend=False,
                   marker=dict(color=y, size=normal,
                               symbol=symbols[y],
                               colorscale=[
                                   custom[0],
                                   custom[-1]],
                               line=dict(
                                   color="black",
                                   width=1)))])
    fig4.update_layout(title=WEIGHTD_DATA_TITLE)
    fig4.show()


if __name__ == '__main__':
    np.random.seed(0)
    fit_and_evaluate_adaboost(0, 250, 5000, 500)
    fit_and_evaluate_adaboost(0.4)
