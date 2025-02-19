import numpy as np
import pandas as pd
from typing import Tuple, List
from IMLearn.metrics.loss_functions import accuracy
from IMLearn.learners.neural_networks.modules import FullyConnectedLayer, ReLU, \
    CrossEntropyLoss, Id
from IMLearn.learners.neural_networks.neural_network import NeuralNetwork
from IMLearn.desent_methods import GradientDescent, FixedLR
from IMLearn.utils.utils import split_train_test
from utils import *

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio

pio.templates.default = "simple_white"


def get_gd_state__callback():
    """
    Callback generator for the GradientDescent class, recording the objective's value and parameters at each iteration

    Return:
    -------
    callback: Callable[[], None]
        Callback function to be passed to the GradientDescent class, recoding the objective's value and parameters
        at each iteration of the algorithm

    values: List[np.ndarray]
        Recorded objective values

    weights: List[np.ndarray]
        Recorded parameters

    gradients: List[np.array]
    """
    weights, values, gradients = [], [], []

    def recorder(obj, lst):
        weights.append(lst[0])
        values.append(lst[1])
        gradients.append(lst[2])

    return recorder, weights, values, gradients


def generate_nonlinear_data(
        samples_per_class: int = 100,
        n_features: int = 2,
        n_classes: int = 2,
        train_proportion: float = 0.8) -> \
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Create a multiclass non linearly-separable dataset. Adopted from Stanford CS231 course code.

    Parameters:
    -----------
    samples_per_class: int, default = 100
        Number of samples per class

    n_features: int, default = 2
        Data dimensionality

    n_classes: int, default = 2
        Number of classes to generate

    train_proportion: float, default=0.8
        Proportion of samples to be used for train set

    Returns:
    --------
    train_X : ndarray of shape (ceil(train_proportion * n_samples), n_features)
        Design matrix of train set

    train_y : ndarray of shape (ceil(train_proportion * n_samples), )
        Responses of training samples

    test_X : ndarray of shape (floor((1-train_proportion) * n_samples), n_features)
        Design matrix of test set

    test_y : ndarray of shape (floor((1-train_proportion) * n_samples), )
        Responses of test samples
    """
    X, y = np.zeros((samples_per_class * n_classes, n_features)), np.zeros(
        samples_per_class * n_classes, dtype='uint8')
    for j in range(n_classes):
        ix = range(samples_per_class * j, samples_per_class * (j + 1))
        r = np.linspace(0.0, 1, samples_per_class)  # radius
        t = np.linspace(j * 4, (j + 1) * 4,
                        samples_per_class) + np.random.randn(
            samples_per_class) * 0.2  # theta
        X[ix], y[ix] = np.c_[r * np.sin(t), r * np.cos(t)], j

    split = split_train_test(pd.DataFrame(X), pd.Series(y), train_proportion)
    return tuple(map(lambda x: x.values, split))


def plot_decision_boundary(nn: NeuralNetwork, lims, X: np.ndarray = None,
                           y: np.ndarray = None, title=""):
    data = [decision_surface(nn.predict, lims[0], lims[1], density=40,
                             showscale=False)]
    if X is not None:
        col = y if y is not None else "black"
        data += [go.Scatter(x=X[:, 0], y=X[:, 1], mode="markers",
                            marker=dict(color=col, colorscale=custom,
                                        line=dict(color="black", width=1)))]

    return go.Figure(data,
                     go.Layout(
                         title=rf"$\text{{Network Decision Boundaries {title}}}$",
                         xaxis=dict(title=r"$x_1$"),
                         yaxis=dict(title=r"$x_2$"),
                         width=400, height=400))


def animate_decision_boundary(nn: NeuralNetwork, weights: List[np.ndarray],
                              lims, X: np.ndarray, y: np.ndarray,
                              title="", save_name=None):
    frames = []
    for i, w in enumerate(weights):
        nn.weights = w
        frames.append(go.Frame(data=[
            decision_surface(nn.predict, lims[0], lims[1], density=40,
                             showscale=False),
            go.Scatter(x=X[:, 0], y=X[:, 1], mode="markers",
                       marker=dict(color=y, colorscale=custom,
                                   line=dict(color="black", width=1)))
        ],
            layout=go.Layout(
                title=rf"$\text{{{title} Iteration {i + 1}}}$")))

    fig = go.Figure(data=frames[0]["data"], frames=frames[1:],
                    layout=go.Layout(title=frames[0]["layout"]["title"]))
    if save_name:
        animation_to_gif(fig, save_name, 200, width=400, height=400)


def get_layers(features, width, classes, layers=True):
    if not layers:
        return [FullyConnectedLayer(n_features, n_classes, Id())]
    first_layer = FullyConnectedLayer(features, width, ReLU())
    second_layer = FullyConnectedLayer(width, width, ReLU())
    last_layer = FullyConnectedLayer(width, n_classes, Id())
    return [first_layer, second_layer, last_layer]


if __name__ == '__main__':
    np.random.seed(0)

    # Generate and visualize dataset
    n_features, n_classes = 2, 3
    train_X, train_y, test_X, test_y = generate_nonlinear_data(
        samples_per_class=500, n_features=n_features, n_classes=n_classes,
        train_proportion=0.8)
    lims = np.array([np.r_[train_X, test_X].min(axis=0),
                     np.r_[train_X, test_X].max(axis=0)]).T + np.array(
        [-.1, .1])

    go.Figure(
        data=[go.Scatter(x=train_X[:, 0], y=train_X[:, 1], mode='markers',
                         marker=dict(color=train_y, colorscale=custom,
                                     line=dict(color="black", width=1)))],
        layout=go.Layout(title=r"$\text{Train Data}$",
                         xaxis=dict(title=r"$x_1$"),
                         yaxis=dict(title=r"$x_2$"),
                         width=400, height=400)) \
        # .write_image(f"../figures/nonlinear_data.png")

    # ---------------------------------------------------------------------------------------------#
    # Question 1: Fitting simple network with two hidden layers                                    #
    # ---------------------------------------------------------------------------------------------#
    callback1, weights1, values1, grads1 = get_gd_state__callback()
    callback2, weights2, values2, grad2 = get_gd_state__callback()
    model = NeuralNetwork(get_layers(n_features, 16, n_classes),
                          CrossEntropyLoss(),
                          GradientDescent(FixedLR(1e-1), max_iter=5000,
                                          callback=get_gd_state__callback()[
                                              0]))
    model.fit(train_X, train_y)
    print(f"accuracy: {accuracy(test_y, model.predict(test_X))}")
    plot_decision_boundary(model, lims, train_X, train_y, title="Two layered "
                                                                "network").show()

    # ---------------------------------------------------------------------------------------------#
    # Question 2: Fitting a network with no hidden layers                                          #
    # ---------------------------------------------------------------------------------------------#
    model2 = NeuralNetwork(get_layers(n_features, 0, n_classes, False),
                           loss_fn=CrossEntropyLoss(),
                           solver=GradientDescent(FixedLR(0.1), max_iter=5000,
                                                  callback=
                                                  get_gd_state__callback()[
                                                      0]))
    model2.fit(train_X, train_y)
    plot_decision_boundary(model2, lims, train_X, train_y, title="No hidden "
                                                                 "layers") \
        .show()

    # ---------------------------------------------------------------------------------------------#
    # Question 3+4: Plotting network convergence process                                           #
    # ---------------------------------------------------------------------------------------------#

    weights = weights1[::100]
    # try:
    animate_decision_boundary(model, weights, lims, train_X, train_y,
                              title=
                              "convergence")
    # except:
    model3 = NeuralNetwork(get_layers(n_features, 16, n_classes),
                           CrossEntropyLoss(), GradientDescent(FixedLR(
            1e-1), max_iter=5000, callback=callback1))
    fig1 = go.Figure().add_trace(go.Scatter(x=list(range(5000)), y=values1,
                                            name=f"convergence"))
    fig1.add_trace(
        go.Scatter(x=list(range(5000)), y=[np.linalg.norm(g) for g in
                                           grads1]))
    fig1.show()
    # first_layer = FullyConnectedLayer(n_features, 6, ReLU())
    # second_layer = FullyConnectedLayer(6, 6, ReLU())
    # last_layer = FullyConnectedLayer(6, n_classes, Id())
    model4 = NeuralNetwork(get_layers(n_features, 6, n_classes),
                           CrossEntropyLoss(),
                           GradientDescent(FixedLR(0.1),
                                           max_iter=5000,
                                           callback=
                                           callback2))
    model4.fit(train_X, train_y)
    weights = weights2[::100]
    fig2 = go.Figure().add_trace(go.Scatter(x=list(range(5000)), y=values2,
                                            name=f"convergence"))
    fig2.add_trace(
        go.Scatter(x=list(range(5000)),
                   y=[np.linalg.norm(g) for g in grad2]))
    fig2.show()
