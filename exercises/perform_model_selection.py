from __future__ import annotations

from IMLearn.learners.regressors.polynomial_fitting import PolynomialFitting
from IMLearn.metrics.loss_functions import mean_square_error
from IMLearn.model_selection.cross_validate import cross_validate
from IMLearn.utils import split_train_test
from utils import *

SAMPEL_SIZE = 100

MAX_VAL = 2

MIN_VAL = -1.2


def select_polynomial_degree(n_samples: int = 100, noise: float = 5):
    """
    Simulate data from a polynomial model and use cross-validation to select the best fitting degree

    Parameters
    ----------
    n_samples: int, default=100
        Number of samples to generate

    noise: float, default = 5
        Noise level to simulate in responses
    """
    # Question 1 - Generate dataset for model f(x)=(x+3)(x+2)(x+1)(x-1)(x-2) + eps for eps Gaussian noise
    # and split into training- and testing portions
    response = lambda x: (x + 3) * (x + 2) * (x + 1) * (x - 1) * (x - 2)
    x = np.linspace(MIN_VAL, MAX_VAL, n_samples)
    y_noiseless = response(x)
    y_noise = y_noiseless + np.random.normal(0, noise, n_samples)
    train_X, train_y, test_X, test_y = split_train_test(pd.DataFrame(x),
                                                        pd.Series(
                                                            y_noise).rename(
                                                            "labels"), 2 / 3)
    train_X, train_y, test_X, test_y = np.array(train_X).reshape(len(
        train_X)), np.array(train_y), np.array(test_X).reshape(len(
        test_X)), np.array(test_y)
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=x, y=y_noiseless, mode="markers+lines"))

    fig1.add_trace(go.Scatter(x=train_X, y=train_y,
                              mode="markers", marker=dict(color='red'),
                              name="train points"))
    fig1.add_trace(go.Scatter(x=test_X, y=test_y,
                              mode="markers", marker=dict(color='yellow'),
                              name="test points"))

    fig1.show()
    # Question 2 - Perform CV for polynomial fitting with degrees 0,1,...,10
    error = []
    for k in range(11):
        error.append(cross_validate(PolynomialFitting(k), train_X, train_y,
                                    mean_square_error))
        # raise NotImplementedError()

    # Question 3 - Using best value of k, fit a k-degree polynomial model and report test error
    best_k = np.argmin([err[1] for err in error])
    model = PolynomialFitting(int(best_k))
    model.fit(train_X, train_y)
    test_err = model.loss(test_X, test_y)
    print(3)


def select_regularization_parameter(n_samples: int = 50,
                                    n_evaluations: int = 500):
    """
    Using sklearn's diabetes dataset use cross-validation to select the best fitting regularization parameter
    values for Ridge and Lasso regressions

    Parameters
    ----------
    n_samples: int, default=50
        Number of samples to generate

    n_evaluations: int, default = 500
        Number of regularization parameter values to evaluate for each of the algorithms
    """
    # Question 6 - Load diabetes dataset and split into training and testing portions
    raise NotImplementedError()

    # Question 7 - Perform CV for different values of the regularization parameter for Ridge and Lasso regressions
    raise NotImplementedError()

    # Question 8 - Compare best Ridge model, best Lasso model and Least Squares model
    raise NotImplementedError()


if __name__ == '__main__':
    np.random.seed(0)
    # raise NotImplementedError()
    # select_polynomial_degree(100, 5)
    # select_polynomial_degree(100, 0)
    select_polynomial_degree(1500, 10)
