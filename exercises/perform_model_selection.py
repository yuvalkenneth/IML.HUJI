from __future__ import annotations

from sklearn.datasets import load_diabetes

from IMLearn.desent_methods.gradient_descent import GradientDescent
from IMLearn.learners.regressors import LinearRegression
# from IMLearn.learners.regressors.lasso_regression import LassoRegression
from sklearn.linear_model import Lasso
from IMLearn.learners.regressors.polynomial_fitting import PolynomialFitting
from IMLearn.learners.regressors.ridge_regression import RidgeRegression
from IMLearn.metrics.loss_functions import mean_square_error
from IMLearn.model_selection.cross_validate import cross_validate
from IMLearn.utils import split_train_test
from utils import *

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
                              mode="markers", marker=dict(color='black'),
                              name="test points"))

    fig1.show()
    # Question 2 - Perform CV for polynomial fitting with degrees 0,1,...,10
    error = []
    for k in range(11):
        error.append(cross_validate(PolynomialFitting(k), train_X, train_y,
                                    mean_square_error))
    val_err = [err[1] for err in error]
    train_err = [err[0] for err in error]
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=list(range(11)), y=val_err,
                              mode="markers+lines", marker=dict(
            color='black'), name="val error"))
    fig2.add_trace(go.Scatter(x=list(range(11)), y=train_err,
                              mode="markers+lines", marker=dict(
            color="green") ,name="train err"))
    fig2.update_layout(title="validation and train error as a function of "
                             "polynomial "
                             "degree", xaxis_title="polynomial degree",
                       yaxis_title="error")
    fig2.show()
    # Question 3 - Using best value of k, fit a k-degree polynomial model and report test error
    best_k = np.argmin([err[1] for err in error])
    print(error[best_k])
    model = PolynomialFitting(int(best_k))
    model.fit(train_X, train_y)
    test_err = model.loss(test_X, test_y)
    print(test_err)


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
    data = load_diabetes()
    X, y = data.data, data.target
    train_X, train_y, test_X, test_y = X[:50], y[:50], X[50:], y[50:]
    # Question 7 - Perform CV for different values of the regularization parameter for Ridge and Lasso regressions
    ridge_params = np.linspace(0, 1, n_evaluations)
    lasso_params = np.linspace(0, 1, n_evaluations)
    ridge_errs, lasso_errs = [], []
    for i in range(n_evaluations):
        ridge_model = RidgeRegression(float(ridge_params[i]))
        lasso_model = Lasso(lasso_params[i])

        ridge_errs.append(cross_validate(ridge_model, train_X, train_y,
                                         mean_square_error, cv=5))
        lasso_errs.append(cross_validate(lasso_model, train_X, train_y,
                                         mean_square_error, cv=5))

    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(x=ridge_params, y=[err[0] for err in ridge_errs],
                              mode="markers+lines", name="train error"))

    fig3.add_trace(go.Scatter(x=ridge_params, y=[err[1] for err in ridge_errs],
                              mode="markers+lines", marker=dict(color='red'),
                              name="validation error"))
    fig3.update_layout(title="ridge train and validation errors a function "
                             "of lambda")
    fig3.show()
    fig4 = go.Figure()
    fig4.add_trace(go.Scatter(x=lasso_params, y=[err[0] for err in lasso_errs],
                              mode="markers+lines", name="train error"))

    fig4.add_trace(go.Scatter(x=lasso_params, y=[err[1] for err in lasso_errs],
                              mode="markers+lines", marker=dict(color='red'),
                              name="validation error"))
    fig4.update_layout(title="lasso train and validation errors a function "
                             "of lambda")
    fig4.show()

    # Question 8 - Compare best Ridge model, best Lasso model and Least Squares model
    best_ridge_lam = ridge_params[np.argmin([err[1] for err in ridge_errs])]
    best_lasso_lam = lasso_params[np.argmin([err[1] for err in lasso_errs])]
    best_ridge = RidgeRegression(best_ridge_lam)
    best_lasso = Lasso(best_lasso_lam)
    linear_model = LinearRegression()
    linear_model.fit(train_X, train_y)
    best_ridge.fit(train_X, train_y)
    best_lasso.fit(train_X, train_y)
    linear_test_err = linear_model.loss(test_X, test_y)
    ridge_test_err = mean_square_error(test_y, best_ridge.predict(test_X))
    lasso_test_err = mean_square_error(test_y, best_lasso.predict(test_X))

    for i in [best_ridge_lam, best_lasso_lam, linear_test_err, ridge_test_err
        , lasso_test_err]:
        print(print(i))


if __name__ == '__main__':
    np.random.seed(0)
    select_polynomial_degree(100, 5)
    select_polynomial_degree(100, 0)
    select_polynomial_degree(1500, 10)
    select_regularization_parameter()
