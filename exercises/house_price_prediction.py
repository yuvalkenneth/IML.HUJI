from typing import NoReturn

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio

from IMLearn.learners.regressors import LinearRegression
from IMLearn.utils.utils import split_train_test

SQFT_LOT_15 = 'sqft_lot15'

SQFT_LIVING_15 = 'sqft_living15'

ZIPCODE = 'zipcode'

ID = 'id'

DATE = 'date'

PRICE = 'price'

SQFT_LOT_15_NEW_LABEL = 'sqft_lot area relative to neighbors'

SQFT_LIVING_15_NEW_LABEL = 'sqft_lvng area relative to ' \
                           'neighbors'

pio.templates.default = "simple_white"
pio.renderers.default = "browser"



def load_data(filename: str):
    """
    Load house prices dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (prices) - either as a single
    DataFrame or a Tuple[DataFrame, Series]
    """

    data = pd.read_csv(filename)
    data, labels = clean_data(data)
    return data, labels


def feature_evaluation(X: pd.DataFrame, y: pd.Series,
                       output_path: str = ".") -> NoReturn:
    """
    Create scatter plot between each feature and the response.
        - Plot title specifies feature name
        - Plot title specifies Pearson Correlation between feature and response
        - Plot saved under given folder with file name including feature name
    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_features)
        Design matrix of regression problem

    y : array-like of shape (n_samples, )
        Response vector to evaluate against

    output_path: str (default ".")
        Path to folder in which plots are saved
    """

    corr = pearson_corr(X.iloc[:, 0:17], y)
    for i in corr:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=list(X[i]), y=list(y), mode='markers',
                                 name=f"pearson correlation : {i}"))
        fig.update_layout(xaxis_title=f"{i}", yaxis_title="price"
                          , title=f"pearson correlation : {corr[i]}")
        fig.show()
        fig.write_image(output_path + f"\{i}.png", format='png')


def pearson_corr(X, y):

    features_std = X.std()
    labels_std = y.std()
    cov_vector = X.apply(lambda col: y.cov(col))
    return {key: cov_vector[key] / (features_std[key] * labels_std) for key in
            cov_vector.keys()}


def clean_data(data):
    data = data.drop(data[(data.bathrooms <= 0) | (data.bedrooms <= 0)
                          | (data.sqft_living <= 0) | (data.sqft_lot <= 0)
                          | (data.condition <= 0) | (data.grade <= 0) |
                          (data.price <= 0) | (data.floors <= 0)
                          | (data.sqft_above <= 0) | (
                                  data.yr_built <= 0)].index)
    grade_categorization(data)
    data = drop_duplicates_by_date(data)
    calculate_area_relative_to_15_neighbors(data)
    data.dropna(inplace=True)
    zipcodes_coding = pd.get_dummies(data[ZIPCODE])
    labels = data[PRICE]
    data = data.drop(labels=[PRICE, DATE, ID, ZIPCODE], axis=1)
    data = data.join(zipcodes_coding)
    return data, labels


def calculate_area_relative_to_15_neighbors(data):
    data.sqft_living15 = data.sqft_living / data.sqft_living15
    data.sqft_lot15 = data.sqft_lot / data.sqft_lot15
    data.rename(columns={SQFT_LIVING_15: SQFT_LIVING_15_NEW_LABEL, \
                         SQFT_LOT_15: SQFT_LOT_15_NEW_LABEL},
                inplace=True)


def drop_duplicates_by_date(data):
    data.sort_values(by='date')
    return data.drop_duplicates(subset='id', keep='last')


def grade_categorization(data):
    data.loc[(data['grade'] > 0) & (data['grade'] <= 3), 'grade'] = 1
    data.loc[(data['grade'] > 3) & (data['grade'] <= 7), 'grade'] = 2
    data.loc[(data['grade'] > 7) & (data['grade'] <= 10), 'grade'] = 3
    data.loc[(data['grade'] > 10) & (data['grade'] <= 13), 'grade'] = 4


if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of housing prices dataset
    path = r"C:\Users\yuval\Desktop\github\IML.HUJI\datasets\house_prices.csv"
    X, y = load_data(path)

    #     # Question 2 - Feature evaluation with respect to response
    feature_evaluation(X, y)
    # Question 3 - Split samples into training- and testing sets.
    train_x, train_y, test_x, test_y = split_train_test(X, y, 0.75)
#
#     # Question 4 - Fit model over increasing percentages of the overall training data
#     # For every percentage p in 10%, 11%, ..., 100%, repeat the following 10 times:
#     #   1) Sample p% of the overall training data
#     #   2) Fit linear model (including intercept) over sampled set
#     #   3) Test fitted model over test set
#     #   4) Store average and variance of loss over test set
#     # Then plot average loss as function of training size with error ribbon of size (mean-2*std, mean+2*std)
means_dictionary = {}
for i in range(10, 101):
    current_mean_error = []
    for j in range(10):
        min_train_x, min_train_y, min_test_xm, min_test_y = split_train_test(
            train_x, train_y, i / 100)
        linear_model = LinearRegression()
        linear_model.fit(min_train_x.to_numpy(), min_train_y.to_numpy())
        current_mean_error.append(linear_model.loss(test_x.to_numpy(),
                                                    test_y.to_numpy()))

    means_dictionary[i] = (np.mean(current_mean_error), np.std(
        current_mean_error, ddof=1))
x_axis = list(means_dictionary.keys())
y_axis = np.array([means_dictionary[i][0] for i in means_dictionary])
var_pred = np.array([means_dictionary[i][1] for i in means_dictionary])
fig2 = go.Figure(go.Scatter(x=x_axis, y=y_axis, mode='markers+lines',
                            name="Mean Prediction", line=dict(dash="dash"),
                            marker=dict(color="green", opacity=.7)))
fig2.add_scatter(x=x_axis, y=y_axis+2*var_pred, mode='lines',name="mean + "
                                                                  "2 * std")
fig2.add_scatter(x=x_axis, y=y_axis-2*var_pred, mode='lines',name="mean - "
                                                                  "2 * std")
fig2.show()

print(5)
