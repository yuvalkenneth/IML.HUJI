import numpy as np
import pandas as pd
import plotly.io as pio
import plotly.express as px
import plotly.graph_objects as go
from IMLearn.learners.regressors import PolynomialFitting
from IMLearn.utils.utils import split_train_test

pio.templates.default = "simple_white"
pio.renderers.default = "browser"


def load_data(filename: str) -> pd.DataFrame:
    """
    Load city daily temperature dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (Temp)
    """
    data = pd.read_csv(filename, parse_dates=['Date'])
    data = data.drop(data[data.Temp < -40].index)
    data['DayOfYear'] = data['Date'].dt.dayofyear

    return data


if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of city temperature dataset
    data = load_data(r"C:\Users\yuval\Desktop\github\IML.HUJI\datasets"
                     r"\City_Temperature.csv")

    # Question 2 - Exploring Data for specific country
    israel_data = data.drop(data[data.Country != 'Israel'].index)
    israel_data['Year'] = israel_data['Year'].astype(str)
    # fig = px.scatter(israel_data, x='DayOfYear', y='Temp',
    #                  color='Year').show()
    df = israel_data.groupby('Month').Temp.agg('std').to_frame()
    df.rename(columns={'Temp': 'std'}, inplace=True)
    # fig2 = px.bar(df, x=df.index, y='std').show()
    # Question 3 - Exploring differences between countries
    country_month = data.groupby(['Country', 'Month']).agg(
        {'Temp': ['mean', 'std']}).reset_index()
    country_month.columns = ['Country', 'Month', 'Mean-Temp', 'Std-Temp']
    # fig3 = px.line(country_month, x='Month', y='Mean-Temp',
    #                error_y='Std-Temp', color='Country').show()
    print(4)

    # # Question 4 - Fitting model for different values of `k`
    X = israel_data['DayOfYear'].to_frame()
    y = israel_data['Temp']
    train_x, train_y, test_x, test_y = split_train_test(X, y, 0.75)
    degrees_err = {}
    for i in range(1, 11):
        model = PolynomialFitting(i)
        model.fit(train_x.to_numpy().reshape(train_x.shape[0]),
                  train_y.to_numpy())
        loss = round(model.loss(test_x.to_numpy().reshape(test_x.shape[0]),
                                test_y.to_numpy()), 2)
        degrees_err[i] = loss

    print(degrees_err)
    fig4 = go.Figure(go.Bar(x=list(degrees_err.keys()), y=list(
        degrees_err.values())))
    fig4.update_layout(xaxis_title='degree of polynomial', yaxis_title='MSE',
                       title='MSE in relation to degree')
    fig4.show()
    # # Question 5 - Evaluating fitted model on different countries
    # raise NotImplementedError()
