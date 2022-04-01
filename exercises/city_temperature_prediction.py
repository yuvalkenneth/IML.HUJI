import numpy as np
import pandas as pd
import plotly.io as pio
import plotly.express as px
import plotly.graph_objects as go
from IMLearn.learners.regressors import PolynomialFitting
from IMLearn.utils.utils import split_train_test



MEAN = 'mean'

COUNTRY = 'Country'

STD = 'std'

MONTH = 'Month'

TEMP = 'Temp'

YEAR = 'Year'

ISRAEL = 'Israel'

DATE = 'Date'

DAY_OF_YEAR  = 'DayOfYear'

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
    data = pd.read_csv(filename, parse_dates=[DATE])
    data = data.drop(data[data.Temp < -40].index)
    data[DAY_OF_YEAR] = data[DATE].dt.dayofyear

    return data


def question4():
    global X, y
    X = israel_data[DAY_OF_YEAR].to_frame()
    y = israel_data[TEMP]
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


def question5():
    israeli_model = PolynomialFitting(5)
    israeli_model.fit(X.to_numpy().reshape(X.shape[0]), y.to_numpy())
    countries = list(data[COUNTRY].drop_duplicates(inplace=False))
    error_by_country = {}
    for country in countries:
        if country == ISRAEL:
            continue
        country_data = data.drop(data[data.Country != country].index)
        X_data = country_data[DAY_OF_YEAR].to_numpy()
        y_data = country_data[TEMP].to_numpy()
        error_by_country[country] = israeli_model.loss(X_data.reshape(
            X_data.shape[0]), y_data)
    fig5 = go.Figure(go.Bar(x=list(error_by_country.keys()), y=list(
        error_by_country.values())))
    fig5.update_layout(xaxis_title=COUNTRY, yaxis_title='MSE',
                       title='MSE for israeli model')
    fig5.show()


def question3():
    country_month = data.groupby([COUNTRY, MONTH]).agg(
        {TEMP: [MEAN, STD]}).reset_index()
    country_month.columns = [COUNTRY, MONTH, 'Mean-Temp', 'Std-Temp']
    fig3 = px.line(country_month, x=MONTH, y='Mean-Temp',
                   error_y='Std-Temp', color=COUNTRY).show()


def question2():
    global israel_data
    israel_data = data.drop(data[data.Country != ISRAEL].index)
    israel_data[YEAR] = israel_data[YEAR].astype(str)
    fig = px.scatter(israel_data, x=DAY_OF_YEAR, y=TEMP,
                     color=YEAR).show()
    df = israel_data.groupby(MONTH).Temp.agg(STD).to_frame()
    df.rename(columns={TEMP: STD}, inplace=True)
    fig2 = px.bar(df, x=df.index, y=STD).show()


if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of city temperature dataset
    data = load_data(r"C:\Users\yuval\Desktop\github\IML.HUJI\datasets"
                     r"\City_Temperature.csv")

    # Question 2 - Exploring Data for specific country
    question2()
    # Question 3 - Exploring differences between countries
    question3()

    # # Question 4 - Fitting model for different values of `k`
    question4()
    # # Question 5 - Evaluating fitted model on different countries
    question5()
