import numpy as np
import pandas as pd
import plotly.io as pio
import plotly.express as px
pio.templates.default = "simple_white"


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
    avg = israel_data.groupby('DayOfYear', as_index=False)['Temp'].mean()
    fig = px.scatter(israel_data, x='DayOfYear', y='Temp',
                                                       color='Year').show()
    # Question 3 - Exploring differences between countries
    raise NotImplementedError()

    # Question 4 - Fitting model for different values of `k`
    raise NotImplementedError()

    # Question 5 - Evaluating fitted model on different countries
    raise NotImplementedError()
