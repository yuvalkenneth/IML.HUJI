import numpy as np
import plotly.graph_objects as go
import plotly.io as pio

from IMLearn.learners import MultivariateGaussian, UnivariateGaussian

HEAT_MAP_TITLE = "log-likelihood evaluation"

pio.templates.default = "simple_white"
pio.renderers.default = "browser"


def test_univariate_gaussian():
    # Question 1 - Draw samples and print fitted model
    actual_mu = 10
    uni_gauss = UnivariateGaussian()
    sample = np.random.normal(actual_mu, 1, 1000)
    uni_gauss.fit(sample)
    print(uni_gauss.mu_, uni_gauss.var_)

    # Question 2 - Empirically showing sample mean is consistent
    Y = []
    gauss = UnivariateGaussian()
    for i in range(10, 1010, 10):
        gauss.fit(sample[:i])
        Y.append(np.abs(gauss.mu_ - actual_mu))
    X = np.arange(10, 1010, 10)
    go.Figure([go.Scatter(x=X, y=Y, mode='markers+lines',
                          name=r'$\widehat\sigma^2$')],
              layout=go.Layout(
                  title=r"$\text{Distance between estimated and true "
                        r"expectation}$",
                  xaxis_title="$\\text{sample size}$",
                  yaxis_title="r$|estimated-actual|$",
                  height=700)).show()

    # Question 3 - Plotting Empirical PDF of fitted model
    X = np.sort(sample)
    go.Figure([go.Scatter(x=X, y=uni_gauss.pdf(X), mode='markers+lines',
                          name=r'$\widehat\sigma^2$')],
              layout=go.Layout(
                  title=r"$\text{Empirical PDF}$",
                  xaxis_title="$\\text{sample value}$",
                  yaxis_title="r$pdf value$",
                  height=700)).show()


#

def test_multivariate_gaussian():
    # Question 4 - Draw samples and print fitted model
    cov_matrix = [[1, 0.2, 0, 0.5], [0.2, 2, 0, 0], [0, 0, 1, 0],
                  [0.5, 0, 0, 1]]
    samples = np.random.multivariate_normal([0, 0, 4, 0], cov_matrix, 1000)
    multi_gauss = MultivariateGaussian()

    multi_gauss.fit(samples)
    print(multi_gauss.mu_, '\n', multi_gauss.cov_)

    # Question 5 - Likelihood evaluation
    data = np.linspace(-10, 10, 200)
    Z = []
    maximum = -np.inf
    coor = ()

    for f1 in data:
        likelihood = []
        for f3 in data:
            log_like = multi_gauss.log_likelihood([f1, 0, f3, 0], cov_matrix,
                                                  samples)
            likelihood.append(log_like)
            if log_like > maximum:
                maximum = log_like
                coor = (f1, f3)

        Z.append(likelihood)
    Z = np.transpose(Z)
    go.Figure(go.Heatmap(x=data, y=data, z=Z)).update_layout(
        title=HEAT_MAP_TITLE
        , xaxis_title="f1 values", yaxis_title="f3  values",
        height=800, width=1500).show()

    # Question 6 - Maximum likelihood
    print(
        f"maximum log-likelihood in : "
        f"{[format(coor[i], '0.3f') for i in [0, 1]]}"
        f" and the value is : {format(maximum, '0.3f')}")


if __name__ == '__main__':
    np.random.seed(0)
    test_univariate_gaussian()
    test_multivariate_gaussian()
