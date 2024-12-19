import streamlit as st
import numpy as np
import plotly.graph_objects as go
from scipy.stats import multivariate_normal

# Introduction
st.title("Multivariate Random Variables")
st.markdown(
    """
### Introduction
This app explains the concept of multivariate random variables, focusing on the PDF and CDF of 2D continuous random variables. It also visually explains marginal distributions.

### Multivariate Normal Distribution
The multivariate normal distribution is a generalization of the one-dimensional normal distribution to higher dimensions. It is defined by a mean vector and a covariance matrix.

The PDF of a 2D multivariate normal distribution is given by:

$$
f(x, y) = \\frac{1}{2\\pi\\sqrt{\\det(\\Sigma)}} \\exp\\left(-\\frac{1}{2} \\mathbf{z}^T \\Sigma^{-1} \\mathbf{z}\\right)
$$

where $ \\mathbf{z} = \\begin{pmatrix} x - \\mu_x \\\\ y - \\mu_y \\end{pmatrix} $ and $ \\Sigma $ is the covariance matrix.
"""
)

# User inputs for mean and covariance matrix
mean_x = st.slider("Mean of X (μx)", min_value=-5.0, max_value=5.0, value=0.0, step=0.1)
mean_y = st.slider("Mean of Y (μy)", min_value=-5.0, max_value=5.0, value=0.0, step=0.1)
cov_xx = st.slider("Variance of X (σ²x)", min_value=0.1, max_value=5.0, value=1.0, step=0.1)
cov_yy = st.slider("Variance of Y (σ²y)", min_value=0.1, max_value=5.0, value=1.0, step=0.1)
cov_xy = st.slider("Covariance (σxy)", min_value=-2.0, max_value=2.0, value=0.0, step=0.1)

mean = [mean_x, mean_y]
cov = [[cov_xx, cov_xy], [cov_xy, cov_yy]]

# Generate grid for plotting
x = np.linspace(-10, 10, 500)
y = np.linspace(-10, 10, 500)
X, Y = np.meshgrid(x, y)
pos = np.dstack((X, Y))

# Multivariate normal distribution
rv = multivariate_normal(mean, cov)
Z = rv.pdf(pos)

# Plot PDF
fig_pdf = go.Figure(data=[go.Surface(z=Z, x=X, y=Y)])
fig_pdf.update_layout(
    title="PDF of 2D Multivariate Normal Distribution",
    scene=dict(
        xaxis_title="X",
        yaxis_title="Y",
        zaxis_title="f(x, y)",
    ),
)

# Marginal distributions
marginal_x = np.sum(Z, axis=0) * (y[1] - y[0])
marginal_y = np.sum(Z, axis=1) * (x[1] - x[0])

# Plot marginal distributions
fig_marginal_x = go.Figure()
fig_marginal_x.add_trace(go.Scatter(x=x, y=marginal_x, mode="lines", name="Marginal X"))
fig_marginal_x.update_layout(
    title="Marginal Distribution of X",
    xaxis_title="X",
    yaxis_title="Density",
)

fig_marginal_y = go.Figure()
fig_marginal_y.add_trace(go.Scatter(x=y, y=marginal_y, mode="lines", name="Marginal Y"))
fig_marginal_y.update_layout(
    title="Marginal Distribution of Y",
    xaxis_title="Y",
    yaxis_title="Density",
)

# Display plots
st.plotly_chart(fig_pdf, use_container_width=True)
st.plotly_chart(fig_marginal_x, use_container_width=True)
st.plotly_chart(fig_marginal_y, use_container_width=True)

# Explanation of marginal distributions
st.markdown(
    """
### Marginal Distributions
The marginal distribution of a subset of a collection of random variables is the probability distribution of the variables in the subset. It is obtained by integrating the joint probability distribution over the other variables.

In the case of the 2D multivariate normal distribution, the marginal distributions of X and Y are both normal distributions with means and variances given by the corresponding elements of the mean vector and covariance matrix.
"""
)
