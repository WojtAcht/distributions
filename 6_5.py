import streamlit as st
import numpy as np
import plotly.graph_objects as go
from scipy.integrate import quad
from scipy.optimize import brentq

# Introduction
st.title("Analiza rozkładu zmiennej losowej")
st.markdown(
    r"""
### Problem
Rozważamy funkcję gęstości \( f(x) \) zadaną jako:

$$
f(x) =
\begin{cases} 
a(1-x), & \text{dla } 0 \leq x \leq 1, \\
a(x-1), & \text{dla } 1 \leq x \leq 2, \\
0, & \text{w pozostałych przypadkach.}
\end{cases}
$$

Znajdź:
1. Wartość stałej \( a \), aby \( f(x) \) była funkcją gęstości.
2. Dystrybuantę \( F(x) \).
"""
)

a = st.slider(
    "Wybierz wartość współczynnika \( a \):",
    min_value=-5.0,
    max_value=5.0,
    value=1.0,
    step=0.1,
)


def pdf(x, a):
    if 0 <= x <= 1:
        return a * (1 - x)
    elif 1 <= x <= 2:
        return a * (x - 1)
    else:
        return 0


def normalize_a(a):
    integral, _ = quad(lambda x: pdf(x, a), 0, 2)
    return integral - 1  # Should equal 0 for a valid PDF


valid_a = brentq(normalize_a, 0, 10)


def cdf(x, a):
    if x < 0:
        return 0
    elif 0 <= x <= 1:
        return a * (x - 0.5 * x**2)
    elif 1 <= x <= 2:
        return cdf(1.0, a) + a * (0.5 - 0.5 * (2 - x) ** 2)
    else:
        return cdf(2.0, a)


# Prepare data for plotting
x_vals = np.linspace(-1, 3, 500)
pdf_vals = [pdf(x, a) for x in x_vals]
cdf_vals = [cdf(x, a) for x in x_vals]

# Plot PDF
fig_pdf = go.Figure()
fig_pdf.add_trace(go.Scatter(x=x_vals, y=pdf_vals, mode="lines", name="PDF"))
fig_pdf.update_layout(
    title="Funkcja gęstości prawdopodobieństwa (PDF)",
    xaxis_title="x",
    yaxis_title="f(x)",
    showlegend=True,
)

# Plot CDF
fig_cdf = go.Figure()
fig_cdf.add_trace(go.Scatter(x=x_vals, y=cdf_vals, mode="lines", name="CDF"))
fig_cdf.update_layout(
    title="Dystrybuanta (CDF)",
    xaxis_title="x",
    yaxis_title="F(x)",
    showlegend=True,
)

# Display results
st.plotly_chart(fig_pdf, use_container_width=True)
st.plotly_chart(fig_cdf, use_container_width=True)

# Add a button to reveal the correct value of `valid_a`
if st.button("Pokaż poprawną wartość \( a \)"):
    st.write(f"Poprawna wartość \( a \) wynosi: {valid_a:.4f}")
