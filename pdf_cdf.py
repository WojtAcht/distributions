import streamlit as st
import numpy as np
import plotly.graph_objects as go

# Introduction
st.title("Gęstość i dystrybuanta")
st.markdown(
    """
Ta aplikacja abstrakcyjnie wyjaśnia idee funkcji gęstości prawdopodobieństwa (PDF) oraz dystrybuanty (CDF).
Interaktywnie zobacz, jak PDF i CDF są ze sobą powiązane:
- PDF: Gęstość prawdopodobieństwa, pokazująca jak prawdopodobieństwo jest "skoncentrowane".
- CDF: Łączne prawdopodobieństwo, że wartość zmiennej losowej jest mniejsza lub równa \( x \).
"""
)

# Generate a sample PDF (Gaussian-like)
x = np.linspace(-5, 5, 500)
pdf = np.exp(-(x**2) / 2) / np.sqrt(2 * np.pi)
cdf = np.cumsum(pdf) / np.cumsum(pdf)[-1]  # Normalize CDF to range [0, 1]

# User interaction: Choose a value of x
x_value = st.slider(
    "Wybierz wartość x:",
    min_value=float(x[0]),
    max_value=float(x[-1]),
    value=0.0,
    step=0.1,
)

# Plot PDF and highlight the area up to x_value
fig_pdf = go.Figure()
fig_pdf.add_trace(go.Scatter(x=x, y=pdf, mode="lines", name="PDF", line=dict(width=2)))
fig_pdf.add_trace(
    go.Scatter(
        x=x[x <= x_value],
        y=pdf[x <= x_value],
        fill="tozeroy",
        mode="none",
        name="Pole pod krzywą (do x)",
        fillcolor="rgba(255, 100, 100, 0.5)",
    )
)
fig_pdf.update_layout(
    title="Funkcja gęstości prawdopodobieństwa (PDF)",
    xaxis_title="x",
    yaxis_title="gęstość prawdopodobieństwa",
    showlegend=True,
)

# Plot CDF and mark the corresponding point
fig_cdf = go.Figure()
fig_cdf.add_trace(go.Scatter(x=x, y=cdf, mode="lines", name="CDF", line=dict(width=2)))
fig_cdf.add_trace(
    go.Scatter(
        x=[x_value],
        y=[cdf[np.abs(x - x_value).argmin()]],
        mode="markers",
        marker=dict(size=10, color="red"),
        name=f"f(x)",
    )
)
fig_cdf.update_layout(
    title="Dystrybuanta (CDF)",
    xaxis_title="x",
    yaxis_title="F(x) = P(X ≤ x)",
    showlegend=True,
)

# Display plots
st.plotly_chart(fig_pdf, use_container_width=True)
st.plotly_chart(fig_cdf, use_container_width=True)

# Explanation
st.markdown(
    """
### Wyjaśnienie:
- **Funkcja gęstości prawdopodobieństwa (PDF):**
  - Gęstość pokazuje, jak "skoncentrowane" jest prawdopodobieństwo w różnych punktach.
  - Pole pod krzywą PDF w przedziale reprezentuje prawdopodobieństwo zmiennej losowej w tym przedziale.

- **Dystrybuanta (CDF):**
  - Wartość dystrybuanty dla \( x \) to pole pod gęstością od -∞ do tego punktu.
  - Dystrybuanta jest niemalejąca i mieści się w przedziale [0, 1].

### Interakcja:
- Wybierz \( x \), aby zobaczyć, jak pole pod gęstością do \( x \) odpowiada wartości dystrybuanty.
"""
)
