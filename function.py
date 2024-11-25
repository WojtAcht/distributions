import streamlit as st
import numpy as np
import plotly.express as px

# Introduction
st.title("Funkcja zmiennej losowej")
st.markdown(
    """
Ta aplikacja demonstruje, jak funkcja zmiennej losowej wpływa na jej rozkład.
Rozważamy **rozkład wykładniczy** z funkcją gęstości prawdopodobieństwa (PDF):

$$f_X(x; \\lambda) = \\lambda e^{-\\lambda x}, \\quad x \\geq 0, \\lambda > 0$$

Funkcja zmiennej losowej definiowana jest jako:

$$Y = X^\\alpha, \\quad Y = \\max(X, 1), \\quad Y = \\min(X, 1)$$

gdzie α to parametr określony przez użytkownika. Możesz również dostosować λ, parametr skali rozkładu wykładniczego.
"""
)

# User Inputs
st.sidebar.header("Parametry wejściowe")
lambda_param = st.sidebar.slider(
    "Parametr skali (λ)", min_value=0.1, max_value=5.0, value=1.0, step=0.1
)
alpha = st.sidebar.slider(
    "Wykładnik transformacji (α)", min_value=0.1, max_value=5.0, value=0.5, step=0.1
)

# Generate random samples
np.random.seed(42)  # For reproducibility
original_samples = np.random.exponential(scale=1 / lambda_param, size=10000)
transformed_samples_alpha = original_samples**alpha
transformed_samples_max = np.maximum(original_samples, 1)
transformed_samples_min = np.minimum(original_samples, 1)

# Plot the original and transformed distributions
fig_original = px.histogram(
    original_samples,
    nbins=50,
    histnorm="density",
    title="Oryginalny rozkład wykładniczy",
    labels={"value": "Wartość", "density": "Gęstość"},
)
fig_original.update_layout(xaxis_title="Wartość", yaxis_title="Gęstość")

fig_transformed_alpha = px.histogram(
    transformed_samples_alpha,
    nbins=50,
    histnorm="density",
    title=f"Przekształcony rozkład (X^{alpha})",
    labels={"value": "Wartość", "density": "Gęstość"},
)
fig_transformed_alpha.update_layout(xaxis_title="Wartość", yaxis_title="Gęstość")

fig_transformed_max = px.histogram(
    transformed_samples_max,
    nbins=50,
    histnorm="density",
    title="Przekształcony rozkład (max(X, 1))",
    labels={"value": "Wartość", "density": "Gęstość"},
)
fig_transformed_max.update_layout(xaxis_title="Wartość", yaxis_title="Gęstość")

fig_transformed_min = px.histogram(
    transformed_samples_min,
    nbins=50,
    histnorm="density",
    title="Przekształcony rozkład (min(X, 1))",
    labels={"value": "Wartość", "density": "Gęstość"},
)
fig_transformed_min.update_layout(xaxis_title="Wartość", yaxis_title="Gęstość")

# Display the plots
st.plotly_chart(fig_original, use_container_width=True)
st.plotly_chart(fig_transformed_alpha, use_container_width=True)
st.plotly_chart(fig_transformed_max, use_container_width=True)
st.plotly_chart(fig_transformed_min, use_container_width=True)

# Statistical properties
st.header("Właściwości statystyczne")
st.write("### Oryginalny rozkład:")
st.write(f"Średnia: {np.mean(original_samples):.4f}")
st.write(f"Wariancja: {np.var(original_samples):.4f}")

st.write("### Przekształcony rozkład (X^α):")
st.write(f"Średnia: {np.mean(transformed_samples_alpha):.4f}")
st.write(f"Wariancja: {np.var(transformed_samples_alpha):.4f}")

st.write("### Przekształcony rozkład (max(X, 1)):")
st.write(f"Średnia: {np.mean(transformed_samples_max):.4f}")
st.write(f"Wariancja: {np.var(transformed_samples_max):.4f}")

st.write("### Przekształcony rozkład (min(X, 1)):")
st.write(f"Średnia: {np.mean(transformed_samples_min):.4f}")
st.write(f"Wariancja: {np.var(transformed_samples_min):.4f}")

st.markdown(
    """
Eksperymentuj z wartościami λ i α w menu po lewej stronie, aby zobaczyć, jak przekształcenie wpływa na rozkład!
"""
)
