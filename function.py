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

$$Y = X^\\alpha$$

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
transformed_samples = original_samples**alpha

# Plot the original and transformed distributions
fig_original = px.histogram(
    original_samples,
    nbins=50,
    histnorm="density",
    title="Oryginalny rozkład wykładniczy",
    labels={"value": "Wartość", "density": "Gęstość"},
)
fig_original.update_layout(xaxis_title="Wartość", yaxis_title="Gęstość")

fig_transformed = px.histogram(
    transformed_samples,
    nbins=50,
    histnorm="density",
    title=f"Przekształcony rozkład (X^{alpha})",
    labels={"value": "Wartość", "density": "Gęstość"},
)
fig_transformed.update_layout(xaxis_title="Wartość", yaxis_title="Gęstość")

# Display the plots
st.plotly_chart(fig_original, use_container_width=True)
st.plotly_chart(fig_transformed, use_container_width=True)

# Statistical properties
st.header("Właściwości statystyczne")
st.write("### Oryginalny rozkład:")
st.write(f"Średnia: {np.mean(original_samples):.4f}")
st.write(f"Wariancja: {np.var(original_samples):.4f}")

st.write("### Przekształcony rozkład:")
st.write(f"Średnia: {np.mean(transformed_samples):.4f}")
st.write(f"Wariancja: {np.var(transformed_samples):.4f}")

st.markdown(
    """
Eksperymentuj z wartościami λ i α w menu po lewej stronie, aby zobaczyć, jak przekształcenie wpływa na rozkład!
"""
)
