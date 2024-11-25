import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px


def calculate_distribution(distribution, params):
    theoretical_data = []
    expected_value = 0
    variance = 0

    if distribution == "poisson":
        lambd = params["lambda"]
        expected_value = lambd
        variance = lambd

        for k in range(0, max(10, int(lambd * 2))):
            prob = (lambd**k * np.exp(-lambd)) / np.math.factorial(k)
            cumulative_prob = sum(
                (lambd**i * np.exp(-lambd)) / np.math.factorial(i) for i in range(k + 1)
            )
            theoretical_data.append(
                {"x": k, "probability": prob, "cumulative": cumulative_prob}
            )

    elif distribution == "binomial":
        n, p = params["n"], params["p"]
        expected_value = n * p
        variance = n * p * (1 - p)

        for k in range(n + 1):
            prob = np.math.comb(n, k) * (p**k) * ((1 - p) ** (n - k))
            cumulative_prob = sum(
                np.math.comb(n, i) * (p**i) * ((1 - p) ** (n - i)) for i in range(k + 1)
            )
            theoretical_data.append(
                {"x": k, "probability": prob, "cumulative": cumulative_prob}
            )

    elif distribution == "geometric":
        p = params["p"]
        expected_value = 1 / p
        variance = (1 - p) / (p**2)

        for k in range(1, int(10 / p)):
            prob = p * ((1 - p) ** (k - 1))
            cumulative_prob = sum(p * ((1 - p) ** (i - 1)) for i in range(1, k + 1))
            theoretical_data.append(
                {"x": k, "probability": prob, "cumulative": cumulative_prob}
            )

    elif distribution == "exponential":
        rate = params["rate"]
        expected_value = 1 / rate
        variance = 1 / (rate**2)

        x_values = np.linspace(0, 10 / rate, 500)
        for x in x_values:
            density = rate * np.exp(-rate * x)
            cumulative_prob = 1 - np.exp(-rate * x)
            theoretical_data.append(
                {"x": x, "density": density, "cumulative": cumulative_prob}
            )

    elif distribution == "uniform":
        a, b = params["a"], params["b"]
        expected_value = (a + b) / 2
        variance = ((b - a) ** 2) / 12

        x_values = np.linspace(a, b, 500)
        for x in x_values:
            density = 1 / (b - a)
            cumulative_prob = (x - a) / (b - a)
            theoretical_data.append(
                {"x": x, "density": density, "cumulative": cumulative_prob}
            )

    elif distribution == "normal":
        mu, sigma = params["mu"], params["sigma"]
        expected_value = mu
        variance = sigma**2

        x_values = np.linspace(mu - 4 * sigma, mu + 4 * sigma, 500)
        for x in x_values:
            density = (1 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(
                -((x - mu) ** 2) / (2 * sigma**2)
            )
            cumulative_prob = (1 + np.math.erf((x - mu) / (np.sqrt(2) * sigma))) / 2
            theoretical_data.append(
                {"x": x, "density": density, "cumulative": cumulative_prob}
            )

    return pd.DataFrame(theoretical_data), expected_value, variance


st.title("Eksplorator Rozkładów Prawdopodobieństwa")

distribution = st.selectbox(
    "Wybierz Rozkład",
    options=[
        "poisson",
        "binomial",
        "geometric",
        "exponential",
        "uniform",
        "normal",
    ],
    index=0,
)

st.subheader("Wzór Matematyczny")
if distribution == "poisson":
    st.latex(r"P(X = k) = \frac{\lambda^k e^{-\lambda}}{k!}")
    lambda_param = st.number_input("λ (Wskaźnik)", min_value=0.1, value=2.0, step=0.1)
    params = {"lambda": lambda_param}

elif distribution == "binomial":
    st.latex(r"P(X = k) = \binom{n}{k} p^k (1-p)^{n-k}")
    n = st.number_input("n (Liczba prób)", min_value=1, value=10, step=1)
    p = st.number_input(
        "p (Prawdopodobieństwo sukcesu)",
        min_value=0.01,
        max_value=1.0,
        value=0.5,
        step=0.01,
    )
    params = {"n": int(n), "p": p}

elif distribution == "geometric":
    st.latex(r"P(X = k) = p (1-p)^{k-1}")
    p = st.number_input(
        "p (Prawdopodobieństwo sukcesu)",
        min_value=0.01,
        max_value=1.0,
        value=0.5,
        step=0.01,
    )
    params = {"p": p}

elif distribution == "exponential":
    st.latex(r"f(x) = \lambda e^{-\lambda x}")
    rate = st.number_input(
        "λ (Wskaźnik intensywności)", min_value=0.1, value=1.0, step=0.1
    )
    params = {"rate": rate}

elif distribution == "uniform":
    st.latex(r"f(x) = \frac{1}{b-a} \text{ dla } a \leq x \leq b")
    a = st.number_input("a (Dolna granica)", value=0.0, step=0.1)
    b = st.number_input("b (Górna granica)", value=1.0, step=0.1)
    if b <= a:
        st.error("b musi być większe od a!")
    params = {"a": a, "b": b}

elif distribution == "normal":
    st.latex(r"f(x) = \frac{1}{\sqrt{2\pi\sigma^2}} e^{-\frac{(x-\mu)^2}{2\sigma^2}}")
    mu = st.number_input("μ (Średnia)", value=0.0, step=0.1)
    sigma = st.number_input(
        "σ (Odchylenie standardowe)", min_value=0.1, value=1.0, step=0.1
    )
    params = {"mu": mu, "sigma": sigma}

# Generate theoretical data
theoretical_data, expected_value, variance = calculate_distribution(
    distribution, params
)

# Display distribution properties
st.subheader("Właściwości Rozkładu")
st.write(f"**Wartość Oczekiwana:** {expected_value:.2f}")
st.write(f"**Wariancja:** {variance:.2f}")

# Plot PDF or PMF
if distribution in ["exponential", "uniform", "normal"]:
    st.subheader("Funkcja Gęstości Prawdopodobieństwa (PDF)")
    pdf_chart = px.line(
        theoretical_data,
        x="x",
        y="density",
        title="Funkcja Gęstości Prawdopodobieństwa",
        labels={"x": "x", "density": "Gęstość"},
    )
    st.plotly_chart(pdf_chart)
else:
    st.subheader("Funkcja Prawdopodobieństwa (PMF)")
    pmf_chart = px.scatter(
        theoretical_data,
        x="x",
        y="probability",
        title="Funkcja Prawdopodobieństwa",
        labels={"x": "x", "probability": "Prawdopodobieństwo"},
    )
    st.plotly_chart(pmf_chart)

# Plot CDF
st.subheader("Dystrybuanta (CDF)")
cdf_chart = px.line(
    theoretical_data,
    x="x",
    y="cumulative",
    title="Dystrybuanta",
    labels={"x": "x", "cumulative": "Dystrybuanta"},
    line_shape=(
        "hv" if distribution not in ["exponential", "uniform", "normal"] else None
    ),
)
st.plotly_chart(cdf_chart)
