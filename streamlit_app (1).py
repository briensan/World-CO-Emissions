import streamlit as st
import numpy as np
import joblib
import matplotlib.pyplot as plt
import pandas as pd

# ---------------- LOAD ----------------
model = joblib.load("rf_model.pkl")
scaler = joblib.load("scaler.pkl")
df = pd.read_csv("final_model_ready_dataset.csv")

# ---------------- CSS ----------------
st.markdown("""
<style>
.stApp {background-color: #f5f7fa;}
h1 {text-align: center; color: #2c3e50;}
.stButton > button {
    background-color: #4CAF50;
    color: white;
    border-radius: 8px;
    height: 3em;
    width: 100%;
}
</style>
""", unsafe_allow_html=True)

# ---------------- TITLE ----------------
st.title("🌍 CO2 Emission Dashboard")
st.markdown("---")

# ---------------- SIDEBAR ----------------
st.sidebar.header("🔧 Input Features")

year = st.sidebar.slider("Year (Historical)", 1990, 2020, 2020)
renewable = st.sidebar.slider("Renewable Energy (%)", 0.0, 100.0, 20.0)
carbon = st.sidebar.number_input("Carbon Intensity", value=50.0, key="c1")
gdp = st.sidebar.number_input("CO2 per GDP (PPP)", value=20.0, key="g1")
methane = st.sidebar.number_input("Methane_CO2e", value=10.0, key="m1")
n2o = st.sidebar.number_input("N2O_CO2e", value=5.0, key="n1")

# ---------------- TREND FUNCTION ----------------
def get_trend(feature):
    x = pd.to_numeric(df["year"], errors='coerce')
    y = pd.to_numeric(df[feature], errors='coerce')

    # remove missing values
    temp = pd.DataFrame({"x": x, "y": y}).dropna()

    slope = np.polyfit(temp["x"], temp["y"], 1)[0]
    return slope
# ---------------- CURRENT ----------------
st.subheader("🌱 Current Prediction")

if st.sidebar.button("Predict Current"):
    input_data = np.array([[year, renewable, carbon, gdp, methane, n2o]])
    pred_current = model.predict(scaler.transform(input_data))

    st.success(f"Current CO2 Emission: {pred_current[0]:.2f}")

    # Feature graph
    fig1, ax1 = plt.subplots()
    features = ["Renewable", "Carbon", "GDP", "Methane", "N2O"]
    values = [renewable, carbon, gdp, methane, n2o]
    ax1.bar(features, values)
    ax1.set_title("Feature Impact")
    st.pyplot(fig1)

# ---------------- FUTURE ----------------
st.subheader("🚀 Future Prediction")

future_year = st.slider("Future Year", 2021, 2050, 2035)

if st.button("Predict Future"):

    years_ahead = future_year - year

    # trends from data
    r_trend = get_trend("Renewable_Consumption_Pct")
    c_trend = get_trend("CO2_Intensity")
    g_trend = get_trend("CO2_per_GDP_PPP")
    m_trend = get_trend("Methane_CO2e")
    n_trend = get_trend("N2O_CO2e")

    future_input = np.array([[
        future_year,
        renewable + r_trend * years_ahead,
        carbon + c_trend * years_ahead,
        gdp + g_trend * years_ahead,
        methane + m_trend * years_ahead,
        n2o + n_trend * years_ahead
    ]])

    pred_future = model.predict(scaler.transform(future_input))

    st.success(f"Future CO2 ({future_year}): {pred_future[0]:.2f}")

    # Trend graph
    years = list(range(year, future_year + 1, 5))
    preds = []

    for y in years:
        ya = y - year
        temp = np.array([[
            y,
            renewable + r_trend * ya,
            carbon + c_trend * ya,
            gdp + g_trend * ya,
            methane + m_trend * ya,
            n2o + n_trend * ya
        ]])
        p = model.predict(scaler.transform(temp))
        preds.append(p[0])

    fig2, ax2 = plt.subplots()
    ax2.plot(years, preds, marker='o')
    ax2.set_title("Future CO2 Trend")
    ax2.set_xlabel("Year")
    ax2.set_ylabel("CO2")
    st.pyplot(fig2)

# ---------------- FOOTER ----------------
st.markdown("---")
st.write("📊 Data-based future prediction using historical trends.")