import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="Trader Dashboard", layout="wide")

rf_model = joblib.load("random_forest_model_2.pkl")
encoder = joblib.load("label_encoder.pkl")
kmeans_model = joblib.load("kmeans_model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("Trader Performance Dashboard")

st.sidebar.header("Input Features")

size_usd = st.sidebar.number_input(
    "Trade Size USD",
    value=1000.0
)

fee = st.sidebar.number_input(
    "Fee",
    value=10.0
)

sentiment = st.sidebar.selectbox(
    "Market Sentiment",
    ["Fear", "Extreme Fear", "Neutral", "Greed"]
)

predict_button = st.sidebar.button("Predict")

if predict_button:

    sentiment_encoded = encoder.transform([sentiment])[0]

    input_df = pd.DataFrame({
        "Size USD": [size_usd],
        "Fee": [fee],
        "classification_encoded": [sentiment_encoded]
    })

    prediction = rf_model.predict(input_df)[0]

    prediction_prob = rf_model.predict_proba(input_df)[0]

    st.header("Random Forest Prediction")

    if prediction == 1:
        st.success("Profitable Trade")
    else:
        st.error("Non-Profitable Trade")

    st.write("Prediction Probability")
    st.write(prediction_prob)

    importance_df = pd.DataFrame({
        "Feature": input_df.columns,
        "Importance": rf_model.feature_importances_
    })

    fig1, ax1 = plt.subplots(figsize=(8,5))

    sns.barplot(
        x="Importance",
        y="Feature",
        data=importance_df,
        ax=ax1
    )

    st.pyplot(fig1)

    st.header("KMeans Trader Clustering")

    cluster_input = pd.DataFrame({
        "Size USD": [size_usd],
        "Closed PnL": [100],
        "Fee": [fee]
    })

    scaled_input = scaler.transform(cluster_input)

    cluster = kmeans_model.predict(scaled_input)[0]

    cluster_names = {
        0: "Conservative Trader",
        1: "Aggressive Trader",
        2: "Moderate Trader"
    }

    st.info(f"Cluster: {cluster_names[cluster]}")

    fig2, ax2 = plt.subplots(figsize=(7,5))

    cluster_data = pd.DataFrame({
        "Cluster": ["Conservative", "Aggressive", "Moderate"],
        "Value": [1, 1, 1]
    })

    sns.barplot(
        x="Cluster",
        y="Value",
        data=cluster_data,
        ax=ax2
    )

    st.pyplot(fig2)
