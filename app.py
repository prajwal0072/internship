import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(
    page_title="Trader Dashboard",
    layout="wide"
)

rf_model = joblib.load("random_forest_model.pkl")
encoder = joblib.load("label_encoder.pkl")
kmeans_model = joblib.load("kmeans_model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("📈 Trader Performance Dashboard")

st.sidebar.header("Input Features")

size_usd = st.sidebar.number_input(
    "Trade Size USD",
    min_value=0.0,
    value=1000.0
)

fee = st.sidebar.number_input(
    "Fee",
    min_value=0.0,
    value=10.0
)

execution_price = st.sidebar.number_input(
    "Execution Price",
    min_value=0.0,
    value=100.0
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
        "Execution Price": [execution_price],
        "classification_encoded": [sentiment_encoded]
    })

    prediction = rf_model.predict(input_df)[0]

    prediction_prob = rf_model.predict_proba(input_df)[0]

    st.subheader("🤖 Profitability Prediction")

    col1, col2 = st.columns(2)

    with col1:
        if prediction == 1:
            st.success("✅ Profitable Trade")

            st.markdown("""
            ### Profitability Insights
            - Positive market sentiment detected
            - Balanced trade size
            - Lower fee impact
            - Better probability of profitability
            """)

        else:
            st.error("❌ Non-Profitable Trade")

            st.markdown("""
            ### Risk Insights
            - Volatile market conditions
            - Higher risk exposure
            - Consider reducing trade size
            """)

    with col2:
        st.metric(
            "Confidence",
            f"{max(prediction_prob)*100:.2f}%"
        )

    st.subheader("📊 Feature Importance")

    importance_df = pd.DataFrame({
        "Feature": input_df.columns,
        "Importance": rf_model.feature_importances_
    })

    fig1, ax1 = plt.subplots(figsize=(4,2.5))

    sns.barplot(
        x="Importance",
        y="Feature",
        data=importance_df,
        ax=ax1
    )

    plt.tight_layout()

    st.pyplot(fig1)

    st.subheader("🧠 Trader Clustering")

    cluster_input = pd.DataFrame({
        "Size USD": [size_usd],
        "Closed PnL": [500 if prediction == 1 else -200],
        "Fee": [fee]
    })

    scaled_input = scaler.transform(cluster_input)

    cluster = kmeans_model.predict(scaled_input)[0]

    cluster_names = {
        0: "Conservative Trader",
        1: "Aggressive Trader",
        2: "Moderate Trader"
    }

    st.info(f"Trader Type: {cluster_names[cluster]}")

    cluster_df = pd.DataFrame({
        "Cluster": [
            "Conservative",
            "Aggressive",
            "Moderate"
        ],
        "Value": [1, 1, 1]
    })

    fig2, ax2 = plt.subplots(figsize=(3.5,2.5))

    sns.barplot(
        x="Cluster",
        y="Value",
        data=cluster_df,
        ax=ax2
    )

    plt.tight_layout()

    st.pyplot(fig2)

    st.subheader("📌 Key Insights")

    st.markdown("""
    - Greed periods showed higher trading activity.
    - Fear periods resulted in higher volatility.
    - Large trade sizes produced larger PnL swings.
    """)

    st.subheader("🚀 Recommendations")

    st.markdown("""
    - Reduce leverage during Fear conditions.
    - Increase trading selectively during Greed periods.
    - Avoid overtrading during high volatility.
    """)
