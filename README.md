internship/
│
├── app.py
├── analyser.ipynb
├── requirements.txt
├── README.md
│
├── data/
│   ├── fear_greed_index.csv
│   └── historical_data.csv
│
├── models/
│   ├── random_forest_model.pkl
│   ├── label_encoder.pkl
│   ├── kmeans_model.pkl
│   └── scaler.pkl
│
└── screenshots/
    ├── dashboard.png
    ├── profitable_trade.png
    └── clustering.png


# 📈 AI-Powered Trader Sentiment Analysis Dashboard

## Overview
This project analyzes trader behavior using market sentiment data (Fear & Greed Index) and trading activity.

The dashboard predicts whether a trade is likely to be profitable using Machine Learning and also classifies traders into behavioral clusters.

---

## Features

- Random Forest profitability prediction
- KMeans trader clustering
- Interactive Streamlit dashboard
- Sentiment-based analysis
- Feature importance visualization
- Risk insights & recommendations

---

## Technologies Used

- Python
- Pandas
- Scikit-learn
- Streamlit
- Matplotlib
- Seaborn

---

## Machine Learning Models

### Random Forest Classifier
Used for:
- Predicting profitable vs non-profitable trades

Features:
- Trade Size USD
- Fee
- Execution Price
- Market Sentiment

### KMeans Clustering
Used for:
- Grouping traders into:
  - Conservative
  - Aggressive
  - Moderate



Live App:
https://prajwal-internship095899.streamlit.app/

---

## Author

Prajwal Pai
