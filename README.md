# 📈 Stock Price Predictor

A machine learning-based tool that predicts the **next day's closing stock price** using historical market data. This project is being developed into a **Streamlit web app** where users can input a stock ticker and view the predicted closing price for the following trading day.

---

## 🔍 Overview

This project explores the application of time series forecasting models for predicting stock prices. It fetches historical stock data, processes it, and uses deep learning models (such as LSTM) to make predictions.

The goal is to build an accessible, interactive web app that can assist users in visualizing and understanding market trends through predictive analytics.

---

## 🚀 Features

- 📊 Fetches and visualizes historical stock data
- 🧠 Trains an LSTM neural network for next-day price prediction
- 🔄 Scales and preprocesses time-series data for training
- 🧪 Evaluates model performance using standard metrics
- 🧵 Future: Streamlit app for real-time predictions

---

## 🛠 Tech Stack

- Python 3
- NumPy, Pandas, Matplotlib, Scikit-learn
- TensorFlow / Keras
- [yfinance](https://pypi.org/project/yfinance/) for stock data
- (Coming soon) Streamlit for the user interface

---

## 📁 Folder Structure

```
📦 Stock-Price-Predictor/
├── 📄 stock_predictor.py     # Main script for model training and prediction
├── 📄 utils.py               # Helper functions (data loading, scaling, etc.)
├── 📄 model.h5               # Saved trained model
├── 📄 requirements.txt       # Python dependencies
├── 📄 README.md              # Project documentation
```

---

## 📈 How It Works

1. **Data Collection**\
   Fetches historical stock price data using `yfinance`.

2. **Preprocessing**\
   Scales and reshapes data for model compatibility using tools like `MinMaxScaler`.

3. **Model Training**\
   Uses an LSTM (Long Short-Term Memory) network to learn time-dependent patterns in the data.

4. **Prediction**\
   Outputs the predicted **next closing price** based on the most recent input window.

---

## 🔮 Example Output

```
Ticker: AAPL  
Predicted Next Close: $175.42  
Actual Last Close: $173.50  
```

---

## ▶️ Running the Project

1. **Clone the repo**

   ```bash
   git clone https://github.com/Daiyan-Zubaier/Stock-Price-Predictor.git
   cd Stock-Price-Predictor
   ```

2. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

3. **Run the script**

   ```bash
   python stock_predictor.py
   ```

---

## 🌐 Coming Soon: Streamlit Web App

This tool will soon be available as a user-friendly Streamlit app where:

- Users can input a stock ticker
- View the latest prediction
- Visualize past trends and confidence intervals

---

## 🤝 Contributing

Got ideas or feedback? Open an issue or submit a pull request!

---
