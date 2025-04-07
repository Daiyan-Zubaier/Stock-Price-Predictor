# Stock Price Predictor

This project predicts the next day's stock price using historical data and a Long Short-Term Memory (LSTM) neural network. It is built with PyTorch, Streamlit, and the Alpha Vantage API to fetch stock market data.

In Progress: Currently working on migrating API to yfinance instead of AlphaVantage. 
---

## 🚀 Features

- Fetches daily stock price data using the Alpha Vantage API.
- Preprocesses data with normalization and windowing.
- Trains an LSTM model to predict stock prices.
- Visualizes actual vs. predicted stock prices.
- Displays predicted stock price for the next day.

---

## 📦 Installation

1. Clone the repository:

```bash
git clone https://github.com/your-username/Stock-Price-Predictor.git
cd Stock-Price-Predictor
```

2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

3. Obtain an API key from [Alpha Vantage](https://www.alphavantage.co/support/#api-key) and replace the placeholder value in the config dictionary in `app.py`:

```python
"key": "YOUR_ALPHA_VANTAGE_API_KEY"
```

---

## ▶️ Usage

Run the Streamlit app:

```bash
streamlit run app.py
```

Then:
- Enter the stock symbol (e.g., IBM) in the input field.
- View the predicted stock price for the next day.
- Explore visualizations comparing actual vs. predicted prices.

---

## 🗂 Project Structure

- `app.py` — Main application script for the Streamlit UI
- `Stock_Predictor_Daiyan_&_Rajit.ipynb` — Jupyter Notebook used to train and test the LSTM model
- `requirements.txt` — Dependency list

---

## 🧠 Model Details

- Input size: `1`
- LSTM layers: `2`
- Hidden layer size: `32`
- Dropout: `0.2`
- Loss function: `Mean Squared Error (MSE)`
- Optimizer: `Adam`

---

## 🧹 Data Preprocessing

- **Normalization**: Data is normalized using the mean and standard deviation.
- **Windowing**: A sliding window approach creates sequences for LSTM input.

---

## 📈 Visualization

The app provides the following charts:
- Actual vs. predicted stock prices (training + validation)
- Zoomed-in view of predictions on validation data

---

## 📋 Requirements

- Python 3.7+
- See `requirements.txt` for all packages

---


