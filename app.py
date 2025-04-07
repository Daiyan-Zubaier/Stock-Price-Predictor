import numpy as np
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from alpha_vantage.timeseries import TimeSeries
import streamlit as st
import requests
import pandas as pd
import logging
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

logging.basicConfig(level=logging.DEBUG)

# App title and description
st.title("📈 Next Day Stock Price Predictor")
st.markdown("""
Welcome to the **Stock Price Predictor**!  
This app uses a pre-trained LSTM model to predict the next day's stock price based on historical data.  
Simply enter the stock symbol below and let the magic happen! 🚀
""")

# User input for stock symbol
user_input = st.text_input("Enter the stock symbol (e.g., IBM, AAPL, TSLA):", "IBM")

config = {
    "alpha_vantage": {
        "key": os.getenv("ALPHA_VANTAGE_API_KEY"),
        "symbol": user_input,
        "outputsize": "full",
        "key_adjusted_close": "4. close",
    },
    "data": {
        "window_size": 20,
        "train_split_size": 0.80,
    },
    "plots": {
        "xticks_interval": 90,
        "color_actual": "#001f3f",
        "color_train": "#3D9970",
        "color_val": "#0074D9",
        "color_pred_train": "#3D9970",
        "color_pred_val": "#0074D9",
        "color_pred_test": "#FF4136",
    },
    "model": {
        "input_size": 1,
        "num_lstm_layers": 2,
        "lstm_size": 32,
        "dropout": 0.2,
    },
    "training": {
        "device": "cpu",
        "batch_size": 64,
        "num_epoch": 100,
        "learning_rate": 0.01,
        "scheduler_step_size": 40,
    }
}

def download_data(config):
    if not config["alpha_vantage"]["key"]:
        st.error("API key is missing. Please set your Alpha Vantage API key in the environment variable 'ALPHA_VANTAGE_API_KEY'.")
        st.stop()

    url = 'https://www.alphavantage.co/query'
    params = {
        'function': 'TIME_SERIES_DAILY',
        'symbol': config["alpha_vantage"]["symbol"],
        'outputsize': config["alpha_vantage"]["outputsize"],
        'apikey': config["alpha_vantage"]["key"]
    }
    response = requests.get(url, params=params)
    data = response.json()
    if "Error Message" in data:
        raise ValueError(data["Error Message"])
    data_date = list(data['Time Series (Daily)'].keys())
    data_date.reverse()
    data_close_price = [float(data['Time Series (Daily)'][date]['4. close']) for date in data_date]
    return data_date, np.array(data_close_price)

try:
    data_date, data_close_price = download_data(config)
    st.subheader(f"📊 Historical Stock Prices for {user_input.upper()}")
    st.line_chart(pd.DataFrame({"Date": data_date, "Close Price": data_close_price}).set_index("Date"))
except ValueError as e:
    st.error(f"Error fetching data: {e}")
    st.stop()

class Normalizer:
    def __init__(self):
        self.mu = None
        self.sd = None

    def fit_transform(self, x):
        self.mu = np.mean(x, axis=0, keepdims=True)
        self.sd = np.std(x, axis=0, keepdims=True)
        return (x - self.mu) / self.sd

    def inverse_transform(self, x):
        return (x * self.sd) + self.mu

scaler = Normalizer()
normalized_data_close_price = scaler.fit_transform(data_close_price)

def prepare_data_x(x, window_size):
    n_row = x.shape[0] - window_size + 1
    output = np.lib.stride_tricks.as_strided(x, shape=(n_row, window_size), strides=(x.strides[0], x.strides[0]))
    return output[:-1], output[-1]

def prepare_data_y(x, window_size):
    return x[window_size:]

data_x, data_x_unseen = prepare_data_x(normalized_data_close_price, config["data"]["window_size"])
data_y = prepare_data_y(normalized_data_close_price, config["data"]["window_size"])

split_index = int(data_y.shape[0] * config["data"]["train_split_size"])
data_x_train, data_x_val = data_x[:split_index], data_x[split_index:]
data_y_train, data_y_val = data_y[:split_index], data_y[split_index:]

class StockDataset(Dataset):
    def __init__(self, x, y):
        x = np.expand_dims(x, 2)
        self.x = x.astype(np.float32)
        self.y = y.astype(np.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

train_dataset = StockDataset(data_x_train, data_y_train)
val_dataset = StockDataset(data_x_val, data_y_val)

train_dataloader = DataLoader(train_dataset, batch_size=config["training"]["batch_size"], shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=config["training"]["batch_size"], shuffle=True)

class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=32, num_layers=2, output_size=1, dropout=0.2):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size
        self.linear_1 = nn.Linear(input_size, hidden_layer_size)
        self.relu = nn.ReLU()
        self.lstm = nn.LSTM(hidden_layer_size, hidden_size=self.hidden_layer_size, num_layers=num_layers, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(num_layers * hidden_layer_size, output_size)

    def forward(self, x):
        batchsize = x.shape[0]
        x = self.linear_1(x)
        x = self.relu(x)
        lstm_out, (h_n, c_n) = self.lstm(x)
        x = h_n.permute(1, 0, 2).reshape(batchsize, -1)
        x = self.dropout(x)
        predictions = self.linear_2(x)
        return predictions[:, -1]

model = LSTMModel(
    input_size=config["model"]["input_size"],
    hidden_layer_size=config["model"]["lstm_size"],
    num_layers=config["model"]["num_lstm_layers"],
    output_size=1,
    dropout=config["model"]["dropout"]
).to(config["training"]["device"])

model_path = os.path.join("./stock_models", f"{user_input}.pth")
if not os.path.exists(model_path):
    st.error(f"Model file not found: {model_path}. Please ensure the model file exists for the stock symbol '{user_input}'.")
    st.stop()

try:
    model.load_state_dict(torch.load(model_path))
except Exception as e:
    st.error(f"Failed to load the model file: {model_path}. Error: {e}")
    st.stop()

model.eval()

def predict(dataloader):
    predictions = []
    for x, _ in dataloader:
        x = x.to(config["training"]["device"])
        out = model(x).cpu().detach().numpy()
        predictions.extend(out)
    return np.array(predictions)

predicted_train = predict(DataLoader(train_dataset, batch_size=config["training"]["batch_size"], shuffle=False))
predicted_val = predict(DataLoader(val_dataset, batch_size=config["training"]["batch_size"], shuffle=False))

next_day_price = scaler.inverse_transform(predicted_val[-1].reshape(-1, 1))[0][0]
st.subheader("📅 Predicted Price for the Next Day")
st.write(f"The predicted price for **{user_input.upper()}** is: **${next_day_price:.2f}**")

to_plot_data_y_train_pred = np.zeros(len(data_date))
to_plot_data_y_val_pred = np.zeros(len(data_date))

to_plot_data_y_train_pred[config["data"]["window_size"]:split_index + config["data"]["window_size"]] = scaler.inverse_transform(predicted_train)
to_plot_data_y_val_pred[split_index + config["data"]["window_size"]:] = scaler.inverse_transform(predicted_val)

to_plot_data_y_train_pred = np.where(to_plot_data_y_train_pred == 0, None, to_plot_data_y_train_pred)
to_plot_data_y_val_pred = np.where(to_plot_data_y_val_pred == 0, None, to_plot_data_y_val_pred)

plot_data = pd.DataFrame({
    "Date": data_date,
    "Actual Prices": pd.to_numeric(data_close_price, errors="coerce"),
    "Predicted Prices (Train)": pd.to_numeric(to_plot_data_y_train_pred, errors="coerce"),
    "Predicted Prices (Validation)": pd.to_numeric(to_plot_data_y_val_pred, errors="coerce")
})

plot_data["Date"] = pd.to_datetime(plot_data["Date"], errors="coerce")
plot_data.set_index("Date", inplace=True)
plot_data.dropna(how="all", inplace=True)

st.subheader("📈 Comparison of Actual and Predicted Prices")
st.line_chart(plot_data)

st.markdown("""
---
Thank you for using the Stock Price Predictor!  
Feel free to try different stock symbols and explore the predictions.  
""")