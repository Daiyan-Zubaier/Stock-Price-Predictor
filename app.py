import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

# To fetch stock market data
from alpha_vantage.timeseries import TimeSeries
import streamlit as st

st.title("Next Day Stock Price Predictor")

user_input = st.text_input("Enter the stock symbol", "IBM")


config = {
    #Collects stock price data
    "alpha_vantage": {
        "key": "CX2NAREQLV2VVYIV",
        "symbol": user_input,
        # Or use Compact for last 100 days
        "outputsize": "full",
        "key_adjusted_close": "4. close",
    },
    #Data preprocessing settings
    "data": {
        # number of past days to predict next price
        "window_size": 20,
        # 80% for training, 20% for testing
        "train_split_size": 0.80,
    },
    # Plotting settings
    "plots": {
        # Show data label every x days
        "xticks_interval": 90,
        "color_actual": "#001f3f",
        "color_train": "#3D9970",
        "color_val": "#0074D9",
        "color_pred_train": "#3D9970",
        "color_pred_val": "#0074D9",
        "color_pred_test": "#FF4136",
    },
    # LSTM NN Settings
    "model": {
        # Number of features
        "input_size": 1,
        "num_lstm_layers": 2,
        "lstm_size": 32,
        # Disables neurons
        "dropout": 0.2,
    },
    "training": {
        "device": "cpu",
        "batch_size": 64,
        # Train for x cycles
        "num_epoch": 100,
        "learning_rate": 0.01,
        # Decreasing lr every x epochs
        "scheduler_step_size": 40,
    }
}

import requests

def download_data(config):
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

    # Extract data
    data_date = list(data['Time Series (Daily)'].keys())
    data_date.reverse()

    data_close_price = [float(data['Time Series (Daily)'][date]['4. close']) for date in data['Time Series (Daily)'].keys()]
    data_close_price.reverse()
    data_close_price = np.array(data_close_price)

    num_data_points = len(data_date)        # number of available dates.
    display_date_range = "from " + data_date[0] + " to " + data_date[num_data_points - 1]
    print("Number data points", num_data_points, display_date_range)

    return data_date, data_close_price, num_data_points, display_date_range
'''
data_date: List of dates.
data_close_price: NumPy array of closing prices.
num_data_points: Total number of data points.
display_date_range: A string indicating the range of dates.
'''

data_date, data_close_price, num_data_points, display_date_range = download_data(config)
print(data_close_price)
print(data_close_price)
print(num_data_points)
print(display_date_range)

# plot

fig = figure(figsize=(25, 5), dpi=80)
fig.patch.set_facecolor((1.0, 1.0, 1.0))
plt.plot(data_date, data_close_price, color=config["plots"]["color_actual"])
xticks = [data_date[i] if ((i%config["plots"]["xticks_interval"]==0 and (num_data_points-i) > config["plots"]["xticks_interval"]) or i==num_data_points-1) else None for i in range(num_data_points)] # make x ticks nice
x = np.arange(0,len(xticks))
plt.xticks(x, xticks, rotation='vertical')
plt.title("Daily close price for " + config["alpha_vantage"]["symbol"] + ", " + display_date_range)
plt.grid(visible=True, which='major', axis='y', linestyle='--')
plt.show()

class Normalizer():
    def __init__(self):
        self.mu = None
        self.sd = None

    def fit_transform(self, x):
        self.mu = np.mean(x, axis=(0), keepdims=True)
        self.sd = np.std(x, axis=(0), keepdims=True)
        normalized_x = (x - self.mu)/self.sd
        return normalized_x

    def inverse_transform(self, x):
        return (x*self.sd) + self.mu

# normalize
scaler = Normalizer()
normalized_data_close_price = scaler.fit_transform(data_close_price)

def prepare_data_x(x, window_size):
    # perform windowing
    n_row = x.shape[0] - window_size + 1
    output = np.lib.stride_tricks.as_strided(x, shape=(n_row, window_size), strides=(x.strides[0], x.strides[0]))
    return output[:-1], output[-1]


def prepare_data_y(x, window_size):
    # # perform simple moving average
    # output = np.convolve(x, np.ones(window_size), 'valid') / window_size

    # use the next day as label
    output = x[window_size:]
    return output

data_x, data_x_unseen = prepare_data_x(normalized_data_close_price, window_size=config["data"]["window_size"])
data_y = prepare_data_y(normalized_data_close_price, window_size=config["data"]["window_size"])

# split dataset

split_index = int(data_y.shape[0]*config["data"]["train_split_size"])
data_x_train = data_x[:split_index]
data_x_val = data_x[split_index:]
data_y_train = data_y[:split_index]
data_y_val = data_y[split_index:]

class StockDataset(Dataset):
    def __init__(self, x, y):
        # expanding dimension of array, currently (batch, window), after (batch, window, features)
        x = np.expand_dims(x, 2)
        self.x = x.astype(np.float32)
        self.y = y.astype(np.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

print("data_x_train", data_x_train.shape)
print("data_x_val", data_x_val.shape)
train_dataset = StockDataset(data_x_train, data_y_train)
val_dataset = StockDataset(data_x_val, data_y_val)

print("Train data shape", train_dataset.x.shape, train_dataset.y.shape)
print("Validation data shape", val_dataset.x.shape, val_dataset.y.shape)

train_dataloader = DataLoader(train_dataset, batch_size=config["training"]["batch_size"], shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=config["training"]["batch_size"], shuffle=True)

class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=32, num_layers=2, output_size=1, dropout=0.2):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size

        self.linear_1 = nn.Linear(input_size, hidden_layer_size)  # Projects input shape into hidden layer shape
        self.relu = nn.ReLU()     # Activation function

        # Defining LSTM model
        self.lstm = nn.LSTM(hidden_layer_size, hidden_size=self.hidden_layer_size, num_layers=num_layers, batch_first=True)

        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(num_layers*hidden_layer_size, output_size)  # Outputs back into input shape


    def forward(self, x):
        batchsize = x.shape[0]

        # layer 1
        x = self.linear_1(x)
        x = self.relu(x)

        # LSTM layer
        lstm_out, (h_n, c_n) = self.lstm(x)

        # reshape output from hidden cell into [batch, features] for linear_2
        x = h_n.permute(1, 0, 2).reshape(batchsize, -1)

        # layer 2
        x = self.dropout(x)
        predictions = self.linear_2(x)
        return predictions[:,-1]
model = LSTMModel(input_size=config["model"]["input_size"], hidden_layer_size=config["model"]["lstm_size"], num_layers=config["model"]["num_lstm_layers"], output_size=1, dropout=config["model"]["dropout"])
model = model.to(config["training"]["device"])

model.load_state_dict(torch.load("" + user_input + ".pth"))

# re-initialize to ensure, data isnt shuffled when presenting data
train_dataloader = DataLoader(train_dataset, batch_size=config["training"]["batch_size"], shuffle=False)
val_dataloader = DataLoader(val_dataset, batch_size=config["training"]["batch_size"], shuffle=False)

model.eval()  # Prime for validation and testing

# predict on the training data, to see how well the model managed to learn and memorize
predicted_train = np.array([])

for idx, (x, y) in enumerate(train_dataloader):
    x = x.to(config["training"]["device"])
    out = model(x)
    out = out.cpu().detach().numpy()
    predicted_train = np.concatenate((predicted_train, out))

# predict on the validation data, to see how the model does

predicted_val = np.array([])

for (x, y) in val_dataloader:
    x = x.to(config["training"]["device"])
    out = model(x)                          # Forward prop
    out = out.cpu().detach().numpy()        # Converting output tensor into a numpy array
    predicted_val = np.concatenate((predicted_val, out))
    
st.write("Predicted price for next day is: ", scaler.inverse_transform(predicted_val[-1].reshape(-1, 1))[0][0])

# prepare data for plotting

to_plot_data_y_train_pred = np.zeros(num_data_points)
to_plot_data_y_val_pred = np.zeros(num_data_points)

to_plot_data_y_train_pred[config["data"]["window_size"]:split_index+config["data"]["window_size"]] = scaler.inverse_transform(predicted_train)
to_plot_data_y_val_pred[split_index+config["data"]["window_size"]:] = scaler.inverse_transform(predicted_val)

to_plot_data_y_train_pred = np.where(to_plot_data_y_train_pred == 0, None, to_plot_data_y_train_pred)
to_plot_data_y_val_pred = np.where(to_plot_data_y_val_pred == 0, None, to_plot_data_y_val_pred)



# plots

fig = figure(figsize=(25, 5), dpi=80)
fig.patch.set_facecolor((1.0, 1.0, 1.0))
plt.plot(data_date, data_close_price, label="Actual prices", color=config["plots"]["color_actual"])
plt.plot(data_date, to_plot_data_y_train_pred, label="Predicted prices (train)", color=config["plots"]["color_pred_train"])
plt.plot(data_date, to_plot_data_y_val_pred, label="Predicted prices (validation)", color=config["plots"]["color_pred_val"])
plt.title("Compare predicted prices to actual prices")
xticks = [data_date[i] if ((i%config["plots"]["xticks_interval"]==0 and (num_data_points-i) > config["plots"]["xticks_interval"]) or i==num_data_points-1) else None for i in range(num_data_points)] # make x ticks nice
x = np.arange(0,len(xticks))
plt.xticks(x, xticks, rotation='vertical')
plt.grid(visible=None, which='major', axis='y', linestyle='--')
plt.legend()
plt.show()
