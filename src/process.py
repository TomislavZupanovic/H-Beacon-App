import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler


def time_parse(csv):
    csv.reset_index(inplace=True)
    csv.time = csv.time.dt.tz_localize('UTC')
    csv.set_index('time', drop=True, inplace=True)
    return csv


def split_sequences(sequences, n_steps):
    x, y = [], []
    for i in range(len(sequences)):
        end_ix = i + n_steps
        if end_ix > len(sequences):
            break
        seq_x, seq_y = sequences[i:end_ix, :-1], sequences[end_ix-1, -1]
        x.append(seq_x)
        y.append(seq_y)
    return np.array(x), np.array(y)


def process_data_model(csv, csv_to_process, model_type):
    whole_data = csv.copy().values
    data = csv_to_process.copy().values
    scaler = MinMaxScaler(feature_range=(0, 1))
    split = int(len(whole_data) * 0.6)
    scale_data = whole_data[:split]
    scaler.fit(scale_data)
    data = scaler.transform(data)
    if model_type == 'LSTM' or model_type == 'TCN':
        x_data, y_data = split_sequences(data, 6)
    elif model_type == 'Neural Network':
        x_data, y_data = data[:, :-1], data[:, -1]
    return x_data, y_data, scaler


def rolling_before_model(csv):
    data = csv.copy()
    data.iloc[:, :-1] = data.iloc[:, :-1].ewm(halflife=140, min_periods=0, adjust=True).mean()
    return data
