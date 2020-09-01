import numpy as np
import pandas as pd
import streamlit as st
from .process import time_parse
from sklearn.metrics import mean_absolute_error, mean_squared_error

data1_path = 'src/data/Data1.csv'
data2_path = 'src/data/Data2.csv'
data3_path = 'src/data/Data3.csv'


@st.cache
def load_data():
    data1 = pd.read_csv(data1_path, index_col=[0], parse_dates=True)
    data2 = pd.read_csv(data2_path, index_col=[0], parse_dates=True)
    data3 = pd.read_csv(data3_path, index_col=[0], parse_dates=True)
    data1 = time_parse(data1)
    data2 = time_parse(data2)
    data3 = time_parse(data3)
    return data1, data2, data3


@st.cache
def merged_humidity(data1, data2, data3):
    csv1 = data1.copy()
    csv2 = data2.copy()
    csv3 = data3.copy()
    csv2.rename(columns={'soil_humidity': 'soil_humidity2'}, inplace=True)
    csv3.rename(columns={'soil_humidity': 'soil_humidity3'}, inplace=True)
    merged_data = pd.merge(csv1, csv2['soil_humidity2'], how='left', left_index=True, right_index=True)
    merged_data = pd.merge(merged_data, csv3['soil_humidity3'], how='left', left_index=True, right_index=True )
    return merged_data[['soil_humidity', 'soil_humidity2', 'soil_humidity3']]


def rol_mean(data, step):
    rolmean = data.rolling(step, min_periods=1).mean()
    return rolmean


def rol_std(data, step):
    rolstd = data.rolling(step, min_periods=1).std()
    return rolstd


def estimate(model, x, y, scaler, model_type):
    pred = model.predict(x)
    pred = pred.reshape(-1, 1)
    y = y.reshape(-1, 1)
    if model_type == 'LSTM' or model_type == 'TCN':
        x = x[:, 0, :]
    pred = np.concatenate((x, pred), axis=1)
    real = np.concatenate((x, y), axis=1)
    prediction = scaler.inverse_transform(pred)
    real_values = scaler.inverse_transform(real)
    prediction = prediction[:, -1]
    real_values = real_values[:, -1]
    return prediction, real_values


def rmse(real, pred):
    return np.sqrt(mean_squared_error(real, pred))


def mae(real, pred):
    return mean_absolute_error(real, pred)
