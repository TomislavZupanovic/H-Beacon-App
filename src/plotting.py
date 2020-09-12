import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import pandas as pd
import numpy as np
from .utilities import rol_mean, rol_std
from statsmodels.graphics import tsaplots
from statsmodels.tsa.stattools import ccf, adfuller


def hum_plot(csv):
    fig, axs = plt.subplots(nrows=1, ncols=1, sharex=False, squeeze=True,
                            figsize=(16, 8))
    plt.tick_params(axis='both', which='major', labelsize=12)
    axs.plot(csv.soil_humidity, color='blue', label='Sensor 1 humidity')
    axs.plot(csv.soil_humidity2, color='orange', label='Sensor 2 humidity')
    axs.plot(csv.soil_humidity3, color='red', label='Sensor 3 humidity')
    axs.legend(loc='best')
    axs.grid()
    return fig


def plotting(csv, column, sensor, roll=None):
    if sensor == 'Sensor 1':
        color = 'blue'
    elif sensor == 'Sensor 2':
        color = 'orange'
    elif sensor == 'Sensor 3':
        color = 'red'
    if roll:
        csv_column = csv[column].rolling(roll, min_periods=1).mean()
    else:
        csv_column = csv[column]
    fig, axs = plt.subplots(nrows=2, ncols=1, squeeze=True, figsize=(16, 13))
    axs[0].plot(csv.soil_humidity, color=color, label=f'{sensor} soil humidity')
    axs[0].legend(loc='best', fontsize=17)
    axs[0].grid()
    axs[0].tick_params(axis='both', which='major', labelsize=17)
    axs[1].plot(csv_column, color=color, label=f'{sensor} {column}', alpha=0.7)
    axs[1].legend(loc='best', fontsize=17)
    axs[1].grid()
    axs[1].tick_params(axis='both', which='major', labelsize=17)
    fig.autofmt_xdate(rotation=45)
    plt.tight_layout()
    return fig


def corr_plot(csv):
    half = np.triu(csv.corr())
    fig, ax = plt.subplots(figsize=(7, 3))
    plt.tick_params(axis='both', which='major', labelsize=7.5)
    sns.heatmap(csv.corr(), vmin=-1, vmax=1, cmap="coolwarm", linewidths=0.5, annot=True, ax=ax, mask=half)
    return fig


def plot_estimations(pred, real):
    fig, axs = plt.subplots(nrows=1, ncols=1, squeeze=True, figsize=(15, 8))
    axs.plot(real, 'b-', label='Real Humidity')
    axs.plot(pred, 'r-', label='Estimated Humidity')
    axs.set_xlabel('Time point', fontsize=17)
    axs.set_ylabel('Humidity [%]', fontsize=17)
    axs.legend(loc='best', fontsize=17)
    axs.tick_params(axis='both', which='major', labelsize=17)
    axs.grid()
    return fig


def scatter_plotting(column_x, column_y, x_label, y_label, distr=False):
    if not distr:
        fig, ax = plt.subplots(figsize=(9, 6))
        ax.scatter(x=column_x[::5], y=column_y[::5], alpha=0.7)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        return fig
    else:
        plot = sns.jointplot(x=column_x[::5], y=column_y[::5], height=7, alpha=0.7)
        return plot


def residual_error_plot(pred, real):
    residuals = real - pred
    fig, axs = plt.subplots(nrows=2, ncols=1, squeeze=True, figsize=(15, 25))
    axs[0].scatter(x=pred[::10], y=residuals[::10], color='g', alpha=0.7)
    axs[0].axhline(y=0, linestyle='--', color='black', linewidth=3.5)
    axs[0].set_xlabel('Estimations', fontsize=17)
    axs[0].set_ylabel('Residuals', fontsize=17)
    axs[0].set_title('Estimation residuals\n (Every 10th data point for visibility)', fontsize=18)
    axs[0].tick_params(axis='both', which='major', labelsize=17)
    axs[0].grid()
    axs[1].scatter(x=pred[::10], y=real[::10], c='orange', alpha=0.7)
    axs[1].plot([0, 1], [0, 1], transform=axs[1].transAxes, ls="--", c=".1", linewidth=3, label='Best fit')
    axs[1].set_xlabel('Estimations', fontsize=17)
    axs[1].set_ylabel('Real values', fontsize=17)
    axs[1].set_title('Estimation error\n (Every 10th data point for visibility)', fontsize=18)
    axs[1].tick_params(axis='both', which='major', labelsize=17)
    axs[1].set_xlim(9, 31)
    axs[1].set_ylim(9, 31)
    axs[1].grid()
    axs[1].legend(loc='best', fontsize=17)
    return fig


def histogram_plot(csv, column):
    fig, axs = plt.subplots(nrows=1, ncols=1, squeeze=True, figsize=(11, 7))
    sns.distplot(csv[column], bins=15)
    plt.grid()
    return fig


def autocorr_plot(csv, column, max_lag):
    data = csv.copy()
    x = data[column].values
    y = data.soil_humidity.values
    cross_corr = ccf(y, x)
    fig, axs = plt.subplots(nrows=2, ncols=1, squeeze=True, figsize=(14, 11))
    axs[0].plot(cross_corr[:max_lag], linewidth=3)
    axs[0].set_title(f'Cross-Correlation {column} with soil humidity', fontsize=17)
    axs[0].grid()
    tsaplots.plot_acf(x, ax=axs[1], lags=max_lag)
    axs[1].set_title(f'Auto-Correlation of {column}', fontsize=17)
    axs[1].set_xlabel('Lags', fontsize=17)
    axs[1].grid()
    return fig


def time_series_analysis(csv, column, transformation, analysis_type, roll_step):
    data = csv.copy()
    if transformation == 'Logarithm':
        for var in ['69886_rssi', '69886_snr']:
            data[var] = data[var].map(lambda x: np.power(10, x / 10))
        data_transformed = np.log(data)
    elif transformation == 'Squared':
        data_transformed = np.square(data)
    elif transformation == 'Square Root':
        for var in ['69886_rssi', '69886_snr']:
            data[var] = data[var].map(lambda x: np.power(10, x / 10))
        data_transformed = np.sqrt(data)
    elif transformation == 'Original':
        data_transformed = data
    if analysis_type == 'Minus Average':
        data_mean = rol_mean(data_transformed, roll_step)
        data_to_analize = data_transformed - data_mean
    elif analysis_type == 'Minus Weighted Average':
        exp_weight_avg = data_transformed.ewm(halflife=roll_step, min_periods=0, adjust=True).mean()
        data_to_analize = data_transformed - exp_weight_avg
    elif analysis_type == 'Minus Shifted':
        data_to_analize = data_transformed - data_transformed.shift(roll_step)
    fig, axs = plt.subplots(nrows=3, ncols=1, squeeze=True, figsize=(13, 20))
    axs[0].plot(data_transformed[column], color='blue', label=f'{column}', alpha=0.5)
    axs[0].plot(rol_mean(data_transformed, roll_step)[column], color='red', label='Rolling Mean')
    axs[0].axhline(y=data_transformed[column].mean(), linestyle='--', color='black', linewidth=2, label='Mean')
    axs[0].legend(loc='best')
    axs[0].set_title(f'{transformation} data', fontsize=17)
    axs[0].grid()
    axs[1].plot(data_to_analize[column], color='blue', label=f'{column}', alpha=0.5)
    axs[1].plot(rol_mean(data_to_analize, roll_step)[column], color='red', label='Rolling mean')
    axs[1].plot(rol_std(data_to_analize, roll_step)[column], color='black', label='Rolling Std')
    axs[1].legend(loc='best')
    axs[1].set_title(f'{transformation} {analysis_type} data', fontsize=17)
    axs[1].grid()
    sns.distplot(data_transformed[column], bins=15, ax=axs[2])
    axs[2].set_title(f'{transformation} data', fontsize=17)
    if data_to_analize.isnull().sum().any():
        data_to_analize.dropna(axis=0, inplace=True)
    return fig, data_transformed, data_to_analize

