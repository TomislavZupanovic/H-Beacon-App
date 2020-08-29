import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from .utilities import rol_mean, rol_std


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
    axs[0].plot(csv.soil_humidity, color=color, label=f'{sensor} humidity')
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
    fig, ax = plt.subplots(figsize=(7, 3))
    plt.tick_params(axis='both', which='major', labelsize=7.5)
    sns.heatmap(csv.corr(), vmin=-1, vmax=1, cmap="coolwarm", linewidths=0.5, annot=True, ax=ax)
    return fig


def plot_estimations(pred, real):
    fig, axs = plt.subplots(nrows=1, ncols=1, squeeze=True, figsize=(18, 10))
    axs.plot(real, 'b-', label='Real Humidity')
    axs.plot(pred, 'r-', label='Estimated Humidity')
    axs.set_xlabel('Time point', fontsize=17)
    axs.set_ylabel('Humidity [%]', fontsize=17)
    axs.legend(loc='best', fontsize=17)
    axs.tick_params(axis='both', which='major', labelsize=17)
    axs.grid()
    return fig


def time_series_analysis(csv, roll_step=140):
    data = csv.copy()
    for var in ['69886_rssi','69886_snr']:
        data[var] = data[var].map(lambda x: np.power(10, x/10))
    fig, axs = plt.subplots(nrows=2, ncols=2, sharex=False, squeeze=True,
                            figsize=(16, 12))
    data_log = np.log(data)
    rolmean_log = rol_mean(data_log, roll_step)
    exp_weight_avg = data_log.ewm(halflife=roll_step, min_periods=0, adjust=True).mean()
    data_log_exp_avg = data_log - exp_weight_avg
    data_log_minus_avg = data_log - rolmean_log
    data_log_shift = data_log - data_log.shift(roll_step)
    for j, data_type in enumerate([data_log, data_log_minus_avg]):
        axs[0, j].plot(data_type.soil_humidity, color='blue', label='Original', alpha=0.6)
        axs[0, j].plot(rol_mean(data_type, roll_step).soil_humidity, color='red', label='Rolling Mean')
        axs[0, j].plot(rol_std(data_type, roll_step).soil_humidity, color='black', label='Rolling Std')
        axs[0, j].legend(loc='best')
        if j == 0:
            axs[0, j].set_title(f'Logartihm scale data\nRolling step: {roll_step}')
        else:
            axs[0, j].set_title(f'Log data minus average\nRolling step: {roll_step}')
        axs[0, j].grid()
    for i, data_type in enumerate([data_log_exp_avg, data_log_shift]):
        axs[1, i].plot(data_type.soil_humidity, color='blue', label='Original', alpha=0.6)
        axs[1, i].plot(rol_mean(data_type, roll_step).soil_humidity, color='red', label='Rolling Mean')
        axs[1, i].plot(rol_std(data_type, roll_step).soil_humidity, color='black', label='Rolling Std')
        axs[1, i].legend(loc='best')
        if i == 0:
            axs[1, i].set_title(f'Log data exp decay weighted average\nEWM step: {roll_step}')
        else:
            axs[1, i].set_title(f'Log data minus shifted log data\nShift step: {roll_step}')
        axs[1, i].grid()
    return fig