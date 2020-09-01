from tensorflow.keras import models
from tensorflow.keras.models import model_from_json
import streamlit as st
from src.plotting import *
from src.utilities import *
from src.process import process_data_model, rolling_before_model
from tcn import TCN

save_dir_nn = 'saved_models/NN_128_64.h5'
save_dir_lstm = 'saved_models/LSTM_step6_1_12.h5'
save_dir_tcn = 'saved_models/TCN.h5'

col_names = {
    'pressure': 'Pressure',
    'air_temp': 'Air Temperature',
    'air_humidity': 'Air Humidity',
    '69886_rssi': 'RSSI',
    '69886_snr': 'SNR',
    'soil_humidity': 'Soil Humidity'
}


def main():
    st.title('H-Beacon: Smart Irrigation System')
    start_text = st.write('Data visualization and Model demonstration tool.'
                          ' Choose the option from the left sidebar.')
    st.sidebar.title("Menu")
    app_mode = st.sidebar.selectbox("Choose the app mode",
                                    ["Home", "Explore data", "Use the Model"])
    if app_mode == "Explore data":
        explore_data()
    elif app_mode == "Use the Model":
        use_model()


def explore_data():
    st.header('Explore data')
    data_load_state = st.text('Loading data...')
    try:
        data1, data2, data3 = load_data()
        data_load_state.text('Loading data...done!')
    except OSError:
        data_load_state.text('Error loading data...')
    show_data = st.button('Show dataframe')
    if show_data:
        st.text('Samples of sensor 1:')
        st.dataframe(data1.head(10))
    st.subheader('Soil humidity')
    plot_type = st.checkbox('Interactive plot (Note: takes a while to load)')
    if plot_type:
        st.line_chart(merged_humidity(data1, data2, data3))
    else:
        st.pyplot(hum_plot(merged_humidity(data1, data2, data3)))
    st.subheader('Explore sensor data')
    sensor_plot = st.selectbox("Choose sensor", ['Sensor 1', 'Sensor 2', 'Sensor 3'])
    column = st.radio('Variables', list(data1.columns), format_func=col_names.get)
    rolling_check = st.checkbox('Use Moving Average')
    if sensor_plot == 'Sensor 1':
        data_to_plot = data1
    elif sensor_plot == 'Sensor 2':
        data_to_plot = data2
    else:
        data_to_plot = data3
    if rolling_check:
        roll_step = st.slider('Moving Average step (140 roughly represents 1 day)', min_value=1, max_value=300)
    else:
        roll_step = None
    time_chunk = st.slider("Time frame", min_value=data_to_plot.index.to_pydatetime()[0],
                           max_value=data_to_plot.index.to_pydatetime()[-1],
                           value=(data_to_plot.index.to_pydatetime()[0], data_to_plot.index.to_pydatetime()[-1]),
                           format='YY/MM/DD')
    lower, upper = time_chunk
    time_frame_data = data_to_plot.loc[lower.strftime('%Y-%m-%d'): upper.strftime('%Y-%m-%d')]
    st.pyplot(plotting(time_frame_data, column, sensor_plot, roll_step))
    st.text('Descriptive statistics (selected time frame)')
    st.dataframe(time_frame_data.describe())
    corr = st.checkbox('Correlation matrix (selected time frame)')
    if corr:
        st.pyplot(corr_plot(time_frame_data))
    scatter_plot = st.checkbox('Scatter plotting (selected time frame)')
    if scatter_plot:
        column_x = st.selectbox('X axis feature', time_frame_data.columns, format_func=col_names.get)
        column_y = st.selectbox('Y axis feature', time_frame_data.columns, format_func=col_names.get)
        distr = st.checkbox('Plot distributions')
        st.pyplot(scatter_plotting(time_frame_data[column_x], time_frame_data[column_y],
                                   column_x, column_y, distr=distr))
    histogram = st.checkbox('Histograms (selected time frame)')
    if histogram:
        col = st.selectbox('Select feature', time_frame_data.columns, format_func=col_names.get)
        st.pyplot(histogram_plot(time_frame_data, col))
    st.subheader("Further soil humidity analysis")
    show_analysis = st.button('Show')
    if show_analysis:
        step = st.number_input('Specify step   (Default value of 140 roughly represents 1 day)', 1, 1000, 140)
        if sensor_plot == 'Sensor 1':
            st.markdown('Sensor 1')
            fig = time_series_analysis(data1, step)
            st.pyplot(fig)
        elif sensor_plot == 'Sensor 2':
            st.markdown('Sensor 2')
            fig = time_series_analysis(data2, step)
            st.pyplot(fig)
        elif sensor_plot == 'Sensor 3':
            st.markdown('Sensor 3')
            fig = time_series_analysis(data3, step)
            st.pyplot(fig)


def use_model():
    st.header('Use the Model')
    data_load = st.text('Loading data...')
    data1, data2, data3 = load_data()
    data_load.text('Loading data...done!')
    model_choose = st.selectbox("Choose trained Model", ['Neural Network', 'LSTM', 'TCN'])
    if model_choose == 'Neural Network':
        try:
            model = models.load_model(save_dir_nn)
            st.text('Loaded Neural Network Model!')
        except RuntimeError:
            st.text('Error while loading model')
    elif model_choose == 'LSTM':
        try:
            model = models.load_model(save_dir_lstm)
            st.text('Loaded LSTM Model!')
        except RuntimeError:
            st.text('Error while loading model')
    elif model_choose == 'TCN':
        try:
            loaded_json = open(r'saved_models/TCN.json', "r").read()
            model = model_from_json(loaded_json, custom_objects={'TCN': TCN})
            model.load_weights(r'saved_models/TCN_weights.h5')
            st.text('Loaded TCN Model!')
        except RuntimeError:
            st.text('Error while loading model')
    summary = st.checkbox('Model summary')
    if summary:
        st.write('---TODO')
    st.text('Note: Models only work on sensor 1 at this moment.')
    time_to_estimate = st.slider("Time frame on which to estimate", min_value=data1.index.to_pydatetime()[0],
                                 max_value=data1.index.to_pydatetime()[-1],
                                 value=(data1.index.to_pydatetime()[0], data1.index.to_pydatetime()[-1]),
                                 format='YY/MM/DD')
    low, high = time_to_estimate
    data_for_model = rolling_before_model(data1)
    data_to_estimate = data_for_model.loc[low.strftime('%Y-%m-%d'): high.strftime('%Y-%m-%d')]
    x, y, scaler = process_data_model(data_for_model, data_to_estimate, model_choose)
    prediction, real_value = estimate(model, x, y, scaler, model_choose)
    st.pyplot(plot_estimations(prediction, real_value))
    st.write('Metrics on selected time frame:')
    st.text('RMSE: {:.3f}\nMAE: {:.3f}'.format(rmse(real_value, prediction), mae(real_value, prediction)))
    st.pyplot(residual_error_plot(prediction, real_value))


if __name__ == "__main__":
    main()
