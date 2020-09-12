# H-Beacon-App
Data analysis, visualization and Model demonstration interface using [Streamlit](https://www.streamlit.io/). It's a part of [H-Beacon: Smart Irrigation System](https://github.com/h-beacon/humidity-time-series) project. Enables easy and quick sensor data analysis, visualization and trained models inference for any newly acquired data without repeated coding. Application is deployed on [Heroku](https://www.heroku.com/).

H-Beacon is the deep sequential neural network model that estimates soil humidity from the strength of the LoRa-beacon IoT signal. We are funded by Horizon 2020 EU funding for Research & Innovation.

<p align="center">
  <img src="http://www.waveform.hr/wp-content/uploads/2020/05/tetramax_EC-768x114.png">
</p>

## Contact

[contact @ waveform.hr](http://www.waveform.hr/#contact)

## Use Web Application

[H-Beacon-App](https://h-beacon.herokuapp.com/)

## Installation and running
```bash
$ git clone https://github.com/TomislavZupanovic/H-Beacon-App.git
```
```bash
$ streamlit run app.py
```

## Examples
- **Variables plot in respect to soil humidity**

  Plotting variables for chosen sensor for any time frame and moving average to see any occuring patterns in data.
<img src="https://github.com/TomislavZupanovic/H-Beacon-App/blob/master/showcase_images/variables_plot.jpg" width="600" height="450">

- **Correlation matrix**
  
  Descriptive statistics and correlations matrix for any chosen time frame.
<img src="https://github.com/TomislavZupanovic/H-Beacon-App/blob/master/showcase_images/correlation_matrix.jpg" width="600" height="550">

- **Scatter plot**

  Scatter plot between any two variables with dropdown menu for choosing.
<img src="https://github.com/TomislavZupanovic/H-Beacon-App/blob/master/showcase_images/scatter_plot.jpg" width="600" height="550">

- **Data transformations and analysis**

  Data can be easily transformed (Logarithm, Squared etc.) to check for distributions with histograms, applying operations to analize stationarity and standard deviations.
<img src="https://github.com/TomislavZupanovic/H-Beacon-App/blob/master/showcase_images/further_analysis.jpg" width="600" height="550">
<img src="https://github.com/TomislavZupanovic/H-Beacon-App/blob/master/showcase_images/further_analysis_2.jpg" width="600" height="550">

- **Model inference**

  Choosing trained models to estimate soil humidity on any time frame, showing metrics, residual and error plots for model performance.
<img src="https://github.com/TomislavZupanovic/H-Beacon-App/blob/master/showcase_images/model.jpg" width="600" height="550">
