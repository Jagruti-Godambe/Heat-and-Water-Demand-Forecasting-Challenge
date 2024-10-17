# Heat-and-Water-Demand-Forecasting-Challenge


## Abstract
This project leverages deep learning techniques, specifically Long Short-Term Memory (LSTM) networks, to forecast heat and water demand in a District Metered Area (DMA) in northern Denmark. Accurate forecasting is critical for optimizing resource usage, enhancing energy efficiency, and reducing operational costs in industries like urban planning, agriculture, and energy management.

The LSTM model predicts demand 24 hours in advance using hourly heat, water consumption, and meteorological data. Model performance is evaluated using Mean Absolute Error (MAE) and Mean Absolute Percentage Error (MAPE). Results show that the LSTM network effectively captures temporal patterns, delivering accurate and reliable demand forecasts.

## Features
- **Deep Learning Approach**: LSTM network for time series forecasting
- **Prediction Horizon**: 24 hours in advance
- **Metrics**: MAE and MAPE for performance evaluation
- **Data Sources**: Hourly heat and water usage data, meteorological data

## Data


### Heat Demand Data for Urban Area

- **Training Data**: Consists of hourly heat consumption for an urban area measured in kWh. This data is referred to as `Training_HeatDMA`.
- **Meter Data**: Contains the number of heat meters in the city that measure consumption, available in the file `HeatDMA_Number_of_Meters`.
- **Testing Data**: Includes consumption data with missing 24-hour ahead forecast values, labeled as `Testing_HeatDMA`.

### Water Demand Data for Urban and Rural Areas

- **Training Data**: Consists of hourly water consumption data for urban and rural areas, measured in kWh. This data is available in the files `WaterDMA1_Training` and `WaterDMA2_Training`.
- **Meter Data**: Contains the number of meters in the respective areas, referred to as `WaterDMA1_Number_of_Meters` and `WaterDMA2_Number_of_Meters`.
- **Testing Data**: Similar to the heat demand data, with missing values every eighth day, requiring 24-hour ahead forecasts. These files are labeled as `WaterDMA1_Testing` and `WaterDMA2_Testing`.


[**Project Data**](Data/): Project data if available here.

[**Project Analysis Report**](Time_Series_Project_Report.pdf): Final report containing data analysis and visualizations.


