# import dependencies
import pandas as pd
import numpy as np
import pmdarima as pm
from yahoo_finance import Share
import matplotlib.pyplot as plt
from matplotlib import style
style.use("fivethirtyseven")

# provide the details of stock to analyse
stock_ticker = None
start_date = None
end_date = None
forecast_horizon  = 7

# if stock-ticker is not provided stop execution
if stock_ticker is None:
    print("Ticker is not provided. Stopping execution.")
    exit()

# if end-date is none, assume latest date
if end_date is None:
    end_date = None

# if start-date is not given, use 01-Jan-2017 as start
if start_date is None:
    start_date = None


# fetch data for stock
stock = Share(stock_ticker)
stock_data = stock.get_historical(start_date, end_date)

# parse stock data

# divide closing stock prices into train-test set

# fit ARIMA model on daily closing price
model = pm.auto_arima(x_train)

# produce forecast for test set using optimal ARIMA model
forecast = model.score(x_test.shape[0])

# calculate Mean Absolute Percentage Error (MAPE)
mape = np.mean(np.abs(1 - x_train/x_test))

# find parameters for optimal model


# train optimal model on complete dataset
model_v2 = None

# plot forecast
plt.figure(figsize=(16,8))
plt.plot(actual, 'g', lael='Actual')
plt.plot(forecast, 'r', label='Forecast')
plt.legend()
plt.show()

# TO-DO
# Find likelihood for forecast