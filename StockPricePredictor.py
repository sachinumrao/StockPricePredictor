# import dependencies
import pandas as pd
import numpy as np
import pmdarima as pm
import pandas_datareader.data as web
import matplotlib.pyplot as plt
from matplotlib import style
import datetime

# provide the details of stock to analyse
stock_ticker = 'AAPL'

start_year = 2015
start_month = 1
start_day = 1

source = 'iex'

forecast_horizon  = 7

start_date = datetime.datetime(start_year, start_month, start_day)
end_date = datetime.datetime.now()

# fetch data for stock
stock = web.DataReader(stock_ticker, source, start_date, end_date)

n = stock.shape[0]
train_size = int(0.8*n)

# divide closing stock prices into train-test set
x_train = stock['close'][:train_size]
x_test = stock['close'][train_size:]

# fit ARIMA model on daily closing price
model = pm.auto_arima(x_train)

# produce forecast for test set using optimal ARIMA model
forecast = model.predict(x_test.shape[0])

# calculate Mean Absolute Percentage Error (MAPE)
mape = np.mean(np.abs(1 - forecast/x_test.values))
print(mape)

# find parameters for optimal model


# train optimal model on complete dataset
#model_v2 = None

# plot forecast
plt.figure(figsize=(16,8))
plt.plot(x_test.values, 'g', label='Actual')
plt.plot(forecast, 'r', label='Forecast')
plt.legend()
plt.show()

# TO-DO
# Optimze SARIMA model
# Find variance for forecast
# Improve the graph, support plotting of variance, date on x-axis