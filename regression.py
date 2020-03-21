#!bin/python3.6
# Predicting Stock market prices: google

import pandas as pd 
import math


# Google stock data
file_name = "./GOOG.csv"
df = pd.read_csv(file_name)

df = df[['Open', 'High', 'Low', 'Adj Close', 'Volume']]

# Calculate High - Low percent
df['HL_PCT'] = ((df['High'] - df['Adj Close'])/df['Adj Close'])

# Calculating percent change

df['PCT_change'] = (df['Adj Close'] - df['Open']) / df['Open']

df = df[['Adj Close', 'HL_PCT', 'PCT_change', 'Volume']]

forecast_col = 'Adj Close'
df.fillna(-99999, inplace=True)

print("Total length", len(df))

# Taking out 10 percent data for prediction
forecast_out = int(math.ceil(0.1 * len(df)))

print("Prediction length: ", forecast_out)


df['label'] = df[forecast_col].shift(-forecast_out)
df.dropna(inplace=True)
print(df)
