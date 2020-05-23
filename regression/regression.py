#!bin/python3.6
# Predicting Stock market prices: google

import pandas as pd 
import math
import numpy as np
from sklearn import preprocessing, svm
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

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

print(df)

df['label'] = df[forecast_col].shift(-forecast_out)
df.dropna(inplace=True)

# Features
X = np.array(df.drop(['label'], 1))

# Labels
y = np.array(df['label'])

# Splitting Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1)

# Classifier
clf = LinearRegression()
clf.fit(X_train, y_train)

epoch = 5


# Testing
confidence = clf.score(X_test, y_test)

print(confidence)
