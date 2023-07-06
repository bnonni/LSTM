#!/usr/bin/env python3
# coding: utf-8

from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.models import Sequential
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from connect import updateCSV
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gc
import os
from sys import *
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
gc.collect()

if(len(argv) > 1):
    if(argv[1]  == 'y'):
        updateCSV()
else:
    pass

BTC_df = pd.read_csv('BTC.csv')
BTC_df.head()

keys = BTC_df.keys()
for k in keys:
    print(f'{k}: {len(BTC_df[k])}')

x = BTC_df["Datetime"]
y = BTC_df["Prices"]
plt.plot(x, y)
plt.show()

BTC_df['High'] = BTC_df.High.astype('float64')
BTC_df['Low'] = BTC_df.Low.astype('float64')
BTC_df['Volumes'] = BTC_df.Volumes.astype('float64')

ADL_avg = BTC_df.ADL.mean()
RSI_avg = BTC_df.RSI.mean()
ADL_slp_avg = BTC_df.ADL_slope.mean()
OBV_slp_avg = BTC_df.OBV_slope.mean()

values = {'OBV_slope': OBV_slp_avg, 'RSI': RSI_avg }
BTC_df = BTC_df.fillna(value=values)
BTC_df.head()
BTC_df.info()

np.random.seed(42)

single_features = ['High', 'Low', 'Volumes', 'RSI', 'ADL', 'ADL_slope', 'OBV', 'OBV_slope']
multi_features = [['High', 'Low'], ['High', 'Volumes'], ['High', 'RSI'], ['High', 'ADL'], ['High', 'ADL_slope'],
                   ['High', 'OBV'], ['High', 'OBV_slope'], ['Low', 'Volumes'], ['Low', 'RSI'], ['Low', 'ADL'],
                   ['Low', 'ADL_slope'], ['Low', 'OBV'], ['Low', 'OBV_slope'], ['Volumes', 'RSI'], ['Volumes', 'ADL'],
                   ['Volumes', 'ADL_slope'], ['Volumes', 'OBV'], ['Volumes', 'OBV_slope'], ['RSI', 'ADL'],
                   ['RSI', 'ADL_slope'], ['RSI', 'OBV'], ['RSI', 'OBV_slope'],['ADL','ADL_slope'], ['ADL', 'OBV'],
                   ['ADL', 'OBV_slope'], ['OBV', 'ADL_slope'], ['OBV', 'OBV_slope'], ['ADL_slope', 'OBV_slope']]


for i,feature in enumerate(single_features):
    try:
        analysis_name = f'\n[{feature}] vs. Prices Analysis\n'
        print(analysis_name)

        f = open(f'results/Loop/{feature}.txt', 'a')
        f.write(analysis_name)

        X = BTC_df[[feature]].values
        y = BTC_df['Prices'].values

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1)

        scalar = MinMaxScaler(feature_range=(0, 1))
        X_train = scalar.fit_transform(X_train)
        X_test = scalar.fit_transform(X_test)

        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

        model = Sequential()

        model.add(LSTM(200, return_sequences=True, input_shape=(X_train.shape[1], 1)))
        model.add(Dropout(0.2))
        # hidden layer
        model.add(LSTM(200, return_sequences=True))
        model.add(Dropout(0.2))

        # hidden layer
        model.add(LSTM(200, return_sequences=True))
        model.add(Dropout(0.2))

        # hidden layer
        model.add(LSTM(200, return_sequences=True))
        model.add(Dropout(0.2))

        # hidden layer
        model.add(LSTM(200, return_sequences=True))
        model.add(Dropout(0.2))

        # hidden layer
        model.add(LSTM(250))
        model.add(Dropout(0.2))

        # output layer
        model.add(Dense(1))

        model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mape'])
        summary = model.summary()
        print(summary)
        f.write(str(summary))
        f.write('\n')

        model.fit(np.asarray(X_train), np.asarray(y_train), validation_split=0.2,
                  epochs=100, shuffle=False, batch_size=32)


        model_eval = model.evaluate(X_test, y_test, verbose=2)
        print(model_eval)
        f.write(str(model_eval))
        f.write('\n')

    except Exception as e:
        err = open('Errors.txt', 'a')
        err.write(str(e))
        err.write('\n')