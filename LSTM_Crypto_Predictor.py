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

# BTC_epochs, BTC_prices, BTC_volumes, \
#         BTC_highs, BTC_lows, BTC_RSIs, \
#         BTC_ADLs, BTC_ADL_slope, BTC_OBVs, BTC_OBV_slope = pullFromMongo()

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

# print('\n[Low, High] Analysis\n')
# print('\n[Low, RSI] Analysis\n')
# print('\n[Low, ADL] Analysis\n')
# print('\n[Low, ADL_slope] Analysis\n')
# print('\n[Low, OBV] Analysis\n')
# print('\n[Low, OBV_slope] Analysis\n')

# print('\n[RSI, High] Analysis\n')
# print('\n[RSI, Low] Analysis\n')
# print('\n[RSI, ADL] Analysis\n')
# print('\n[RSI, ADL_slope] Analysis\n')
# print('\n[RSI, OBV] Analysis\n')
# print('\n[RSI, OBV_slope] Analysis\n')

# print('\n[ADL, High] Analysis\n')
# print('\n[ADL, Low] Analysis\n')
# print('\n[ADL, RSI] Analysis\n')
# print('\n[ADL, ADL_slope] Analysis\n')
# print('\n[ADL, OBV] Analysis\n')
# print('\n[ADL, OBV_slope] Analysis\n')

# print('\n[OBV, High] Analysis\n')
# print('\n[OBV, Low] Analysis\n')
# print('\n[OBV, RSI] Analysis\n')
# print('\n[OBV, ADL] Analysis\n')
# print('\n[OBV, ADL_slope] Analysis\n')
# print('\n[OBV, OBV_slope] Analysis\n')

# print('\n[ADL_slope, High] Analysis\n')
# print('\n[ADL_slope, Low] Analysis\n')
# print('\n[ADL_slope, RSI] Analysis\n')
# print('\n[ADL_slope, ADL] Analysis\n')
# print('\n[ADL_slope, OBV] Analysis\n')
# print('\n[ADL_slope, OBV_slope] Analysis\n')

# print('\n[OBV_slope, High] Analysis\n')
# print('\n[OBV_slope, Low] Analysis\n')
# print('\n[OBV_slope, RSI] Analysis\n')
# print('\n[OBV_slope, ADL] Analysis\n')
# print('\n[OBV_slope, ADL_slope] Analysis\n')
# print('\n[OBV_slope, OBV] Analysis\n')

# print('\n[High, Prices] Analysis\n')
# print('\n[Volumes, Prices] Analysis\n')
# print('\n[Low, Prices] Analysis\n')
# print('\n[RSI, Prices] Analysis\n')
# print('\n[ADL, Prices] Analysis\n')
# print('\n[OBV, Prices] Analysis\n')
# print('\n[ADL_slope, Prices] Analysis\n')
# print('\n[OBV_slope, Prices] Analysis\n')

features = ['Low', 'RSI', 'ADL', 'ADL_slope', 'OBV', 'OBV_slope']

# for i,feature in enumerate(features):
try:
    # CHANGE ANALYSIS NAME PER ROUND
    analysis_name = ''
        # f'\n[High + {feature}] vs. Prices Analysis\n'
    print(analysis_name)

    # CHANGE FILE PATH PER ROUND
    # f = open(f'results/Auto_Looping/Round3/High_{feature}.txt', 'a')
    f = open(f'results/42.txt', 'a')
    f.write(analysis_name)

    # CHANGE DATA PER ROUND
    x = BTC_df[['High']].values
    y = BTC_df['Prices'].values

    X, X_test, Y, y_test = train_test_split(x, y, test_size=0.4, random_state=1)
    X_train, X_validate, y_train, y_validate = train_test_split(X, Y, test_size=0.25, random_state=1)

    scalar = MinMaxScaler(feature_range=(0, 1))
    X_train = scalar.fit_transform(X_train)
    X_test = scalar.fit_transform(X_test)
    X_validate = scalar.fit_transform(X_validate)

    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    X_validate = np.reshape(X_validate, (X_validate.shape[0], X_validate.shape[1], 1))

    model = Sequential()

    model.add(LSTM(250, return_sequences=True, input_shape=(X_train.shape[1], 1)))

    # hidden layer
    model.add(LSTM(250, return_sequences=True))

    # hidden layer
    model.add(LSTM(250, return_sequences=True))

    # hidden layer
    model.add(LSTM(250, return_sequences=True))

    # hidden layer
    model.add(LSTM(250, return_sequences=True))
    model.add(Dropout(0.5))

    # hidden layer
    model.add(LSTM(250))
    model.add(Dropout(0.5))

    # output layer
    model.add(Dense(1))

    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mse', 'mape'])
    summary = model.summary()
    print(summary)
    f.write(str(summary))
    f.write('\n')

    model.fit(np.asarray(X_train), np.asarray(y_train), validation_data=(X_validate, y_validate),
              epochs=100, shuffle=True, batch_size=32)


    model_eval = model.evaluate(X_test, y_test, verbose=2)
    print(model_eval)
    f.write(str(model_eval))
    f.write('\n')

except Exception as e:
    err = open('Errors.txt', 'a')
    err.write(str(e))
    err.write('\n')

# DONE

# print('\n[High, Low] Analysis\n')
# print('\n[High, RSI] Analysis\n')
# print('\n[High, ADL] Analysis\n')
# print('\n[High, ADL_slope] Analysis\n')
# print('\n[High, OBV] Analysis\n')
# print('\n[High, OBV_slope] Analysis\n')

# print('\n[Volumes] Analysis\n')
# print('\n[High] Analysis\n')
# print('\n[Low] Analysis\n')
# print('\n[RSI] Analysis\n')
# print('\n[ADL] Analysis\n')
# print('\n[OBV] Analysis\n')
# print('\n[ADL_slope] Analysis\n')
# print('\n[OBV_slope] Analysis\n')

# print('\n[Volumes, High] Analysis\n')
# print('\n[Volumes, Low] Analysis\n')
# print('\n[Volumes, RSI] Analysis\n')
# print('\n[Volumes, ADL] Analysis\n')
# print('\n[Volumes, ADL_slope] Analysis\n')
# print('\n[Volumes, OBV] Analysis\n')
# print('\n[Volumes, OBV_slope] Analysis\n')

