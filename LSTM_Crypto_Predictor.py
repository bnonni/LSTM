#!/usr/bin/env python3

from tensorflow.keras.layers import Dense, LSTM, InputLayer
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

BTC_df = pd.read_csv('data.csv')
BTC_df.head()

keys = BTC_df.keys()
for k in keys:
    print(f'{k}: {len(BTC_df[k])}')

x = BTC_df["Datetime"]
y = BTC_df["Prices"]
plt.plot(x, y)
plt.show()

# BTC_df['High'] = BTC_df.High.astype('float64')
# BTC_df['Low'] = BTC_df.Low.astype('float64')
# BTC_df['Volumes'] = BTC_df.Volumes.astype('float64')

# ADL_avg = BTC_df.ADL.mean()
# RSI_avg = BTC_df.RSI.mean()
# ADL_slp_avg = BTC_df.ADL_slope.mean()
# OBV_slp_avg = BTC_df.OBV_slope.mean()

# values = {'OBV_slope': OBV_slp_avg, 'RSI': RSI_avg }
# BTC_df = BTC_df.fillna(value=values)
BTC_df.head()
BTC_df.info()

np.random.seed(42)

# 'High', 'Low', 'Volumes', 'RSI', 'ADL', 'OBV', 'ADL_slope', 'OBV_slope'

# features = [['High'], ['Low'], ['Volumes'], ['RSI'],
#              ['ADL'], ['OBV'], ['ADL_slope'], ['OBV_slope'],
#              ['High', 'Low'], ['High', 'Volumes'], ['High', 'RSI'], ['High', 'ADL'],
#              ['High', 'OBV'], ['High', 'ADL_slope'], ['High', 'OBV_slope']]

# for i, feature in enumerate(features):
try:
    # analysis_name = f'\n{feature} vs. Prices Analysis\n'
    analysis_name = f'\n[High, Low] vs. Prices Analysis\n'
    print(analysis_name)
    # dir_name = f'_vs_Prices'
    cwd = os.getcwd()
    if os.path.exists(f'{cwd}/results/New_Dataset/High_Low_vs_Prices'):
        pass
    else:
        os.mkdir(f'{cwd}/results/New_Dataset/High_Low_vs_Prices')
    # os.mkdir(f'{cwd}/results/Slim_Model/High_Low_vs_Prices')
    # file_path = f''

    f = open('results/New_Dataset/Results.txt', 'a')
    f.write(analysis_name)

    x = BTC_df[['High', 'Low']].values
    y = BTC_df['Prices'].values

    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.4, shuffle=False)

    scalar = MinMaxScaler(feature_range=(0, 1))
    X_train = scalar.fit_transform(X_train)
    X_test = scalar.fit_transform(X_test)

    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    model = Sequential()

    # input layer, initial dropout layer
    model.add(LSTM(50, input_shape=(X_train.shape[1], 1), return_sequences=True))

    model.add(LSTM(250, return_sequences=True))

    model.add(LSTM(250, return_sequences=True))

    model.add(LSTM(250, return_sequences=True))

    model.add(LSTM(250, return_sequences=True))

    # output layer
    model.add(LSTM())

    model.add(Dense(1, activation='tanh'))

    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mse', 'mape'])
    model.summary()

    # X_train = np.reshape(X_train, (X_train.shape[1], X_train.shape[0], 20))
    # print(f'X_train.shape: {X_train.shape}')
    # X_test = np.reshape(X_test, (X_test.shape[1], X_test.shape[0], 20))
    # print(f'X_test.shape: {X_test.shape}')

    model.fit(X_train, y_train, batch_size=32, validation_split=0.5, epochs=100)

    model_eval = model.evaluate(X_test, y_test, verbose=2)
    print(model_eval)
    f.write('\nModel Evaluation [X_test, y_test]')
    f.write(str(model_eval))
    f.write('\n')

    y_pred = model.predict(X_test, batch_size=20, verbose=2)
    f.write('\nModel Predictions [X_test => y_pred]')
    # f.write(str(y_pred))
    for i in y_pred:
        f.write(str(i))
    f.write('\n')

    model_eval_pred = model.evaluate(y_test, y_pred, verbose=2)
    print(model_eval_pred)
    f.write('\nModel Predictions Evaluation [X_test, y_pred]')
    f.write(str(model_eval_pred))
    f.write('\n')

except Exception as e:
    print(f'Error: {e}')
    err = open('Errors.txt', 'a')
    err.write(str(e))
    err.write('\n')