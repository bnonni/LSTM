#!/usr/bin/env python
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
# In[1]:

if(len(argv) > 1):
    if(argv[1]  == 'y'):
        updateCSV()
else:
    pass

# In[1]:
BTC_df = pd.read_csv('BTC.csv')
BTC_df.head()

# In[1]:
keys = BTC_df.keys()
for k in keys:
    print(f'{k}: {len(BTC_df[k])}')


# In[1]:
x = BTC_df["Datetime"]
y = BTC_df["Prices"]
plt.plot(x, y)
plt.show()
# In[1]:
BTC_df['High'] = BTC_df.High.astype('float64')
BTC_df['Low'] = BTC_df.Low.astype('float64')
BTC_df['Volumes'] = BTC_df.Volumes.astype('float64')
# In[1]:
ADL_avg = BTC_df.ADL.mean()
RSI_avg = BTC_df.RSI.mean()
ADL_slp_avg = BTC_df.ADL_slope.mean()
OBV_slp_avg = BTC_df.OBV_slope.mean()
# In[1]:
values = {'OBV_slope': OBV_slp_avg, 'RSI': RSI_avg }
BTC_df = BTC_df.fillna(value=values)
BTC_df.head()
BTC_df.info()
# In[1]:
np.random.seed(42)

# print('\n[OBV_slope, Prices] Analysis\n')

# print('\n[ADL_slope, Prices] Analysis\n')

# print('\n[OBV, OBV_slope, Prices] Analysis\n')

# print('\n[ADL, ADL_slope, Prices] Analysis\n')

# print('\n[RSI, High, Low] Analysis\n')

# print('\n[ADL, High, Low] Analysis\n')

# print('\n[OBV, High, Low] Analysis\n')

# print('\n[ADL, ADL_slope, High, Low] Analysis\n')

# print('\n[OBV, OBV_slope, High, Low] Analysis\n')

# print('\n[ADL_slope, High, Low] Analysis\n')

# print('\n[OBV_slope, High, Low] Analysis\n')

print('\n[ADL, High, Low] Analysis\n')
X = BTC_df[['ADL','High','Low']].values
Y = BTC_df['Prices'].values
print(X.shape)
print(y.shape)

# In[1]:
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.4, random_state=1)

# In[1]:
scalar = MinMaxScaler(feature_range=(0, 1))
X_train = scalar.fit_transform(X_train)
X_test = scalar.fit_transform(X_test)

X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
# In[1]:
model = Sequential()

#input layer
model.add(LSTM(200, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model.add(Dropout(0.2))

#hidden layer
model.add(LSTM(200, return_sequences=True))
model.add(Dropout(0.2))

# #hidden layer
model.add(LSTM(200, return_sequences=True))
model.add(Dropout(0.2))

# #hidden layer
model.add(LSTM(200, return_sequences=True))
model.add(Dropout(0.2))

# hidden layer
model.add(LSTM(200))
model.add(Dropout(0.2))

# output layer
model.add(Dense(1))
# In[1]:
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mse', 'mae', 'mape'])
print(model.summary())
# In[1]:
model.fit(np.asarray(X_train), np.asarray(y_train), validation_data=(X_test, y_test), epochs=100, shuffle=True, batch_size=32)

model_loss = model.evaluate(X_test, y_test, verbose=2)
print(model_loss)

