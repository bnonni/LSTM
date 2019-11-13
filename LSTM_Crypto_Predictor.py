#!/usr/bin/env python
# coding: utf-8

# In[394]:


from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.models import Sequential
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from url import URL
from pymongo import *
import statistics as stat

import numpy as np
import pandas as pd
from time import strptime, mktime
import gc
import sys
import re
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
gc.collect()

client = MongoClient(URL)
db = client.crypto_wallet


# In[395]:
def checkLen(a, b):
    if len(a) == len(b):
        return True
    else:
        return f'DB Objs:{len(a)} < Clean Arr Items:{len(b)}' if len(a) < len(b) else f'Clean Arr Items:{len(b)} < DB Objs:{len(a)}'


# In[396]:


def p(o):
    return print(o)


# In[397]:


def filterData(coll, st, narr):
    for obj in coll:
        try:
            tmp = obj.get(st)
            narr.append(tmp)
        except Exception as e:
            print(e, o['_id'])
    return narr


# In[398]:


def datetime_converter(dtstr):
    tmstmp = strptime(dtstr, '%Y-%m-%d %H:%M:%S')
    epoch = mktime(tmstmp)
    return int(epoch)


# In[399]:


BTC_Tickers_Collection = db.BTC_Tickers
BTC_Tickers_Objs = list(BTC_Tickers_Collection.find())
BTC_dt_epochs = []
BTC_prices = []
BTC_volumes = []
BTC_highs = []
BTC_lows = []
for obj in BTC_Tickers_Collection.find():
    dt = re.sub(r'\..*', '', obj.get('time')).replace('T', ' ').rstrip('Z')
    BTC_dt_epochs.append(datetime_converter(dt))
    BTC_prices.append(float(obj.get('price')))
    BTC_volumes.append(obj.get('volume'))
    BTC_highs.append(obj.get('ask'))
    BTC_lows.append(obj.get('bid'))


# In[400]:


p(checkLen(BTC_Tickers_Objs, BTC_prices))
p(checkLen(BTC_dt_epochs, BTC_prices))
p(checkLen(BTC_Tickers_Objs, BTC_volumes))
p(checkLen(BTC_Tickers_Objs, BTC_highs))
p(checkLen(BTC_Tickers_Objs, BTC_lows))


# In[401]:


BTC_RSI_Collection = db.BTC_RSI14_Data
BTC_RSI_Objs = list(BTC_RSI_Collection.find())
BTC_RSIs = []
Errors = []
for rsio in BTC_RSI_Collection.find():
    RSI = rsio.get('RSI')
    try:
        if type(RSI) == float:
            BTC_RSIs.append(int(RSI))
        elif type(RSI) == list:
            if RSI[0] == None:
                pass
            else:
                BTC_RSIs.append(int(stat.mean(RSI)))
        else:
            BTC_RSIs.append(RSI)
    except Exception as e:
        Errors.append(rsio['_id'])
        print(e, rsio['_id'])
        sys.exit(1)


# In[402]:


if len(Errors) > 0:
    print(Errors)


# In[403]:


p(checkLen(BTC_RSI_Objs, BTC_RSIs))


# In[404]:


BTC_ADL_Collection = db.BTC_ADL_Data
BTC_ADL_Objs = list(BTC_ADL_Collection.find())
BTC_ADLs = []
BTC_ADL_slope = []
for o in BTC_ADL_Collection.find():
    ADL = o.get('ADL')
    slope = o.get('slope')
    try:
        if type(ADL) == float:
            BTC_ADLs.append(int(ADL))
        elif type(ADL) == list:
            BTC_ADLs.append(int(stat.mean(ADL)))
        else:
            BTC_ADLs.append(ADL)
        if type(slope) == int:
            BTC_ADL_slope.append(float(slope))
        elif type(slope) == list:
            BTC_ADL_slope.append(int(stat.mean(slope)))
        else:
            BTC_ADL_slope.append(slope)
    except Exception as e:
        print(e, o['_id'])
        sys.exit(1)


# In[405]:
p(checkLen(BTC_ADL_Objs, BTC_ADLs))
p(checkLen(BTC_ADL_slope, BTC_ADLs))


# In[406]:
BTC_OBV_Collection = db.BTC_OBV_Data
BTC_OBV_Objs = list(BTC_OBV_Collection.find())
BTC_OBVs = []
BTC_OBV_slope = []
for o in BTC_OBV_Collection.find():
    OBV = o.get('OBV')
    slope = o.get('slope')
    try:
        if type(OBV) == float:
            BTC_OBVs.append(int(OBV))
        elif type(OBV) == list:
            BTC_OBVs.append(int(stat.mean(OBV)))
        else:
            BTC_OBVs.append(ADL)
        if type(slope) == int:
            BTC_OBV_slope.append(float(slope))
        elif type(slope) == list:
            BTC_ADL_slope.append(int(stat.mean(slope)))
        else:
            BTC_OBV_slope.append(slope)
    except Exception as e:
        print(e, o['_id'])
        sys.exit(1)


# In[407]:
p(checkLen(BTC_OBV_Objs, BTC_OBVs))
p(checkLen(BTC_OBV_slope, BTC_OBVs))


# In[408]:

print(f'datetime: {len(BTC_dt_epochs)}\nprices: {len(BTC_prices)}')

collection_lengths = [len(BTC_volumes), len(BTC_highs), len(BTC_lows), len(BTC_ADLs), len(BTC_ADL_slope), len(BTC_OBVs), len(BTC_OBV_slope)]

print(f'Volumes: {len(BTC_RSIs)}\nHighs: {len(BTC_RSIs)}\nLows: {len(BTC_RSIs)}\nRSI: {len(BTC_RSIs)}\nADL_slp: {len(BTC_ADL_slope)}\nOBV_slp: {len(BTC_OBV_slope)}')


# In[409]:


min = collection_lengths[0]
for i in range(1, len(collection_lengths)):
    if collection_lengths[i] < min:
        min = collection_lengths[i]


# In[410]:
# BTC_Data = { 'Datetime': BTC_dt_epochs[0:min], 'Prices': BTC_prices[0:min], 'High': BTC_highs[0:min], 'Low': BTC_lows[0:min] }

BTC_Data = { 'Datetime': BTC_dt_epochs[0:min],
             'Prices': BTC_prices[0:min],
             'Volumes':BTC_volumes[0:min],
             'High':BTC_highs[0:min],
             'Low':BTC_lows[0:min],
             'RSI': BTC_RSIs[0:min],
             'ADL' : BTC_ADLs[0:min],
             'ADL_slope': BTC_ADL_slope[0:min],
             'OBV' : BTC_OBVs[0:min],
             'OBV_slope': BTC_OBV_slope[0:min] }

# In[411]:
keys = BTC_Data.keys()
for k in keys:
    print(f'{k}: {len(BTC_Data[k])}')


# In[412]:
BTC_df = pd.DataFrame(BTC_Data)
BTC_df.head()

# In[413]:
BTC_df.describe()

# In[414]:
BTC_df.info()

# In[415]:
plt.plot( BTC_df.Datetime, BTC_df.Prices)

# In[417]:
BTC_df['Volumes'] = BTC_df.Volumes.astype('float64')
BTC_df['High'] = BTC_df.High.astype('float64')
BTC_df['Low'] = BTC_df.Low.astype('float64')

# In[419]:
# ADL_avg = BTC_df.ADL.mean()
# RSI_avg = BTC_df.RSI.mean()
# ADL_slp_avg = BTC_df.ADL_slope.mean()
# OBV_slp_avg = BTC_df.OBV_slope.mean()

# In[420]:
# values = {'OBV_slope': OBV_slp_avg, 'RSI': RSI_avg, 'ADL_slope': ADL_slp_avg }
# BTC_df = BTC_df.fillna(value=values)
# BTC_df.head()

# In[421]:
BTC_df.info()

# In[423]:
np.random.seed(42)

# In[426]:
X = BTC_df.drop('Prices', axis=1).values
y = BTC_df['Prices'].values
print(X.shape)
print(y.shape)

# In[416]:
plt.plot( X.Datetime, y)

# In[430]:
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1, shuffle=False)

# In[416]:
plt.plot( X_train, y_train)

# In[431]:
scalar = MinMaxScaler(feature_range=(0, 1))
X_train = scalar.fit_transform(X_train)
X_test = scalar.fit_transform(X_test)


# In[432]:
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1],  1))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1],  1))


# In[385]:
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

# #hidden layer
model.add(LSTM(200, return_sequences=True))
model.add(Dropout(0.2))

# hidden layer
model.add(LSTM(200))
model.add(Dropout(0.2))

# output layer
model.add(Dense(1))


# In[346]:
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae', 'mape'])
print(model.summary())


# In[ ]:
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=100, shuffle=False, batch_size=25)


# In[ ]:
model_loss = model.evaluate(X_test, y_test, verbose=0)
print(model_loss)

