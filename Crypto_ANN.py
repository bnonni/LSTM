#!/usr/bin/env python
# coding: utf-8

# In[10]:


from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
import tensorflow as tf
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from numpy.random import seed
from url import URL
from pymongo import *
import statistics as stat
import numpy as np
import pandas as pd
from time import strptime, strftime, mktime, gmtime
import gc
import sys
import re
gc.collect()

client = MongoClient(URL)
db = client.crypto_wallet

# In[14]:


def checkLen(a, b):
    if len(a) == len(b):
        return True
    else:
        return f'a:{len(a)} < b:{len(b)}' if len(a) < len(b) else f'b:{len(b)} < a:{len(a)}'


# In[15]:
def filterData(obj, coll, st, narr):
    for obj in coll:
        try:
            tmp = obj.get(st)
            narr.append(tmp)
        except Exception as e:
            print(e, o['_id'])
    return narr


# In[16]:
def datetime_converter(dtstr):
    tmstmp = strptime(dtstr, '%Y-%m-%d %H:%M:%S')
    epoch = mktime(tmstmp)
    return int(epoch)


# In[17]:
BTC_Tickers_Collection = db.BTC_Tickers
BTC_Tickers_Objs = list(BTC_Tickers_Collection.find())
BTC_dt_epochs = []
BTC_prices = []
for obj in BTC_Tickers_Collection.find():
    date_time = obj.get('time')
    dt = re.sub(r'\..*', '', date_time).replace('T', ' ').rstrip('Z')
    epoch = datetime_converter(dt)
    BTC_dt_epochs.append(epoch)
    price = obj.get('price')
    BTC_prices.append(float(price))


# In[18]:
checkLen(BTC_Tickers_Objs, BTC_prices)

# In[19]:
checkLen(BTC_dt_epochs, BTC_prices)

# In[20]:
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
        sys.exit


# In[21]:
checkLen(BTC_RSI_Objs, BTC_RSIs)

# In[22]:
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


# In[23]:
checkLen(BTC_ADL_Objs, BTC_ADLs)


# In[24]:
checkLen(BTC_ADL_slope, BTC_ADLs)


# In[25]:
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

# In[26]:
checkLen(BTC_OBV_Objs, BTC_OBVs)

# In[27]:
checkLen(BTC_OBV_slope, BTC_OBVs)

# In[28]:
print(f'datetime: {len(BTC_dt_epochs)}\nprices: {len(BTC_prices)}')

# In[33]:
collection_lengths = [len(BTC_ADLs), len(
    BTC_ADL_slope), len(BTC_OBVs), len(BTC_OBV_slope)]
print(f'RSI: {len(BTC_RSIs)}\nADL: {len(BTC_ADLs)}\nslp: {len(BTC_ADL_slope)}\nOBV: {len(BTC_OBVs)}\nslp: {len(BTC_OBV_slope)}')

min = collection_lengths[0]
for i in range(1, len(collection_lengths)):
    if collection_lengths[i] < min:
        min = collection_lengths[i]

# In[36]:
BTC_Data = {'Datetime': BTC_dt_epochs[0:min], 'Prices': BTC_prices[0:min], 'RSI': BTC_RSIs[0:min],
            'ADL': BTC_ADLs, 'ADL_slope': BTC_ADL_slope, 'OBV': BTC_OBVs[0:min], 'OBV_slope': BTC_OBV_slope[0:min]}

# In[37]:
print(len(BTC_Data['Datetime']), len(BTC_Data['Prices']), len(BTC_Data['RSI']), len(
    BTC_Data['ADL']), len(BTC_Data['ADL_slope']), len(BTC_Data['OBV']), len(BTC_Data['OBV_slope']))

# In[38]:
BTC_df = pd.DataFrame(BTC_Data)
BTC_df.head()

# In[39]:
BTC_df.describe()

# In[40]:
BTC_df.info()

# In[42]:
# ADL_avg = BTC_df.ADL.mean()
RSI_avg = BTC_df.RSI.mean()
# ADL_slp_avg = BTC_df.ADL_slope.mean()
OBV_slp_avg = BTC_df.OBV_slope.mean()

# In[43]:
# values = {'RSI': RSI_avg, 'ADL': ADL_avg, 'ADL_slope': ADL_slp_avg, 'OBV_slope': OBV_slp_avg}
values = {'OBV_slope': OBV_slp_avg, 'RSI': RSI_avg}
BTC_df = BTC_df.fillna(value=values)
BTC_df.head()


# In[44]:
BTC_df.info()

# In[46]:
seed(42)

# In[47]:
X = BTC_df.drop('Prices', axis=1).values
y = BTC_df['Prices'].values
print(X.shape)
print(y.shape)

# In[48]:
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=1)

# In[49]:
scalar = MinMaxScaler(feature_range=(0, 1))
X_train = scalar.fit_transform(X_train)
X_test = scalar.fit_transform(X_test)

# In[50]:
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# In[51]:
model = Sequential()

model.add(LSTM(250, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model.add(Dropout(0.4))

model.add(LSTM(250, return_sequences=True))
model.add(Dropout(0.4))

model.add(LSTM(250, return_sequences=True))
model.add(Dropout(0.4))

model.add(LSTM(250))
model.add(Dropout(0.4))

# try adding more layers
model.add(Dense(1, activation=tf.nn.relu))

# In[54]:
model.compile(optimizer='adam', loss='mean_squared_error')
# In[125]:
model.fit(X_train, y_train, epochs=150, batch_size=32)

# In[86]:
model_loss = model.evaluate(X_test, y_test, verbose=2)
print(f"Loss: {model_loss} = {round(model_loss)}")
