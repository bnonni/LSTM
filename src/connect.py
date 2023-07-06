from pymongo import MongoClient
from env import URL
import statistics as stat
from time import strptime, mktime
import pandas as pd
import sys
import re

client = MongoClient(URL)
db = client._wallet

def checkLen(a, b):
    if len(a) == len(b):
        return True
    else:
        return f'DB Objs:{len(a)} < Clean Arr Items:{len(b)}' \
            if len(a) < len(b) else f'Clean Arr Items:{len(b)} < DB Objs:{len(a)}'

def p(o):
    return print(o)

def filterData(coll, st, narr):
    for obj in coll:
        try:
            tmp = obj.get(st)
            narr.append(tmp)
        except Exception as e:
            print(e, obj['_id'])
    return narr

def datetime_converter(dtstr):
    tmstmp = strptime(dtstr, '%Y-%m-%d %H:%M:%S')
    epoch = mktime(tmstmp)
    return int(epoch)

def updateCSV():
    BTC_Tickers_Collection = db.BTC_Tickers
    BTC_Tickers_Objs = list(BTC_Tickers_Collection.find())
    BTC_epochs = []
    BTC_prices = []
    BTC_volumes = []
    BTC_highs = []
    BTC_lows = []
    for obj in BTC_Tickers_Objs:
        dt = re.sub(r'\..*', '', obj.get('time')).replace('T', ' ').rstrip('Z')
        BTC_epochs.append(datetime_converter(dt))
        BTC_prices.append(float(obj.get('price')))
        BTC_volumes.append(obj.get('volume'))
        BTC_highs.append(obj.get('ask'))
        BTC_lows.append(obj.get('bid'))
    for i, e in enumerate(BTC_epochs):
        if i == 0:
            pass
        else:
            BTC_epochs[i] = BTC_epochs[i - 1] + 60
    p(checkLen(BTC_Tickers_Objs, BTC_prices))
    p(checkLen(BTC_epochs, BTC_prices))
    p(checkLen(BTC_Tickers_Objs, BTC_volumes))
    p(checkLen(BTC_Tickers_Objs, BTC_highs))
    p(checkLen(BTC_Tickers_Objs, BTC_lows))

    BTC_RSI_Collection = db.BTC_RSI14_Data
    BTC_RSI_Objs = list(BTC_RSI_Collection.find())
    BTC_RSIs = []
    Errors = []
    for rsio in BTC_RSI_Objs:
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
    if len(Errors) > 0:
        print(Errors)
    p(checkLen(BTC_RSI_Objs, BTC_RSIs))

    BTC_ADL_Collection = db.BTC_ADL_Data
    BTC_ADL_Objs = list(BTC_ADL_Collection.find())
    BTC_ADLs = []
    BTC_ADL_slope = []
    for o in BTC_ADL_Objs:
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
    p(checkLen(BTC_ADL_Objs, BTC_ADLs))
    p(checkLen(BTC_ADL_slope, BTC_ADLs))

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
                BTC_OBVs.append(OBV)
            if type(slope) == int:
                BTC_OBV_slope.append(float(slope))
            elif type(slope) == list:
                BTC_ADL_slope.append(int(stat.mean(slope)))
            else:
                BTC_OBV_slope.append(slope)
        except Exception as e:
            print(e, o['_id'])
            sys.exit(1)
    p(checkLen(BTC_OBV_Objs, BTC_OBVs))
    p(checkLen(BTC_OBV_slope, BTC_OBVs))

    print(f'datetime: {len(BTC_epochs)}\nprices: {len(BTC_prices)}')

    collection_lengths = [len(BTC_volumes), len(BTC_highs),
                          len(BTC_lows), len(BTC_ADLs),
                          len(BTC_ADL_slope), len(BTC_OBVs), len(BTC_OBV_slope)]

    print(f'Volumes: {len(BTC_RSIs)}\nHighs: {len(BTC_RSIs)}\n'
          f'Lows: {len(BTC_RSIs)}\nRSI: {len(BTC_RSIs)}'
          f'\nADL_slp: {len(BTC_ADL_slope)}\nOBV_slp: {len(BTC_OBV_slope)}')

    min = collection_lengths[0]
    for i in range(1, len(collection_lengths)):
        if collection_lengths[i] < min:
            min = collection_lengths[i]

    BTC_Data = {'Datetime': BTC_epochs[0:min],
                 'Prices': BTC_prices[0:min],
                 'Volumes':BTC_volumes[0:min],
                 'High':BTC_highs[0:min],
                 'Low':BTC_lows[0:min],
                 'RSI': BTC_RSIs[0:min],
                 'ADL' : BTC_ADLs[0:min],
                 'ADL_slope': BTC_ADL_slope[0:min],     'OBV' : BTC_OBVs[0:min], 'OBV_slope': BTC_OBV_slope[0:min]}

    BTC_df = pd.DataFrame(BTC_Data)
    BTC_df.to_csv('BTC.csv')


