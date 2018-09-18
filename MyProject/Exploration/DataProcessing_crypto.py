#Import basic libraries
import pandas as pd
from pandas import Series,DataFrame
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import requests
import json

# * Data Collection *
# function: call CryptoCompare APIs for historical price data
# timeUnit: minute, hour, day
# fromPair, toPair: a coin or a fiat
# limit: number of rows, maximum 2000
# exchange: a crypto exchange, eg> Kraken, Poloniex, etc
def get_cryptocompare_data(timeUnit, fromPair, toPair, limit, exchange):
    url = 'https://min-api.cryptocompare.com/data/histo{}?fsym={}&tsym={}&limit={}&aggregate=1&e={}'
    response = requests.post(url.format(timeUnit, fromPair, toPair, limit, exchange))
    data = json.loads(response.content)
    df = pd.DataFrame.from_dict(data["Data"], orient="columns")

    return pd.DataFrame(df)

# function: merge difference crypto prices into one DataFrame
def merge_dfs_on_column(dataframes, labels, col):
    '''Merge a single column of each dataframe into a new combined dataframe'''
    series_dict = {}
    for index in range(len(dataframes)): #index = rowIndex
        series_dict[labels[index]] = dataframes[index][col]

    return pd.DataFrame(series_dict)

# Raw json lists to pandas dataframe =============
# json list to pandas dataframe
price_data = {}
pairs = ['BTC','LTC','ETC','ETH','BCH','XMR','XRP','ZEC','DASH'] #'REP','GNO','EOS','XLM'
for pair in pairs:
    price_data[pair] = get_cryptocompare_data('day',pair, 'USD', 365, 'Kraken')

# Merge each coin's close price
df = merge_dfs_on_column(list(price_data.values()), list(price_data.keys()), 'close')
print(df.head())

# * Exploratory Analysis *
# Plot: historical prices ==================
f, ax = plt.subplots(figsize=(15, 9.7))
time = pd.to_datetime(price_data['BTC'].time,unit='s')

for pair in pairs:
    plt.plot(time, df[pair], label=pair)

plt.xlabel('Time', fontsize=18)
plt.ylabel('Price(USD)', fontsize=16)
plt.title('Historical prices',fontsize=20)
plt.legend(loc='best')
plt.show()

# Plot: historical prices in log ===========
f, ax = plt.subplots(figsize=(15, 10))
time = pd.to_datetime(price_data['BTC'].time,unit='s')

for pair in pairs:
    plt.plot(time, df[pair], label=pair)

plt.xlabel('Time', fontsize=18)
plt.ylabel('Price in USD', fontsize=16)
plt.title('Historical prices in log',fontsize=20)
plt.legend(loc='best')

ax.set_xscale("log")
ax.set_yscale("log")
plt.show()

# Price of currencies in a box plot
f, ax = plt.subplots(figsize=(20, 10))
df.boxplot()
plt.title('Prices')
plt.show()

# Voluje of currencies in a box plot
df_volume = merge_dfs_on_column(list(price_data.values()), list(price_data.keys()), 'volumeto')
f, ax = plt.subplots(figsize=(20, 10))
df_volume.boxplot()
plt.title('Volume')
plt.show()

# Plot: correlation of prices ===========
df[pairs].corr()
corrmat = df[pairs].corr()
f,ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, vmax=1, square=True);
plt.title('Correlation of prices',fontsize=20)
plt.show()


# * Data Processing *
# Shift price data and label each coin's next price column
# Add new columns to compare price between current and the next time unit.
for pair in pairs:
    df[pair+"next"] = df[pair].shift(-1)

print(df.columns)

# Label the next price movevements for each coin
for pair in pairs:
    df[pair+'move'] = '0'
    df.loc[df[pair] <= df[pair+'next'], pair+'move'] = '1'
    df.loc[df[pair] > df[pair+'next'], pair+'move'] = '-1'

pair_move_str = []
for pair in pairs:
    pair_move_str.append(pair+'move')

df_moves = df[pair_move_str].copy()
df_moves = df_moves.apply(pd.to_numeric)
df_moves.fillna(0, inplace=True)

df_moves.head(5)
print(df_moves.head())


# Plot: number of classes in the dataset ===========
f, ax = plt.subplots(figsize=(20, 7))

sns.countplot(x="variable", hue="value", data=pd.melt(df_moves[:-1]))
plt.title('The number of classes in the dataset',fontsize=20)
plt.show()
