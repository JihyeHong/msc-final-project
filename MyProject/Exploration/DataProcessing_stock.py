#Import basic libraries
import pandas as pd
from pandas import Series,DataFrame
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import requests
import json

# function: merge difference crypto prices into one DataFrame
def merge_dfs_on_column(dataframes, labels, col): 
    '''Merge a single column of each dataframe into a new combined dataframe'''
    series_dict = {}
    for index in range(len(dataframes)): #index = rowIndex
        series_dict[labels[index]] = dataframes[index][col]
        
    return pd.DataFrame(series_dict)

#data import
stock_data = {}

stock_list = ['GOOGL', 'MSFT', 'FB', 'T', 'INTC', 'VZ', 'TSM','CSCO','ORCL','CHL', ] #Tech Industry
stock_list2 = ['NEE', 'DUK', 'SO', 'D', 'EXC', 'NGG', 'AEP', 'SRE', 'OKE', 'PEG' ] #Utility Industry
total_stock_list = stock_list + stock_list2

num_of_days = 365

for stock in total_stock_list:
    stock_data[stock] = pd.read_csv(('../data/stock/{}.txt').
                                    format(stock)).sort_values('Date',ascending=False)
    stock_data[stock] = stock_data[stock].head(num_of_days).sort_values('Date')
    stock_data[stock] = stock_data[stock].reset_index(drop=True)
    
print(stock_data['GOOGL'].tail())


# Merge datasets 
df = merge_dfs_on_column(list(stock_data.values()), list(stock_data.keys()), 'Close')

# Plot: correlation of prices in tech industry ===========
df[stock_list].corr()
corrmat = df[stock_list].corr()
f,ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, vmax=1, square=True);
plt.title('Correlation of stocks in Tech industry',fontsize=20)
plt.show()

# Plot: correlation of prices in utility industry ===========
df[stock_list2].corr()
corrmat = df[stock_list2].corr()
f,ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, vmax=1, square=True);
plt.title('Correlation of stocks in Utility industry',fontsize=20)
plt.show()

# Plot: correlation of prices in tech & utility industry ===========
df[stock_list].corr()
corrmat = df[total_stock_list].corr()
f,ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, vmax=1, square=True);
plt.title('Correlation of stocks in tech & utility industry',fontsize=20)
plt.show()

# Add Time Column converted to a readable datetime format from timestamp
df['DATE'] = stock_data['GOOGL'].Date
df['DATE'] = pd.to_datetime(df['DATE'])
    
print(df.tail(10))


for stock in total_stock_list:
    df[stock+"next"] = df[stock].shift(-1)
    
    df[stock+'move'] = '0'
    df.loc[df[stock] <= df[stock+'next'], stock+'move'] = '1'
    df.loc[df[stock] > df[stock+'next'], stock+'move'] = '-1'
    
stock_move_column_list = [ stock+'move' for stock in stock_list ] 
stock_move_column_list2 = [ stock+'move' for stock in stock_list2 ] 
dfTech = df[stock_move_column_list].copy()
dfTech = dfTech.apply(pd.to_numeric)
dfUtil = df[stock_move_column_list2].copy()
dfUtil = dfUtil.apply(pd.to_numeric)

print("Tech Stocks")
print(dfTech.tail())

print("Utility Stocks")
print(dfUtil.tail())

dfTech.apply(pd.value_counts)
dfUtil.apply(pd.value_counts)

# Plot: the number of classes in the dataset
f, ax = plt.subplots(figsize=(20, 7))
sns.countplot(x="variable", hue="value", data=pd.melt(dfTech[:-1]))
plt.title('The number of classes in tech stocks', fontsize=20)
plt.show()

# Plot: the number of classes in the dataset
f, ax = plt.subplots(figsize=(20, 7))
sns.countplot(x="variable", hue="value", data=pd.melt(dfUtil[:-1]))
plt.title('The number of classes in tech stocks', fontsize=20)
plt.show()


