#import
import pandas as pd
from pandas import Series,DataFrame
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.naive_bayes import GaussianNB

import itertools
from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

# -------------- Data Import --------------
#data import
stock_data = {}
stock_list = ['GOOGL']
num_of_days = 365

for stock in stock_list:
    stock_data[stock] = pd.read_csv(('../data/stock/{}.txt').
                                    format(stock)).sort_values('Date',ascending=False)
    stock_data[stock] = stock_data[stock].head(num_of_days).sort_values('Date')
    stock_data[stock] = stock_data[stock].reset_index(drop=True)


print('--------------Data imported--------------')
print(stock_data['GOOGL'].tail())

# Function to merge dataframes into a single dataframe
def merge_dfs_on_column(dataframes, labels, col): 
    '''Merge a single column of each dataframe into a new combined dataframe'''
    series_dict = {}
    for index in range(len(dataframes)): #index = rowIndex
        series_dict[labels[index]] = dataframes[index][col]
        
    return pd.DataFrame(series_dict)

df = merge_dfs_on_column(list(stock_data.values()), list(stock_data.keys()), 'Close')
# Add Time Column converted to a readable datetime format from timestamp
df['DATE'] = stock_data['GOOGL'].Date
df['DATE'] = pd.to_datetime(df['DATE'])
    
print(df.tail(10))

for stock in stock_list:
    df[stock+"next"] = df[stock].shift(-1)
    
    df[stock+'move'] = '0'
    df.loc[df[stock] <= df[stock+'next'], stock+'move'] = '1'
    df.loc[df[stock] > df[stock+'next'], stock+'move'] = '-1'
    #df.loc[df[pair] == df[pair+'next'], pair+'move'] = '0'
    
stock_move_column_list = [ stock+'move' for stock in stock_list ] 
df2 = df[stock_move_column_list].copy()
df2 = df2.apply(pd.to_numeric)

print('--------------Price movements extracted--------------')
print(df2.head())
print(df2.apply(pd.value_counts))

# -------------- Plot: The number of classes in features --------------
f, ax = plt.subplots(figsize=(4, 7))
sns.countplot(x="variable", hue="value", data=pd.melt(df2[:-1]))
plt.title('The number of classes')
plt.show()

# -------------- Initialise Models --------------
nb = GaussianNB()
knn = KNeighborsClassifier(n_neighbors = 3)
svm = SVC(kernel='rbf') 

stock = 'GOOGL'

last_index = df.shape[0]-1 
test_size = 100
n_features = 4
dfStock = pd.DataFrame()
for i in range(1,n_features+1):
    dfStock[stock+str(i)] = df2[stock+'move'].rolling(min_periods=1,window=i).sum()

print('***********Forecasting***********')
nb = GaussianNB()
knn = KNeighborsClassifier(n_neighbors = 3)
svm = SVC(kernel='rbf') 

#pred_list = []
#real_list = []
#time_list = []
accuracy_list3 = []

#X = dfBTC.drop('BTC1',axis=1)
X = dfStock

train_features = []
for j in range(1,n_features+1):
    correct = 0
    pred_list = []
    real_list = []
    time_list = []
    train_features.append(stock+str(j))
    
    for i in range(0,test_size): #0~99
        
        X_train = X.loc[i:(last_index-test_size+i),train_features]
        y_train = dfStock.loc[i:(last_index-test_size+i),[stock+'1']].shift(-1).fillna(1)

        X_test = X.loc[last_index-test_size+(i+1),train_features].values.reshape(1,-1)

        model = nb.fit(X_train,y_train.values.ravel())
        y_pred = model.predict(X_test)

        pred_list.append(y_pred[0])
        real_list.append(df.loc[last_index-test_size+(i+1),[stock+'move']].values[0])
        time_list.append(df.loc[last_index-test_size+(i+1),['DATE']].values[0])
        
        if y_pred[0] == int(df.loc[last_index-test_size+(i+1),[stock+'move']].values[0]):
            correct = correct + 1
            #print("pred", y_pred[0])
            #print("real", int(df.loc[last_index-test_size+(i+1),[stock+'move']].values[0]))
            
    accuracy_list3.append(correct/test_size)
    print('num of feature: {}, accuracy: {}'.format(j,correct/test_size))
    
    
resultDf = pd.DataFrame(
    {
     'prediction': pred_list,
     'trueMove': real_list,
     'time': time_list
    })

resultDf['trueMove'] = resultDf['trueMove'].astype(int)
resultDf['equal'] = (resultDf.prediction == resultDf.trueMove.astype(int))

resultDf.shape

# Error of prediction
num_of_error = len(resultDf.loc[~(resultDf['prediction'] == resultDf['trueMove'])])
print('***********Naive Bayes Model with p=4***********')
print('Model Accuracy:',1-(num_of_error/test_size))

#GOOGL
real_list.count('-1'), real_list.count('1'), real_list.count('0')

naive_pred = resultDf.trueMove.shift(1)

compDf = pd.DataFrame(
    {
     'naivePred': naive_pred,
     'trueMove': real_list,
     'time': time_list
    })

compDf = compDf.fillna(0)
compDf['equal'] = (compDf.naivePred.astype(int) == resultDf.trueMove.astype(int))

print('Naive model result ', sum(compDf.equal), '%')

compDf['pred'] = pred_list
print(compDf.head())

compDf['naive_correct'] = (compDf.naivePred.astype(int) == compDf.trueMove.astype(int))
compDf['model_correct'] = (compDf.pred.astype(int) == compDf.trueMove.astype(int))
compDf['naive_model_equal'] = (compDf.naive_correct == compDf.model_correct)

#GOOGL
print('***********Calculate the P-Value***********')
print(sum(compDf['naive_correct']), sum(compDf['model_correct']), sum(compDf['naive_model_equal']))
print('A+B=',sum(compDf['naive_correct']))
print('A+C=',sum(compDf['model_correct']))
print('A+D=',sum(compDf['naive_model_equal']))
print('B+C=',test_size-sum(compDf['naive_model_equal']))

# Calculate the P-Value
from scipy.stats import binom_test
#alternative : {‘two-sided’, ‘greater’, ‘less’}

a = binom_test(0, 32, 1/2, 'less')
print('p-value = {0:.30f}'.format(a))

#a = binom_test(32, 32, 1/2, 'greater')
#print('p-value = {0:.30f}'.format(a))
