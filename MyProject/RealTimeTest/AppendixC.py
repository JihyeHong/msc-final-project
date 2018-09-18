#Import basic libraries
import pandas as pd
from pandas import Series,DataFrame
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import requests
import json
import warnings

def get_cryptocompare_data(exchange, fromPair, toPair):
    url = 'https://min-api.cryptocompare.com/data/histoday?fsym={}&tsym={}&limit=365&aggregate=1&e={}'
    response = requests.post(url.format(fromPair, toPair, exchange))
    data = json.loads(response.content)
    df = pd.DataFrame.from_dict(data["Data"], orient="columns")
    
    return pd.DataFrame(df)

# json list to pandas dataframe
price_data = {}
pairs = ['ETH','XLM','EOS','BCH','XRP','ETC','XMR','DASH','LTC','MLN','ZEC','REP','ICN','GNO'] #'BCH','OMG'
for pair in pairs:
    price_data[pair] = get_cryptocompare_data('Kraken', pair, 'BTC')
    
def merge_dfs_on_column(dataframes, labels, col): 
    '''Merge a single column of each dataframe into a new combined dataframe'''
    series_dict = {}
    for index in range(len(dataframes)): #index = rowIndex
        series_dict[labels[index]] = dataframes[index][col]
        
    return pd.DataFrame(series_dict)

# Merge each coin's close price
df = merge_dfs_on_column(list(price_data.values()), list(price_data.keys()), 'close')

print(df.tail())

# Add new columns to compare price between current and the next time unit.
for pair in pairs:
    df[pair+"next"] = df[pair].shift(-1)
    
# Label the next price movevements for each coin
for pair in pairs:
    df[pair+'move'] = '0'
    df.loc[df[pair] <= df[pair+'next'], pair+'move'] = '1'
    df.loc[df[pair] > df[pair+'next'], pair+'move'] = '-1'
    
pair_move_str = []
for pair in pairs:
    pair_move_str.append(pair+'move')

df_moves = df[pair_move_str].copy()
#df_moves = df[['BCHmove', 'ETCmove', 'ETHmove', 'LSKmove', 'LTCmove', 'OMGmove', 'XMRmove', 'XRPmove', 'ZECmove']].copy()
df_moves = df_moves.apply(pd.to_numeric)

from sklearn.utils import class_weight
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, LabelEncoder
 
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB

import itertools
from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

dfset = {} # set of dataframes with sum of movement in different window sizes
for j in (str(x) for x in range(1,11)):
    dfset[j] = pd.DataFrame()
    for pair in pairs:
        df_moves[pair+j] = df_moves[pair+'move'].rolling(min_periods=1,window=int(j)).sum()
        dfset[j][pair+j] = df_moves[pair+j]
        
print('******* Prediction Running, KNN k=3 ******')
knn = KNeighborsClassifier(n_neighbors = 3)

rows = []
columns = ['nrows','feeds', 'scores']

for i in ['100','200','300','365']:
    for j in (str(x) for x in range(1,11)):
        df_here = dfset[j].head(int(i))
        X = df_here.drop('ETH'+j,axis=1)
        Y = dfset['1']['ETH1'].head(int(i))
        
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=8) #18
        
        modelV = knn.fit(X_train, Y_train)
        Y_pred = modelV.predict(X_test)
        score = modelV.score(X_test,Y_test)*100
        
        row = [i,j,score]
        rows.append(row)

acc = pd.DataFrame(rows, columns=columns)

acc['feeds'] = pd.to_numeric(acc['feeds'])

print(acc.tail(10))

# test with KNN = 3
fig,ax = plt.subplots()

for nrow in '100','200','300','365':
    ax.plot(acc[acc.nrows==nrow].feeds,acc[acc.nrows==nrow].scores,label=nrow)

ax.set_xlabel("maximum feature size")
ax.set_ylabel("classification rate(%)")
ax.legend(loc='best')
plt.show()

print('******* Approach 1.1 with various models *******')
print('******* Prediction for ETH_BTC, p=1, sample_size=365 *******')
# Validation Set approach
from sklearn.model_selection import train_test_split
X = dfset['1'].drop('ETH1',axis=1)
#X = dfset['10']
Y = dfset['1']['ETH1']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

accuracyList = {}

clf = tree.DecisionTreeClassifier(class_weight="balanced")
gaussian = GaussianNB()
logreg = LogisticRegression(class_weight="balanced")
boost = GradientBoostingClassifier()
knn = KNeighborsClassifier(n_neighbors = 3)
forest = RandomForestClassifier(n_estimators = 18)
svm = SVC(kernel='linear')
nn = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)

models = [clf,gaussian,logreg,boost,knn,forest, svm, nn]

for model in models:
    accuracyList[model] = 0
    
for model in models:
    modelV = model.fit(X_train, Y_train)
    Y_pred = modelV.predict(X_test)
    accuracyList[model] = round(modelV.score(X_test,Y_test)*100,3)

#models = [clf,gaussian,logreg,boost,knn,forest]
accResult = pd.DataFrame({
    'Model': ['Random Forest', 'KNN', 'Naive Bayes', 
              'Logistic Regression', 'Decision Tree', 'Gradient Boosting', 'svm', 'nn'],
    'Score': [accuracyList[forest], accuracyList[knn], accuracyList[gaussian], 
              accuracyList[logreg], accuracyList[clf], accuracyList[boost], accuracyList[svm], accuracyList[nn]]})
accResult.sort_values(by='Score', ascending=False)

print(accResult.sort_values(by='Score', ascending=False))

print('******* Approach 1.2 with various models *******')
print('******* Prediction for Next ETH_BTC, p=1, sample_size=365 *******')
# Validation Set approach
from sklearn.model_selection import train_test_split
X = dfset['1'].drop('ETH1',axis=1)
#X = dfset['10']
Y = dfset['1']['ETH1'].shift(-1)
Y = Y.fillna(1)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

accuracyList = {}

clf = tree.DecisionTreeClassifier(class_weight="balanced")
gaussian = GaussianNB()
logreg = LogisticRegression(class_weight="balanced")
boost = GradientBoostingClassifier()
knn = KNeighborsClassifier(n_neighbors = 3)
forest = RandomForestClassifier(n_estimators = 18)
svm = SVC(kernel='linear')
nn = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)

models = [clf,gaussian,logreg,boost,knn,forest, svm, nn]

for model in models:
    accuracyList[model] = 0
    
for model in models:
    modelV = model.fit(X_train, Y_train)
    Y_pred = modelV.predict(X_test)
    accuracyList[model] = round(modelV.score(X_test,Y_test)*100,3)

#models = [clf,gaussian,logreg,boost,knn,forest]
accResult = pd.DataFrame({
    'Model': ['Random Forest', 'KNN', 'Naive Bayes', 
              'Logistic Regression', 'Decision Tree', 'Gradient Boosting', 'svm', 'nn'],
    'Score': [accuracyList[forest], accuracyList[knn], accuracyList[gaussian], 
              accuracyList[logreg], accuracyList[clf], accuracyList[boost], accuracyList[svm], accuracyList[nn]]})
accResult.sort_values(by='Score', ascending=False)

print(accResult.sort_values(by='Score', ascending=False))
