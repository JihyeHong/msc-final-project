#import
import pandas as pd
from pandas import Series,DataFrame
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

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

import requests
import json

# BTC_USD Prediction

# -------------- Data Import --------------
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

# -------------- Prediction with ML --------------
def predictMove(pred_coin, base_coin, MLmodel, input_size):
    
    # Raw json lists to pandas dataframe =============
    # json list to pandas dataframe
    price_data = {}
    pairs = ['ETH','LTC','ETC','BCH','XMR','XRP','ZEC','DASH'] #'REP','GNO','EOS','XLM'
    for pair in pairs:
        price_data[pair] = get_cryptocompare_data('day', pair, base_coin, 365, 'Kraken')

    # Merge each coin's close price
    df = merge_dfs_on_column(list(price_data.values()), list(price_data.keys()), 'close')
    df['TIME'] = pd.to_datetime(price_data[pred_coin].time,unit='s')
    print(df.tail())

    # -------------- Data Processing --------------
    for pair in pairs:
        df[pair+"next"] = df[pair].shift(-1)

        df[pair+'move'] = '0'
        df.loc[df[pair] <= df[pair+'next'], pair+'move'] = '1'
        df.loc[df[pair] > df[pair+'next'], pair+'move'] = '-1'
        #df.loc[df[pair] == df[pair+'next'], pair+'move'] = '0'
    
    pair_move_column_list = [ pair+'move' for pair in pairs ] 
    df2 = df[pair_move_column_list].copy()
    df2 = df2.apply(pd.to_numeric)

    df2.apply(pd.value_counts)

    print('--------------Price movements extracted--------------')
    print(df2.head())

    # -------------- Plot: The number of classes in features --------------
    f, ax = plt.subplots(figsize=(20, 7))
    sns.countplot(x="variable", hue="value", data=pd.melt(df2[:-1]))
    plt.title('The number of classes in features')
    plt.show()


    print('--------------Prediction--------------')
    print('Coin = ',pred_coin)
    print('ML Model = ',MLmodel)
    coin = pred_coin

    last_index = df.shape[0]-1 
    test_size = 100
    n_features = input_size
    dfCoin = pd.DataFrame()
    for i in range(1,n_features+1):
        dfCoin[coin+str(i)] = df2[coin+'move'].rolling(min_periods=1,window=i).sum()

    accuracy_list = []

    #X = dfBTC.drop('BTC1',axis=1)
    X = dfCoin

    train_features = []
    for j in range(1,n_features+1):
        correct = 0
        pred_list = []
        real_list = []
        time_list = []
        train_features.append(coin+str(j))

        for i in range(0,test_size): #0~99
            X_train = X.loc[i:(last_index-test_size+i),train_features]
            y_train = dfCoin.loc[i:(last_index-test_size+i),[coin+'1']].shift(-1).fillna(1)

            X_test = X.loc[last_index-test_size+(i+1),train_features].values.reshape(1,-1)

            model = MLmodel.fit(X_train,y_train.values.ravel())
            y_pred = model.predict(X_test)

            pred_list.append(y_pred[0])
            real_list.append(df.loc[last_index-test_size+(i+1),[coin+'move']].values[0])
            time_list.append(df.loc[last_index-test_size+(i+1),['TIME']].values[0])

            if y_pred[0] == int(df.loc[last_index-test_size+(i+1),[coin+'move']].values[0]):
                correct = correct + 1

        accuracy_list.append(correct/test_size)
        print('num of feature: {}, accuracy: {}'.format(j,correct/test_size))
        
        
    resultDf = pd.DataFrame(
        {
         'prediction': pred_list,
         'trueMove': real_list,
         'time': time_list
        })

    resultDf['trueMove'] = resultDf['trueMove'].astype(int)
    resultDf['equal'] = (resultDf.prediction == resultDf.trueMove.astype(int))

    #number of classes
    real_list.count('-1'), real_list.count('1'), real_list.count('0')

    naive_pred = resultDf.trueMove.shift(1)

    compDf = pd.DataFrame(
        {
         'prediction': pred_list,
         'naivePred': naive_pred,
         'trueMove': real_list,
         'time': time_list
        })

    compDf = compDf.fillna(0)
    compDf['equal'] = (compDf.naivePred.astype(int) == resultDf.trueMove.astype(int))

    print(resultDf)
    print(compDf)
    print('Naive approach result: ', sum(compDf.equal), '%')


# Models: nb, knn, svm, clf, logreg, boost, forest, nn
nb = GaussianNB() # Naive Bayes
knn = KNeighborsClassifier(n_neighbors = 3) # 3 KNNs
svm = SVC(kernel='rbf') # SVM with Gaussian kernel
clf = tree.DecisionTreeClassifier(class_weight="balanced") # Decision Tree
logreg = LogisticRegression(class_weight="balanced") # Logistic Regression
boost = GradientBoostingClassifier() # Gradiant Boost
forest = RandomForestClassifier(n_estimators = 18) # Random Forest
nn = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1) # Neural Network

# predictMove(target_coin, base_coin, MLmodel, max_feature_size)
predictMove('ETH','BTC',svm,5) 


