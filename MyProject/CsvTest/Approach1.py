#Import basic libraries
import pandas as pd
from pandas import Series,DataFrame
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import requests
import json

from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

import itertools
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

# Plot fonfusion matrix
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()


data = '../data/data_days_USD.csv'
# Predict BTC price from other crypto prices at simultaneous time steps
def test_knn(data, timeUnit):
    print('Predict BTC price from other crypto prices in paralell time')
    # Read csv
    df = pd.read_csv(data)
    print(df.head())

    # Add new columns to compare price between current and the next time unit.
    pairs = ['BTC','LTC','ETC','ETH','BCH','XMR','XRP','ZEC','DASH']
    for pair in pairs:
        df[pair+"next"] = df[pair].shift(-1)

    # Label the next price movevements for each coin
    for pair in pairs:
        df[pair+'move'] = '0'
        df.loc[df[pair] <= df[pair+'next'], pair+'move'] = '1'
        df.loc[df[pair] > df[pair+'next'], pair+'move'] = '-1'
        #df.loc[df[pair] == df[pair+'next'], pair+'move'] = '0'

    pair_move_str = []
    for pair in pairs:
        pair_move_str.append(pair+'move')


    df_moves = df[pair_move_str].copy()
    df_moves = df_moves.iloc[:-2]
    df_moves = df_moves.apply(pd.to_numeric)

    df_moves.fillna(0, inplace=True)

    X = df_moves.drop('BTCmove',axis=1)
    y = df_moves['BTCmove']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=12)

    knn = KNeighborsClassifier(n_neighbors = 3)
    gaussian = GaussianNB()

    model = knn.fit(X_train, y_train.values.ravel())
    y_pred = model.predict(X_test)
    accuracy = round(model.score(X_test,y_test)*100,3)

    #scores = cross_val_score(knn, X, y, cv=5)
    #print(scores)
    #print(np.mean(scores))

    # Compute confusion matrix
    cnf_matrix = confusion_matrix(y_test, y_pred)
    np.set_printoptions(precision=2)

    # Plot non-normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=[-1,1],
                          title='Confusion matrix for {}s'.format(timeUnit) )

    plt.show()

    print('       Accuracy for {}s: {}'.format(timeUnit, accuracy))

# Predict BTC price from other crypto prices at the next time step
def test_knn2(data, timeUnit):
    # Read csv
    pred_coin = 'BTC'
    df = pd.read_csv(data)
    print(df.head())

    # Add new columns to compare price between current and the next time unit.
    pairs = ['BTC','LTC','ETC','ETH','BCH','XMR','XRP','ZEC','DASH']
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
    df_moves[pred_coin+'move_next'] = df_moves[pred_coin+'move'].shift(-1)
    df_moves = df_moves.iloc[:-2]
    df_moves = df_moves.apply(pd.to_numeric)

    df_moves.fillna(0, inplace=True)


    X = df_moves.drop(pred_coin+'move_next',axis=1)
    y = df_moves[pred_coin+'move_next']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=12)

    knn = KNeighborsClassifier(n_neighbors = 3)

    model = knn.fit(X_train, y_train.values.ravel())
    y_pred = model.predict(X_test)
    accuracy = round(model.score(X_test,y_test)*100,3)
    
    dfResult = pd.DataFrame({'y_test':y_test,'y_pred':y_pred})
    dfResult.to_csv('output_approach1.csv')

    #scores = cross_val_score(knn, X, y, cv=5)
    #print(scores)
    #print(np.mean(scores))

    # Compute confusion matrix
    cnf_matrix = confusion_matrix(y_test, y_pred)
    np.set_printoptions(precision=2)

    # Plot non-normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=[-1,1],
                          title='Confusion matrix for {}'.format(pred_coin) )

    plt.show()

    print('       Accuracy for {}: {}'.format(pred_coin, accuracy))

print('===== Approach 1.1 for daily data ===== ')
test_knn('../data/data_days_USD.csv', 'day')

print('===== Approach 1.1 for hourly data ===== ')
test_knn('../data/data_hour_USD.csv', 'hour')

print('===== Approach 1.2 for daily data ===== ')
test_knn2('../data/data_days_USD.csv', 'day')

print('===== Approach 1.2 for hourly data ===== ')
test_knn2('../data/data_hour_USD.csv', 'hour')
