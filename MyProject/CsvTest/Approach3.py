#Import basic libraries
import pandas as pd
from pandas import Series,DataFrame
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from sklearn.neighbors import KNeighborsClassifier

import itertools
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

def merge_dfs_on_column(dataframes, labels, col): 
    '''Merge a single column of each dataframe into a new combined dataframe'''
    series_dict = {}
    for index in range(len(dataframes)): #index = rowIndex
        series_dict[labels[index]] = dataframes[index][col]
        
    return pd.DataFrame(series_dict)

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
    

def test_knn(pred_stock, stock_list, days):
    stock_data = {}

    num_of_days = days

    for stock in stock_list:
        stock_data[stock] = pd.read_csv(('../data/stock/{}.txt').
                                    format(stock)).sort_values('Date',ascending=False)
        stock_data[stock] = stock_data[stock].head(num_of_days).sort_values('Date')
        stock_data[stock] = stock_data[stock].reset_index(drop=True)

    # Merge each price's close price
    df = merge_dfs_on_column(list(stock_data.values()), list(stock_data.keys()), 'Close')
    print(df.tail(10))
    
    # Add new columns to compare price between current and the next time unit.
    for stock in stock_list:
        df[stock+"next"] = df[stock].shift(-1)

    # Label the next price movevements for each coin
    for stock in stock_list:
        df[stock+'move'] = '0'
        df.loc[df[stock] <= df[stock+'next'], stock+'move'] = '1'
        df.loc[df[stock] > df[stock+'next'], stock+'move'] = '-1'
        #df.loc[df[pair] == df[pair+'next'], pair+'move'] = '0'

    move_str_list = []
    for stock in stock_list:
        move_str_list.append(stock+'move')
    
    
    df_moves = df[move_str_list].copy()
    #df_moves['BTCmove_next'] = df_moves['BTCmove'].shift(-1)
    df_moves = df_moves.iloc[:-2]
    df_moves = df_moves.apply(pd.to_numeric)

    df_moves.fillna(0, inplace=True)
    
    X = df_moves.drop(pred_stock+'move',axis=1)
    y = df_moves[pred_stock+'move']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=12)

    knn = KNeighborsClassifier(n_neighbors = 3)
    
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
                          title='Confusion matrix for {}'.format(pred_stock) )

    plt.show()
    
    print('       Accuracy for {}: {}'.format(pred_stock, accuracy))
    
    
def test_knn2(pred_stock, stock_list, days):
    stock_data = {}

    num_of_days = days

    for stock in stock_list:
        stock_data[stock] = pd.read_csv(('../data/stock/{}.txt').
                                    format(stock)).sort_values('Date',ascending=False)
        stock_data[stock] = stock_data[stock].head(num_of_days).sort_values('Date')
        stock_data[stock] = stock_data[stock].reset_index(drop=True)
    
    # Merge each price's close price
    df = merge_dfs_on_column(list(stock_data.values()), list(stock_data.keys()), 'Close')
    
    print(df.tail(10))
    
    # Add new columns to compare price between current and the next time unit.
    for stock in stock_list:
        df[stock+"next"] = df[stock].shift(-1)

    # Label the next price movevements for each coin
    for stock in stock_list:
        df[stock+'move'] = '0'
        df.loc[df[stock] <= df[stock+'next'], stock+'move'] = '1'
        df.loc[df[stock] > df[stock+'next'], stock+'move'] = '-1'
        #df.loc[df[pair] == df[pair+'next'], pair+'move'] = '0'

    move_str_list = []
    for stock in stock_list:
        move_str_list.append(stock+'move')
    
    
    df_moves = df[move_str_list].copy()
    df_moves[pred_stock+'move_next'] = df_moves[pred_stock+'move'].shift(-1)
    df_moves = df_moves.iloc[:-2]
    df_moves = df_moves.apply(pd.to_numeric)

    df_moves.fillna(0, inplace=True)
    
    #X = df_moves.drop(pred_stock+'move',axis=1)
    #y = df_moves[pred_stock+'move']
    
    X = df_moves.drop(pred_stock+'move_next',axis=1)
    y = df_moves[pred_stock+'move_next']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=12)

    knn = KNeighborsClassifier(n_neighbors = 3)
    
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
                          title='Confusion matrix for {}'.format(pred_stock) )

    plt.show()
    
    print('       Accuracy for {}: {}'.format(pred_stock, accuracy))
    
print('******* Prediction Running *******')
stock_list = ['GOOGL', 'MSFT', 'FB', 'T', 'INTC', 'VZ', 'TSM','CSCO','ORCL','CHL'] #Tech Industry
days = 365
print('===== Approach 1.1 for GOOGL daily data ===== ')
test_knn('GOOGL', stock_list, days)
print('===== Approach 1.1 for MSFT daily data ===== ')
test_knn('MSFT',stock_list, days)
print('===== Approach 1.1 for ORCL daily data ===== ')
test_knn('ORCL', stock_list, days)

print('===== Approach 1.2 for GOOGL daily data ===== ')
test_knn2('GOOGL', stock_list, days)
print('===== Approach 1.2 for MSFT daily data ===== ')
test_knn2('MSFT', stock_list, days)
print('===== Approach 1.2 for ORCL daily data ===== ')
test_knn2('ORCL', stock_list, days)

print('===== Approach 1.1 for GOOGL with utility stocks ===== ')
#Google along with utility stocks
stock_list2 = ['GOOGL', 'NEE', 'DUK', 'SO', 'D', 'EXC', 'NGG', 'AEP', 'SRE', 'OKE', 'PEG', ] 
test_knn('GOOGL', stock_list2, days)
print('===== Approach 1.2 for GOOGL with utility stocks ===== ')
test_knn2('GOOGL', stock_list2, days)
