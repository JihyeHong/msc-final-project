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

# -------------- Build prediction models --------------
print('--------------GOOGL Prediction for p=4--------------')
stock = 'GOOGL'

last_index = df.shape[0]-1 
test_size = 100
n_features = 4
dfStock = pd.DataFrame()
for i in range(1,n_features+1):
    dfStock[stock+str(i)] = df2[stock+'move'].rolling(min_periods=1,window=i).sum()

print(dfStock.head()) 
    
nb = GaussianNB()
knn = KNeighborsClassifier(n_neighbors = 3)
svm = SVC(kernel='rbf', probability = True) 

train_features = []

for i in range(1,n_features+1):
    train_features.append(stock+str(i))
    
X_test_array = np.array([[0,0,0,0]])
accuracy_list3 = []

#X = dfBTC.drop('BTC1',axis=1)
X = dfStock

train_features = []
for j in range(1,n_features+1):
    correct = 0
    pred_list_NB = []
    pred_list_KNN = []
    pred_list_SVM = []
    real_list = []
    time_list = []
    train_features.append(stock+str(j))
    
    for i in range(0,test_size): #0~99
        
        X_train = X.loc[i:(last_index-test_size+i),train_features]
        y_train = dfStock.loc[i:(last_index-test_size+i),[stock+'1']].shift(-1).fillna(1)

        X_test = X.loc[last_index-test_size+(i+1),train_features].values.reshape(1,-1)
        
        if j == n_features:
            X_test_array = np.append(X_test_array, X_test, 0) 

        modelNB = nb.fit(X_train,y_train.values.ravel())
        modelKNN = knn.fit(X_train,y_train.values.ravel())
        modelSVM = svm.fit(X_train,y_train.values.ravel())
        
        y_pred_NB = modelNB.predict(X_test)
        y_pred_KNN = modelKNN.predict(X_test)
        y_pred_SVM = modelSVM.predict(X_test)

        pred_list_NB.append(y_pred_NB[0])
        pred_list_KNN.append(y_pred_KNN[0])
        pred_list_SVM.append(y_pred_SVM[0])
        
        real_list.append(df.loc[last_index-test_size+(i+1),[stock+'move']].values[0])
        time_list.append(df.loc[last_index-test_size+(i+1),['DATE']].values[0])
        
        if y_pred_NB[0] == int(df.loc[last_index-test_size+(i+1),[stock+'move']].values[0]):
            correct = correct + 1
            #print("pred", y_pred[0])
            #print("real", int(df.loc[last_index-test_size+(i+1),[stock+'move']].values[0]))
            
    accuracy_list3.append(correct/test_size)
    print('NB test result')
    print('num of feature: {}, accuracy: {}'.format(j,correct/test_size))

X_test_array = np.delete(X_test_array, (0), axis=0)

resultDf = pd.DataFrame(
    {
     'predictionNB': pred_list_NB,
     'predictionKNN': pred_list_KNN,
     'predictionSVM': pred_list_SVM,
     'trueMove': real_list,
     'time': time_list
    })

resultDf['trueMove'] = resultDf['trueMove'].astype(int)
resultDf['equal'] = (resultDf.predictionSVM == resultDf.trueMove.astype(int))

print('--------------Plot ROC-AUC --------------')
from sklearn import metrics
print("NB Accuracy", metrics.accuracy_score(resultDf.trueMove, pred_list_NB))   

plt.figure(figsize=(9,7))

y_pred_proba = nb.predict_proba(X_test_array)[::,1]
fpr, tpr, _ = metrics.roc_curve(pred_list_NB,  y_pred_proba)
auc = metrics.roc_auc_score(pred_list_NB, y_pred_proba)
plt.plot(fpr,tpr,label="NB auc="+str('% 6.3f' % auc))

y_pred_proba2 = knn.predict_proba(X_test_array)[::,1]
fpr, tpr, _ = metrics.roc_curve(pred_list_KNN,  y_pred_proba2)
auc2 = metrics.roc_auc_score(pred_list_KNN, y_pred_proba2)
plt.plot(fpr,tpr,label="KNN auc="+str('% 6.3f' % auc2))

y_pred_proba3 = svm.predict_proba(X_test_array)[::,1]
fpr, tpr, _ = metrics.roc_curve(pred_list_SVM,  y_pred_proba3)
auc3 = metrics.roc_auc_score(pred_list_SVM, y_pred_proba3)
plt.plot(fpr,tpr,label="SVM auc="+str('% 6.3f' % auc3))

plt.xlabel("false positive")
plt.ylabel("true positive")
plt.legend(loc=4)
plt.show()