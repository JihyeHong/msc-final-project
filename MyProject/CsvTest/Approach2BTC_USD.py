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

# BTC_USD Prediction

# -------------- Data Import --------------
df = pd.read_csv('../data/data_days_USD.csv')

print('--------------Data imported--------------')
print(df.head())

# Pairs
pairs = ['BTC','LTC','ETC','ETH','BCH','XMR','XRP','ZEC','DASH']

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

# -------------- Initialise Models --------------
nb = GaussianNB()
knn = KNeighborsClassifier(n_neighbors = 3)
svm = SVC(kernel='rbf') 

# -------------- Prediction with SVM --------------
print('--------------BTC_USD Prediction with SVM--------------')
coin = 'BTC'

last_index = df.shape[0]-1 
test_size = 100
n_features = 30
dfCoin = pd.DataFrame()
for i in range(1,n_features+1):
    dfCoin[coin+str(i)] = df2[coin+'move'].rolling(min_periods=1,window=i).sum()

pred_list = []
real_list = []
time_list = []
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

        model = svm.fit(X_train,y_train.values.ravel())
        y_pred = model.predict(X_test)

        pred_list.append(y_pred[0])
        real_list.append(df.loc[last_index-test_size+(i+1),[coin+'move']].values[0])
        time_list.append(df.loc[last_index-test_size+(i+1),['TIME']].values[0])
        
        if y_pred[0] == int(df.loc[last_index-test_size+(i+1),[coin+'move']].values[0]):
            correct = correct + 1
            
    accuracy_list.append(correct/test_size)
    print('num of feature: {}, accuracy: {}'.format(j,correct/test_size))
    
# -------------- Prediction with NB --------------
print('--------------BTC_USD Prediction with NB--------------')
accuracy_list2 = []

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

        model = nb.fit(X_train,y_train.values.ravel())
        y_pred = model.predict(X_test)

        pred_list.append(y_pred[0])
        real_list.append(df.loc[last_index-test_size+(i+1),[coin+'move']].values[0])
        time_list.append(df.loc[last_index-test_size+(i+1),['TIME']].values[0])
        
        if y_pred[0] == int(df.loc[last_index-test_size+(i+1),[coin+'move']].values[0]):
            correct = correct + 1
            
    accuracy_list2.append(correct/test_size)
    print('num of feature: {}, accuracy: {}'.format(j,correct/test_size))
    
    
# -------------- Prediction with KNN --------------
print('--------------BTC_USD Prediction with KNN--------------')

accuracy_list3 = []
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

        model = knn.fit(X_train,y_train.values.ravel())
        y_pred = model.predict(X_test)

        pred_list.append(y_pred[0])
        real_list.append(df.loc[last_index-test_size+(i+1),[coin+'move']].values[0])
        time_list.append(df.loc[last_index-test_size+(i+1),['TIME']].values[0])
        
        if y_pred[0] == int(df.loc[last_index-test_size+(i+1),[coin+'move']].values[0]):
            correct = correct + 1
            
    accuracy_list3.append(correct/test_size)
    print('num of feature: {}, accuracy: {}'.format(j,correct/test_size))
    

# -------------- Plot: Prediction results --------------
print('--------------Plot: Prediction results--------------')
fig,ax = plt.subplots(figsize=(10,7))

ax.plot(accuracy_list, label = 'SVM')
ax.plot(accuracy_list2, label = 'NB')
ax.plot(accuracy_list3, label = 'KNN')
ax.set_xlabel("number of features")
ax.set_ylabel("classification rate(%)")
ax.legend(loc='best')
plt.show()

# -------------- Export the predictuion result as a csv file --------------
print('--------------Export the predictuion result as a csv file--------------')
df_accuracy = pd.DataFrame([accuracy_list, accuracy_list2, accuracy_list3])

"""

df_accuracy['SVM'] = accuracy_list
df_accuracy['KNN'] = accuracy_list2
df_accuracy['NB'] = accuracy_list3
"""

df_accuracy.to_csv('output_approach2_BTCUSD.csv')


resultDf = pd.DataFrame(
    {
     'prediction': pred_list,
     'trueMove': real_list,
     'time': time_list
    })

resultDf['trueMove'] = resultDf['trueMove'].astype(int)
resultDf['equal'] = (resultDf.prediction == resultDf.trueMove.astype(int))
#BTC_USD
real_list.count('-1'), real_list.count('1'), real_list.count('0')

print('--------------BTC_USD Prediction with a naive approach--------------')
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

print('Naive model prediction result', sum(compDf.equal), '%')

# Function: plot fonfusion matrix
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

# ------------ Compute confusion matrix ------------
resultDf = resultDf.replace(0,1)
compDf = compDf.replace(0,1)
cnf_matrix = confusion_matrix(resultDf.trueMove.astype(int), compDf.naivePred.astype(int))
np.set_printoptions(precision=2)

# ------------ Plot non-normalized confusion matrix ------------
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=[-1,1],
                      title='Confusion matrix for {}'.format('BTC_USD with a naive model') )

plt.show()
