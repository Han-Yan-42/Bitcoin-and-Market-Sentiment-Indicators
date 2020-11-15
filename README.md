# Bitcoin-and-Market-Sentiment-Indicators
This project manage to explore relationship between bitcoin price and market sentiment indicator derived from tweets and to build a LSTM RNN model to predict future price

import tensorflow as tf
import numpy as np
import pandas as pd
import pyspark as spark
import sklearn as sk
import seaborn as sns
import os
import datetime
%matplotlib inline
import matplotlib as mpl
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Dropout
from numpy import concatenate
from sklearn.preprocessing import MinMaxScaler
from math import sqrt
from sklearn.metrics import mean_squared_error
from keras.constraints import max_norm

csv_path = '/Users/hanyan/Desktop/DSO428Project/df.csv'
df = pd.read_csv(csv_path,sep=";")
df = df.set_index('Date')
df.index = pd.to_datetime(df.index)
df_null = pd.isnull(df)
df_null = df[df_null == True]
df.dropna(inplace=True)
df=df[['Close', 'Compound_Score', 'Count_Negatives',
       'Count_Positives', 'Count_Neutrals']]
df

close = df['Close'].resample('D',label='right').mean()
df = df.resample('D').sum()
df = df.drop(columns=['Close'])
df = pd.concat([close,df], axis=1)

df_null = pd.isnull(df)
df_null = df[df_null == True]
df.dropna(inplace=True)
df

df_train = df.iloc[0:430]
df_test = df.iloc[430:]
df_train

dataset = df.values
scaler = MinMaxScaler(feature_range=(-1, 1))
dataset = scaler.fit_transform(dataset)
train_size = int(len(dataset) * 0.80)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]

train_X, train_y = train[:, 1:10], train[:,:1] # Determining multivariates and label
test_X, test_y = test[:,1:10], test[:,:1] # Determining multivariates and label
print(train_X.shape, train_y.shape, test_X.shape, test_y.shape) #Determining train_X,train_y,test_X,test_y

# Shaping data
train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1])) 
test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)

opt = keras.optimizers.Adam(learning_rate=0.01)
model = Sequential()

model.add(LSTM(250, return_sequences = True,input_shape=(train_X.shape[1], train_X.shape[2])))# Determining # of Neural Nodes
model.add(Dense(1))
model.add(Dropout(0.1))

model.compile(loss='mape', optimizer=opt)
model.summary()

MTS_RNN = model.fit(train_X, train_y, epochs=2000, batch_size=72, validation_data=(test_X, test_y), verbose=2, shuffle= False)

plt.plot(MTS_RNN.history['loss'], label='train')
plt.plot(MTS_RNN.history['val_loss'], label='test')
plt.legend()
plt.show()

#Prediction by Multivariate Time Series through NN

train_predict = model.predict(train_X)    
test_predict = model.predict(test_X) 

#Converting from three dimension to two dimension

train_X = train_X.reshape((train_X.shape[0], train_X.shape[2]))
test_X = test_X.reshape((test_X.shape[0], test_X.shape[2]))

#Reshaping

train_predict = train_predict.reshape((train_X.shape[0],1))
test_predict = test_predict.reshape((test_X.shape[0],1))

#Concatenate

inv_train_predict = concatenate((train_predict, train_X), axis=1)
inv_test_predict = concatenate((test_predict, test_X), axis=1)

#Transforming to original scale

inv_train_predict = scaler.inverse_transform(inv_train_predict)
inv_test_predict = scaler.inverse_transform(inv_test_predict) 

#Predicted values on training data
inv_train_predict = inv_train_predict[:,0]
inv_train_predict
#Predicted values on testing data
inv_test_predict = inv_test_predict[:,0]
inv_test_predict

train_y = train_y.reshape((len(train_y), 1))
inv_train_y = concatenate((train_y, train_X), axis=1)
inv_train_y = scaler.inverse_transform(inv_train_y)
inv_train_y = inv_train_y[:,0]
test_y = test_y.reshape((len(test_y), 1))
inv_test_y = concatenate((test_y, test_X), axis=1)
inv_test_y = scaler.inverse_transform(inv_test_y)
inv_test_y = inv_test_y[:,0]
inv_test_y = inv_test_y.reshape(-1,1)
inv_test_y.shape

t = np.arange(0,108,1)

plt.plot(t,inv_test_y,label="actual")
plt.plot(t,inv_test_predict,'r',label="predicted")
plt.show()
plt.figure(figsize=(25, 10))
plt.plot(df_train.index, inv_train_predict,label="actual")
plt.plot(df_test.index, inv_test_predict, color='r',label="predicted")
plt.legend(loc='best', fontsize='xx-large')
plt.xticks(fontsize=18)
plt.yticks(fontsize=16)
plt.show()
plt.figure(figsize=(50, 20))
plt.plot(df_train.index, inv_train_predict,label="actual")
plt.plot(df_test.index, inv_test_predict,label="predicted")
plt.plot(df_train.index, inv_train_y,label="actual")
plt.plot(df_test.index, inv_test_y,label="predicted")
plt.legend(loc='best', fontsize='xx-large')
plt.xticks(fontsize=18)
plt.yticks(fontsize=16)
plt.show()

MAPE = tf.keras.losses.MAPE(
    inv_test_y, inv_test_predict
)
def average(a,n):
    sum = 0
    for i in range(n):
        sum += a[i]
    return sum/n;
n = len(MAPE)
print(average(MAPE,n))
