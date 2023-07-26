# -*- coding: utf-8 -*-
"""
Created on Fri Jan  1 12:59:37 2021

@author: MANOJ KUMAR
"""

import matplotlib.pyplot as plt
from matplotlib.pyplot import savefig
import pandas as pd
import numpy as np
import keras
from keras.layers import Input, Dense, Dropout, Concatenate,Flatten, BatchNormalization
from keras.layers import Conv2D,Conv1D,GRU, MaxPooling2D, ConvLSTM2D, Conv3D,LSTM,Bidirectional
from keras.models import Sequential, Model
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

from keras import regularizers
import math
import time

#.............................Data Preparation.............................................
# load the new file
spe = pd.read_csv('speedcatSEx2.csv', header=0, infer_datetime_format=True, parse_dates=['datetime'], index_col=['datetime'])
print(spe.head())
print(spe.shape)
sample=len(spe)
print(sample)
# save
#spe.to_csv('speedcat.csv')
#---------------------------Data normalization---------------------------------------
scaler1 = MinMaxScaler(feature_range=(0, 1))
spe=scaler1.fit_transform(spe)
spe=np.reshape(spe,(sample,1,14))



# load the new file
target = pd.read_csv('IStargetSExSpe.csv', header=0, infer_datetime_format=True, parse_dates=['datetime'], index_col=['datetime'])
print(target.head())
print(target.shape)
# save
#spe.to_csv('speedcat.csv')
#---------------------------Data normalization---------------------------------------
scaler8 = MinMaxScaler(feature_range=(0, 1))
target=scaler8.fit_transform(target)
target=np.reshape(target,(sample,1))


#-------Set the prediction horizon and input time window--------------------------------------------

#--------------Create the input data set------------------------------------------------------------
train_spe= spe


                 
test_spe= target


#the dataset was divided into two parts: the training dataset and the testing dataset
train_size = int(len(train_spe) * 0.80)
X1=train_spe[0:train_size,:]                 


Y1=test_spe[0:train_size,:]                


y1=Y1


X1_test=train_spe[train_size:,:]                 




Y1_test=test_spe[train_size:,:]                 


y1_test=Y1_test

samples=sample
timesteps=1
features=14
flters_no1=10
testsize = len(X1_test)
print(testsize)
#......................................LSTM Model....................................
spe_input = Input(shape=(timesteps,features))
spe_input1 = BatchNormalization()(spe_input)
layer1 = Bidirectional(LSTM((flters_no1), activation='relu', return_sequences=False))(spe_input1)
flat1 = Flatten()(layer1)

#w2_input = Input(shape=(timesteps,features))
#w2_input1 = BatchNormalization()(w2_input)
#layer2 = LSTM((10), activation='relu', return_sequences=False)(w2_input1)
#layer2 = BatchNormalization()(layer2)
#flat1 = Flatten()(layer1)

#------------Combining the spatio-temporal information using a fusion layer----------------------------------
merged_output = flat1
#out = keras.layers.Dense(128)(merged_output)
out = keras.layers.Dense(1)(merged_output)
model = Model(inputs=spe_input, outputs=out)
model.compile(loss='mean_absolute_error', optimizer='Adamax')
start = time.time()
#-----------------------Record training history---------------------------------------------------------------
train_history = model.fit(X1, y1, epochs=140, batch_size=64, verbose=1,validation_data=(X1_test, y1_test))
#print(X1.shape)
#print(y1.shape)
#print(X1)
#print(y1)

#merged_output = layer1
#out = keras.layers.Dense(128)(merged_output)
#out = keras.layers.Dense(1)(merged_output)
#model = Model(inputs=w4_input, outputs=out)
#model.compile(loss='mean_squared_error', optimizer='Adamax')
#start = time.time()
#-----------------------Record training history---------------------------------------------------------------
#train_history = model.fit(X1, y, epochs=50, batch_size=32, verbose=1,validation_data=(X1_test, y_test))
#callbacks=[history]
#history.loss_plot('epoch')
loss = train_history.history['loss']
val_loss=train_history.history['loss']
end = time.time()
print (end-start)
plt.plot(train_history.history['loss'], label='Train')
plt.plot(train_history.history['val_loss'], label='Test')
plt.legend()
plt.xlabel("Number of epoch")
plt.ylabel("Loss")
plt.show()
#--------------------------------Make prediction----------------------------------------------------------------
y1_pre = model.predict(X1_test)
# Reverse normalization of data
y1_test1 = scaler8.inverse_transform(y1_test)
y1_pre1 = scaler8.inverse_transform(y1_pre)

y1_test2=np.reshape(y1_test1,(1,testsize))
y1_pre2=np.reshape(y1_pre1,(1,testsize))
#print(Y_test1)
#print(Y_pre1)



# save the prediction values and the real values
#np.savetxt( 'test.txt',y1_test1)
# save the prediction values and the real values
#np.savetxt( 'pre.txt',y1_pre1 )
#--------------------------------Calculate evaluation index-----------------------------------------------------
MSE=mean_squared_error(y1_pre1,y1_test1)
MAE=mean_absolute_error(y1_pre1,y1_test1)
mape= np.mean((abs(y1_test1- y1_pre1)) /y1_test1)
rmse=(y1_test1- y1_pre1)*(y1_test1- y1_pre1)
rm=np.sum(rmse)
RMSE=math.sqrt(rm/(rmse.size))
ape2=(abs(y1_test1- y1_pre1)) /y1_test1
ape22=ape2*ape2
summape2=np.sum(ape2)
summape22=np.sum(ape22)
len2=ape2.size
vape=math.sqrt((len2*summape22-summape2*summape2)/(len2*(len2-1)))
ec=(math.sqrt((np.sum((y1_test1- y1_pre1)**2))/len(y1_test1)))/(math.sqrt((np.sum(y1_test1**2))/len(y1_test1))+math.sqrt((np.sum(y1_pre1**2))/len(y1_test1)))
tic = (math.sqrt( (np.sum((y1_test1- y1_pre1)**2)) / len(y1_test1) )) / (math.sqrt((np.sum((y1_pre1)**2)) / len(y1_pre1) ) + math.sqrt((np.sum((y1_test1)**2)) / len(y1_test1)))
cc = np.corrcoef(y1_test2, y1_pre2)
#print('MSE:', MSE)
print('RMSE:', RMSE)
print('MAE:', MAE)
print('MAPE' , mape)
#print('EC' , ec)
print('TIC' , tic)
print('cc',cc)
print('Train Score: %.4f VAPE' % (vape))
