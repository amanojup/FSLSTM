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
from keras.layers import LSTM
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
carSpe = pd.read_csv('carSpe.csv', header=0, infer_datetime_format=True, parse_dates=['datetime'], index_col=['datetime'])
print(carSpe.head())
print(carSpe.shape)
sample=len(carSpe)
print(sample)
# save
#spe.to_csv('speedcat.csv')
#---------------------------Data normalization---------------------------------------
scaler1 = MinMaxScaler(feature_range=(0, 1))
carSpe=scaler1.fit_transform(carSpe)
carSpe=np.reshape(carSpe,(sample,1,1))

twSpe = pd.read_csv('2wSpe.csv', header=0, infer_datetime_format=True, parse_dates=['datetime'], index_col=['datetime'])

#spe.to_csv('speedcat.csv')
#---------------------------Data normalization---------------------------------------
scaler2 = MinMaxScaler(feature_range=(0, 1))
twSpe=scaler2.fit_transform(twSpe)
twSpe=np.reshape(twSpe,(sample,1,1))

thwSpe = pd.read_csv('3wSpe.csv', header=0, infer_datetime_format=True, parse_dates=['datetime'], index_col=['datetime'])
#spe.to_csv('speedcat.csv')
#---------------------------Data normalization---------------------------------------
scaler3 = MinMaxScaler(feature_range=(0, 1))
thwSpe=scaler3.fit_transform(thwSpe)
thwSpe=np.reshape(thwSpe,(sample,1,1))

BSpe = pd.read_csv('BSpe.csv', header=0, infer_datetime_format=True, parse_dates=['datetime'], index_col=['datetime'])
#spe.to_csv('speedcat.csv')
#---------------------------Data normalization---------------------------------------
scaler4 = MinMaxScaler(feature_range=(0, 1))
BSpe=scaler4.fit_transform(BSpe)
BSpe=np.reshape(BSpe,(sample,1,1))

LCVSpe = pd.read_csv('LCVSpe.csv', header=0, infer_datetime_format=True, parse_dates=['datetime'], index_col=['datetime'])
#spe.to_csv('speedcat.csv')
#---------------------------Data normalization---------------------------------------
scaler5 = MinMaxScaler(feature_range=(0, 1))
LCVSpe=scaler5.fit_transform(LCVSpe)
LCVSpe=np.reshape(LCVSpe,(sample,1,1))

TSpe = pd.read_csv('TSpe.csv', header=0, infer_datetime_format=True, parse_dates=['datetime'], index_col=['datetime'])
#spe.to_csv('speedcat.csv')
#---------------------------Data normalization---------------------------------------
scaler6 = MinMaxScaler(feature_range=(0, 1))
TSpe=scaler4.fit_transform(TSpe)
TSpe=np.reshape(TSpe,(sample,1,1))

cycSpe = pd.read_csv('cycSpe.csv', header=0, infer_datetime_format=True, parse_dates=['datetime'], index_col=['datetime'])
#spe.to_csv('speedcat.csv')
#---------------------------Data normalization---------------------------------------
scaler7 = MinMaxScaler(feature_range=(0, 1))
cycSpe=scaler7.fit_transform(cycSpe)
cycSpe=np.reshape(cycSpe,(sample,1,1))

carVol = pd.read_csv('carVol.csv', header=0, infer_datetime_format=True, parse_dates=['datetime'], index_col=['datetime'])
sample=len(carVol)
print(sample)
# save
#spe.to_csv('speedcat.csv')
#---------------------------Data normalization---------------------------------------
scaler9 = MinMaxScaler(feature_range=(0, 1))
carVol=scaler9.fit_transform(carVol)
carVol=np.reshape(carVol,(sample,1,1))

twVol = pd.read_csv('2wVol.csv', header=0, infer_datetime_format=True, parse_dates=['datetime'], index_col=['datetime'])

#spe.to_csv('speedcat.csv')
#---------------------------Data normalization---------------------------------------
scaler10 = MinMaxScaler(feature_range=(0, 1))
twVol=scaler10.fit_transform(twVol)
twVol=np.reshape(twVol,(sample,1,1))

thwVol = pd.read_csv('3wVol.csv', header=0, infer_datetime_format=True, parse_dates=['datetime'], index_col=['datetime'])
#spe.to_csv('speedcat.csv')
#---------------------------Data normalization---------------------------------------
scaler11 = MinMaxScaler(feature_range=(0, 1))
thwVol=scaler11.fit_transform(thwVol)
thwVol=np.reshape(thwVol,(sample,1,1))

BVol = pd.read_csv('BVol.csv', header=0, infer_datetime_format=True, parse_dates=['datetime'], index_col=['datetime'])
#spe.to_csv('speedcat.csv')
#---------------------------Data normalization---------------------------------------
scaler12 = MinMaxScaler(feature_range=(0, 1))
BVol=scaler12.fit_transform(BVol)
BVol=np.reshape(BVol,(sample,1,1))

LCVVol = pd.read_csv('LCVVol.csv', header=0, infer_datetime_format=True, parse_dates=['datetime'], index_col=['datetime'])
#spe.to_csv('speedcat.csv')
#---------------------------Data normalization---------------------------------------
scaler13 = MinMaxScaler(feature_range=(0, 1))
LCVVol=scaler13.fit_transform(LCVVol)
LCVVol=np.reshape(LCVVol,(sample,1,1))

TVol = pd.read_csv('TVol.csv', header=0, infer_datetime_format=True, parse_dates=['datetime'], index_col=['datetime'])
#spe.to_csv('speedcat.csv')
#---------------------------Data normalization---------------------------------------
scaler14 = MinMaxScaler(feature_range=(0, 1))
TVol=scaler14.fit_transform(TVol)
TVol=np.reshape(TVol,(sample,1,1))

cycVol = pd.read_csv('cycVol.csv', header=0, infer_datetime_format=True, parse_dates=['datetime'], index_col=['datetime'])
#spe.to_csv('speedcat.csv')
#---------------------------Data normalization---------------------------------------
scaler15 = MinMaxScaler(feature_range=(0, 1))
cycVol=scaler15.fit_transform(cycVol)
cycVol=np.reshape(cycVol,(sample,1,1))

# load the new file
target = pd.read_csv('speedtargetSEx.csv', header=0, infer_datetime_format=True, parse_dates=['datetime'], index_col=['datetime'])
print(target.head())
print(target.shape)
# save
#spe.to_csv('speedcat.csv')
#---------------------------Data normalization---------------------------------------
scaler8 = MinMaxScaler(feature_range=(0, 1))
target=scaler8.fit_transform(target)
target=np.reshape(target,(sample,1))

targetVol = pd.read_csv('voltargetSEx.csv', header=0, infer_datetime_format=True, parse_dates=['datetime'], index_col=['datetime'])
scaler16 = MinMaxScaler(feature_range=(0, 1))
targetVol=scaler16.fit_transform(targetVol)
targetVol=np.reshape(targetVol,(sample,1))
#-------Set the prediction horizon and input time window--------------------------------------------

#--------------Create the input data set------------------------------------------------------------



#the dataset was divided into two parts: the training dataset and the testing dataset
train_size = int(len(carSpe) * 0.80)
X1=carSpe[0:train_size,:] 
X2=twSpe[0:train_size,:]
X3=thwSpe[0:train_size,:]
X4=BSpe[0:train_size,:]
X5=LCVSpe[0:train_size,:]
X6=TSpe[0:train_size,:]
X7=cycSpe[0:train_size,:]
X8=carVol[0:train_size,:] 
X9=twVol[0:train_size,:]
X10=thwVol[0:train_size,:]
X11=BVol[0:train_size,:]
X12=LCVVol[0:train_size,:]
X13=TVol[0:train_size,:]
X14=cycVol[0:train_size,:]                
Y1=target[0:train_size,:] 
Y2=targetVol[0:train_size,:]               
y1=Y1
y2=Y2

X1_test=carSpe[train_size:,:]
X2_test=twSpe[train_size:,:] 
X3_test=thwSpe[train_size:,:] 
X4_test=BSpe[train_size:,:] 
X5_test=LCVSpe[train_size:,:] 
X6_test=TSpe[train_size:,:] 
X7_test=cycSpe[train_size:,:] 
X8_test=carVol[train_size:,:]
X9_test=twVol[train_size:,:] 
X10_test=thwVol[train_size:,:] 
X11_test=BVol[train_size:,:] 
X12_test=LCVVol[train_size:,:] 
X13_test=TVol[train_size:,:] 
X14_test=cycVol[train_size:,:]                  
Y1_test=target[train_size:,:] 
Y2_test=targetVol[train_size:,:]   
y1_test=Y1_test             
y2_test=Y2_test

samples=sample
timesteps=1
features=1
neurons=10
rate = 0.001
testsize = len(X1_test)
print(testsize)
#X, y = make_regression(n_samples=1000, n_features=10, n_informative=5, n_targets=3, random_state=2)
 
#......................................LSTM Model....................................
carSpe = Input(shape=(timesteps,features))
carSpe1 = BatchNormalization()(carSpe)
layer1 = LSTM((neurons), activation='relu', kernel_regularizer = regularizers.l2(rate), bias_regularizer = regularizers.l2(rate), return_sequences=False)(carSpe1)
flat1 = Flatten()(layer1)

twSpe = Input(shape=(timesteps,features))
twSpe1 = BatchNormalization()(twSpe)
layer2 = LSTM((neurons), activation='relu', kernel_regularizer = regularizers.l2(rate), bias_regularizer = regularizers.l2(rate), return_sequences=False)(twSpe1)
flat2 = Flatten()(layer2)

thwSpe = Input(shape=(timesteps,features))
thwSpe1 = BatchNormalization()(thwSpe)
layer3 = LSTM((neurons), activation='relu', kernel_regularizer = regularizers.l2(rate), bias_regularizer = regularizers.l2(rate), return_sequences=False)(thwSpe1)
flat3 = Flatten()(layer3)

BSpe = Input(shape=(timesteps,features))
BSpe1 = BatchNormalization()(BSpe)
layer4 = LSTM((neurons), activation='relu', kernel_regularizer = regularizers.l2(rate), bias_regularizer = regularizers.l2(rate), return_sequences=False)(BSpe1)
flat4 = Flatten()(layer4)

LCVSpe = Input(shape=(timesteps,features))
LCVSpe1 = BatchNormalization()(LCVSpe)
layer5 = LSTM((neurons), activation='relu', kernel_regularizer = regularizers.l2(rate), bias_regularizer = regularizers.l2(rate), return_sequences=False)(LCVSpe1)
flat5 = Flatten()(layer5)

TSpe = Input(shape=(timesteps,features))
TSpe1 = BatchNormalization()(TSpe)
layer6 = LSTM((neurons), activation='relu', kernel_regularizer = regularizers.l2(rate), bias_regularizer = regularizers.l2(rate), return_sequences=False)(TSpe1)
flat6 = Flatten()(layer6)

cycSpe = Input(shape=(timesteps,features))
cycSpe1 = BatchNormalization()(cycSpe)
layer7 = LSTM((neurons), activation='relu', kernel_regularizer = regularizers.l2(rate), bias_regularizer = regularizers.l2(rate), return_sequences=False)(cycSpe1)
flat7 = Flatten()(layer7)

carVol = Input(shape=(timesteps,features))
carVol1 = BatchNormalization()(carVol)
layer8 = LSTM((neurons), activation='relu', kernel_regularizer = regularizers.l2(rate), bias_regularizer = regularizers.l2(rate), return_sequences=False)(carVol1)
flat8 = Flatten()(layer8)

twVol = Input(shape=(timesteps,features))
twVol1 = BatchNormalization()(twVol)
layer9 = LSTM((neurons), activation='relu', kernel_regularizer = regularizers.l2(rate), bias_regularizer = regularizers.l2(rate), return_sequences=False)(twVol1)
flat9 = Flatten()(layer9)

thwVol = Input(shape=(timesteps,features))
thwVol1 = BatchNormalization()(thwVol)
layer10 = LSTM((neurons), activation='relu', kernel_regularizer = regularizers.l2(rate), bias_regularizer = regularizers.l2(rate), return_sequences=False)(thwVol1)
flat10 = Flatten()(layer10)

BVol = Input(shape=(timesteps,features))
BVol1 = BatchNormalization()(BVol)
layer11 = LSTM((neurons), activation='relu', kernel_regularizer = regularizers.l2(rate), bias_regularizer = regularizers.l2(rate), return_sequences=False)(BVol1)
flat11 = Flatten()(layer11)

LCVVol = Input(shape=(timesteps,features))
LCVVol1 = BatchNormalization()(LCVVol)
layer12 = LSTM((neurons), activation='relu', kernel_regularizer = regularizers.l2(rate), bias_regularizer = regularizers.l2(rate), return_sequences=False)(LCVVol1)
flat12 = Flatten()(layer12)

TVol = Input(shape=(timesteps,features))
TVol1 = BatchNormalization()(TVol)
layer13 = LSTM((neurons), activation='relu', kernel_regularizer = regularizers.l2(rate), bias_regularizer = regularizers.l2(rate), return_sequences=False)(TVol1)
flat13 = Flatten()(layer13)

cycVol = Input(shape=(timesteps,features))
cycVol1 = BatchNormalization()(cycVol)
layer14 = LSTM((neurons), activation='relu', kernel_regularizer = regularizers.l2(rate), bias_regularizer = regularizers.l2(rate), return_sequences=False)(cycVol1)
flat14 = Flatten()(layer14)
#------------Combining the spatio-temporal information using a fusion layer----------------------------------
#merged_output = keras.layers.concatenate([layer2, layer4, layer6, layer8, layer10, layer12, layer14])
merged_output = keras.layers.concatenate([flat1,flat2,flat3,flat4,flat5,flat6,flat7,flat8,flat9,flat10,flat11,flat12,flat13,flat14])
out = keras.layers.Dense(1)(merged_output)
model = Model(inputs=[carSpe,twSpe,thwSpe,BSpe,LCVSpe,TSpe,cycSpe,carVol,twVol,thwVol,BVol,LCVVol,TVol,cycVol], outputs=out)
model.compile(loss='mean_absolute_error', optimizer='Adamax')
start = time.time()
#-----------------------Record training history---------------------------------------------------------------
train_history = model.fit([X1,X2,X3,X4,X5,X6,X7,X8,X9,X10,X11,X12,X13,X14], y2, epochs=120, batch_size=64, verbose=1,validation_data=([X1_test,X2_test,X3_test,X4_test,X5_test,X6_test,X7_test,X8_test,X9_test,X10_test,X11_test,X12_test,X13_test,X14_test], y2_test))




#merged_output = layer12
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
plt.legend(fontsize=14.0)
plt.xlabel("Epoch number",fontsize=14.0)
plt.ylabel("Loss",fontsize=14.0)
plt.show()
#--------------------------------Make prediction----------------------------------------------------------------
#y1_pre = model.predict(X1_test)
y1_pre = model.predict([X1_test,X2_test,X3_test,X4_test,X5_test,X6_test,X7_test,X8_test,X9_test,X10_test,X11_test,X12_test,X13_test,X14_test])
# Reverse normalization of data
y1_test1 = scaler16.inverse_transform(y2_test)
y1_pre1 = scaler16.inverse_transform(y1_pre)

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
tic = (math.sqrt( (np.sum((y1_test1- y1_pre1)**2)) / len(y1_test1) )) / (math.sqrt((np.sum((y1_test1)**2)) / len(y1_test1) ) + math.sqrt((np.sum((y1_test1)**2)) / len(y1_test1)))
cc = np.corrcoef(y1_test2, y1_pre2)
#print('MSE:', MSE)
print('RMSE:', RMSE)
print('MAE:', MAE)
print('MAPE' , mape)
#print('EC' , ec)
#print('TIC' , tic)
print('cc',cc)
print('Train Score: %.4f VAPE' % (vape))
