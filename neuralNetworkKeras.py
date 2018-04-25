# -*- coding: utf-8 -*-
"""
Created on Fri Feb 16 13:49:00 2018

@author: sushi
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

file = pd.read_csv('Churn_Modelling.csv')
#print(file.head(5))
#print(file.columns)
#import karas
#from keras.models import Sequential
#from keras.layers import Dense

x = file.iloc[:, 3:13].values
y = file.iloc[:,13].values

#preprocessing
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_x1 = LabelEncoder()
x[:,1]=labelencoder_x1.fit_transform(x[:,1])
labelencoder_x2 = LabelEncoder()
x[:,2] = labelencoder_x2.fit_transform(x[:,2])
onehot = OneHotEncoder(categorical_features=[1])
x = onehot.fit_transform(x).toarray()
x = x[:, 1:]





#split datasets
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0)

#normalise or standerise
from sklearn.preprocessing import StandardScaler
standerization = StandardScaler()
x_train = standerization.fit_transform(x_train)
x_test = standerization.fit_transform(x_test)

#aaba NN -> Now Neural Network
import keras
from keras.models import Sequential
from keras.layers import Dense

network = Sequential()
#input and first hidden layer
network.add(Dense(units=6,kernel_initializer='uniform',activation='relu',input_dim=11))
#lets try second hidden layer
network.add(Dense(units=6,kernel_initializer='uniform',activation='relu'))
#output layer
network.add(Dense(units=1,kernel_initializer='uniform',activation='sigmoid'))

network.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])


network.fit(x_train,y_train,batch_size=10,epochs=100)


#test network
y_pred = network.predict(x_test)

from sklearn.metrics import confusion_matrix
y_pred = y_pred>0.5
matrix = confusion_matrix(y_test,y_pred)


