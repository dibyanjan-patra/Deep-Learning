# -*- coding: utf-8 -*-
"""
Created on Wed Jun 10 11:08:36 2020

@author: dibya
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("D:\\Data Science\\Deep Learning\\ANN with keras\\concrete.csv")
df.info()
df.head()
df.isnull().sum()

#segregating to input an output variable
x= df.iloc[:,0:8]
y= df.iloc[:,-1]

#splitting train and test split
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2, random_state=0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

# Part 2 - Now let's make the ANN!

# Importing the Keras libraries and packages
#pip install keras
import keras
from keras.models import Sequential
from keras.layers import Dense,Dropout
from keras.layers import ReLU,LeakyReLU, ELU

# Initialising the ANN
classifier = Sequential()

#first layer of model
classifier.add(Dense(units=32,input_dim=8,kernel_initializer='he_uniform',activation="relu"))

#second layer of model
classifier.add(Dense(units=32, kernel_initializer='he_uniform',activation='relu'))

#adding third layer
classifier.add(Dense(units=1, kernel_initializer = "glorot_uniform",activation="linear"))
#cla
# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'mean_squared_error')

# Fitting the ANN to the Training set
model_history=classifier.fit(x_train, y_train,validation_split=0.33, batch_size = 10, epochs = 200)


# Part 3 - Making the predictions and evaluating the model

# EVALUATE MODEL IN THE TEST SET
score_mse_test = classifier.evaluate(x_test, y_test)
print('Test Score:', score_mse_test)

# EVALUATE MODEL IN THE TRAIN SET
score_mse_train = classifier.evaluate(x_train, y_train)
print('Train Score:', score_mse_train)

# Predicting the Test set results
y_pred = classifier.predict(x_test)
error = y_pred.flatten()-y_test

model_history.history.keys()
#plotting the data

plt.figure(figsize=(15, 6))
plt.plot(model_history.history['loss'], lw =3, ls = '--', label = 'Loss')
plt.plot(model_history.history['val_loss'], lw =2, ls = '-', label = 'Val Loss')
plt.xlabel('Epochs', fontsize=15)
plt.ylabel('Loss', fontsize=15)
plt.title('MSE')
plt.legend()


###### Using RMSE, wich is square of MSE
from keras import backend as K

def root_mean_squared_error(y_true, y_pred):
        return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))

# ANN with Keras
np.random.seed(10)
classifier = Sequential()

# better values with tanh agains relu, sigmoid...
classifier.add(Dense(13, kernel_initializer = 'uniform', activation = 'tanh', input_dim = 8)) 
classifier.add(Dense(1, kernel_initializer = 'uniform'))
classifier.compile(optimizer = 'adam', loss = root_mean_squared_error)        # metrics=['mse','mae']
#early_stopping_monitor = EarlyStopping(monitor='val_loss', patience=500)  # ignored
history = classifier.fit(x_train, y_train, epochs = 100,validation_split = 0.2)

print('Loss:    ', history.history['loss'][-1], '\nVal_loss: ', history.history['val_loss'][-1])

# EVALUATE MODEL IN THE TEST SET
score_rmse_test = classifier.evaluate(x_test, y_test)
print('Test Score:', score_rmse_test)

# EVALUATE MODEL IN THE TRAIN SET
score_rmse_train = classifier.evaluate(x_train, y_train)
print('Train Score:', score_rmse_train)

plt.figure(figsize=(15, 6))
plt.plot(history.history['loss'], lw =3, ls = '--', label = 'Loss')
plt.plot(history.history['val_loss'], lw =2, ls = '-', label = 'Val Loss')
plt.xlabel('Epochs', fontsize=15)
plt.ylabel('Loss', fontsize=15)
plt.title('RMSE')
plt.legend()










