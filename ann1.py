# -*- coding: utf-8 -*-
"""
Created on Fri Jan 18 15:20:44 2019

@author: HP
"""

#Data preprocessing
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:,3:13].values
y = dataset.iloc[:, 13].values

# Encoding categorical data
# Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

#lable countries 0,1,2 
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])

#label gender as  and 1
labelencoder_X_2 = LabelEncoder()
X[:,2] = labelencoder_X_2.fit_transform(X[:,2])

#one hot encoding
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()

#Removing the dummy variable produced during one hot encoding
X = X[:,1:]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Create your classifier here
#ANN
import keras
from keras.models import Sequential
from keras.layers import Dense

#initializing ANN

classifier = Sequential()
  
#Adding the input layer and the first hidden layer
classifier.add(Dense(units = 6,activation = 'relu', kernel_initializer = 'uniform', input_shape = (11,) ))

#Adding the 2nd hidden layer
classifier.add(Dense( units = 6,activation = 'relu',kernel_initializer = 'uniform'))

#Adding the output layer
classifier.add(Dense(units = 1,activation = 'sigmoid',kernel_initializer = 'uniform'))

#Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy',metrics = ['accuracy'] )

# Fitting ANN to the Training set
classifier.fit( X_train, y_train ,batch_size = 10, epochs = 100)

# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred>0.5)
# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)