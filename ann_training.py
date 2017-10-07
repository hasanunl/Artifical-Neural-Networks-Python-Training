# Artificial Neural Network

# Installing Theano
# pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git

# Installing Tensorflow
# Install Tensorflow from the website: https://www.tensorflow.org/versions/r0.12/get_started/os_setup.html

# Installing Keras
# pip install --upgrade keras

# Part 1 - Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
np.set_printoptions(threshold=np.inf)

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
Y = dataset.iloc[:, 13].values

# Encoding Categorical Data
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:,1] = labelencoder_X_1.fit_transform(X[:,1])
labelencoder_X_2 = LabelEncoder()
X[:,2] = labelencoder_X_2.fit_transform(X[:,2])
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
X=X[:,1:]

#Splitting dataset to Training set and Test set
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)

# Feature scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

# Part 2 - Making the ANN

# Importing the keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense

# Initializing the ANN
classifier = Sequential()

# Adding the input layer and the hidden layer
classifier.add(Dense(6,activation = 'relu',kernel_initializer = 'uniform' ,input_shape=(11,)))

# Adding the second hidden layer
classifier.add(Dense(6,activation = 'relu',kernel_initializer = 'uniform' ))

# Adding the output layer
classifier.add(Dense(1,activation = 'sigmoid',kernel_initializer = 'uniform' ))

# Compiling the ANN
classifier.compile(optimizer = 'adam' ,loss = 'binary_crossentropy' ,metrics = ['accuracy'])

# Fİtting the ANN to the training set
classifier.fit(X_train ,Y_train ,batch_size = 10 ,nb_epoch =100)

# Part 3 - Making the predictions and evaluating the model

# Predicting the test set results
Y_pred = classifier.predict(X_test)
Y_pred = (Y_pred > 0.5) 

# Making the confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, Y_pred)
