# -*- coding: utf-8 -*-
"""
Created on Fri Jul 24 00:41:10 2020

@author: beniw
"""
from tensorflow.keras import utils as np_utils
import mne
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import tensorflow as tf
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Input, Flatten
from tensorflow.keras.constraints import max_norm
from tensorflow.keras.models import Sequential


fname = "./epoch-epo.fif"
epochs = mne.read_epochs(fname)

event_id = dict(non_target_stim = 1, target_stim = 2)

fname_diff_sub = "./epoch-epo-A02.fif"
epochs_diff_sub = mne.read_epochs(fname)

event_id_diff_sub = dict(non_target_stim = 1, target_stim = 2)


X = epochs.get_data()
y = epochs.events[:,-1] 

X_diff_sub = epochs_diff_sub.get_data()
y_diff_sub = epochs_diff_sub.events[:,-1] 
        

# Training Set
x_train = X[:57600]
y_train = y[:57600]

# Test
x_test = X_diff_sub[:50000]
y_test = y_diff_sub[:50000]

# ------------XXXXXX-------------------

# # Sampling the data to remove imbalance -- OverSampler :
#Around (95%)  

x_train_2d = x_train.reshape(len(x_train), -1)
x_test_2d = x_test.reshape(len(x_test), -1)

from imblearn.over_sampling import RandomOverSampler 

rus = RandomOverSampler(random_state=0)
X_resampled, y_resampled = rus.fit_resample(x_train_2d, y_train)
x_test_diff_sub_resampled, y_test_diff_sub_resampled = rus.fit_resample(x_test_2d, y_test)

# Sampling the data to remove imbalance -- UnderSampler : With the training set of size 6000
# Around(84-85%)

# from imblearn.under_sampling import RandomUnderSampler

# # sampling_strategy={1:5000, 2:1000}
# rus = RandomUnderSampler(random_state=0)
# X_resampled, y_resampled = rus.fit_resample(x_train_2d, y_train)
# x_test_diff_sub_resampled, y_test_diff_sub_resampled = rus.fit_resample(x_test_2d, y_test)

# Converting and transferrinf the data back

x_train = np.reshape(X_resampled, (len(X_resampled),8,257))
y_train = y_resampled
x_test = np.reshape(x_test_diff_sub_resampled, (len(x_test_diff_sub_resampled),8,257))
y_test = y_test_diff_sub_resampled

#from sklearn.model_selection import train_test_split
#x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.4, random_state = 0)


# Reshaping data for sending it to CNN

kernels, channels, samples = 1, 8, 257

x_train      = x_train.reshape(x_train.shape[0], kernels, channels, samples)
x_test       = x_test.reshape(x_test.shape[0], kernels, channels, samples)

# Converting the y data in binary matrix form


y_train  = np_utils.to_categorical(y_train-1)
y_test   = np_utils.to_categorical(y_test-1)


dropout_rate = 0.5

# Part 2 - Building the CNN

# Taking the input and putting it in the CNN
cnn = Sequential()
# Step 1 - Convolution Pool Block 1 : Designed to handle EEG data

# Convolution(Temporal) - 25 Linear Units
cnn.add(Conv2D(filters=25, kernel_size = [1,5], input_shape=(1, channels, samples),
            kernel_constraint = max_norm(3., axis = (0,1,2)),
            data_format = "channels_first"))

# Convolution(Spatial) - 25 Exponential Linear Units
cnn.add(Conv2D(filters = 25, kernel_size = [channels, 1], kernel_constraint = max_norm(3., axis=(0,1,2)),
            data_format = "channels_first"))

# BatchNormalization
cnn.add(BatchNormalization(axis=1, epsilon=1e-05, momentum=0.1))

# Activation - elu
cnn.add(Activation('elu'))

# MaxPooling(2D) - stride = 3x1
cnn.add(MaxPooling2D(pool_size=(1, 3), strides=(1, 3), data_format = "channels_first"))

# Dropout
cnn.add(Dropout(dropout_rate))

# Step 2 - Convolution Pool Block 2 : Standard Convolutional Pool Block

cnn.add(Conv2D(filters=50, kernel_size = [1,5], kernel_constraint = max_norm(3., axis = (0,1,2)),
            data_format = "channels_first"))

# BatchNormalization
cnn.add(BatchNormalization(axis=1, epsilon=1e-05, momentum=0.1))

# Activation - elu
cnn.add(Activation('elu'))

# MaxPooling(2D) - stride = 3x1
cnn.add(MaxPooling2D(pool_size=(1, 3), strides=(1, 3), data_format = "channels_first"))

# Dropout
cnn.add(Dropout(dropout_rate))

# Step 3 - Convolution Pool Block 3 : Standard Convolutional Pool Block

cnn.add(Conv2D(filters=100, kernel_size = [1,5], kernel_constraint = max_norm(3., axis = (0,1,2)),
            data_format = "channels_first"))

# BatchNormalization
cnn.add(BatchNormalization(axis=1, epsilon=1e-05, momentum=0.1))

# Activation - elu
cnn.add(Activation('elu'))

# MaxPooling(2D) - stride = 3x1
cnn.add(MaxPooling2D(pool_size=(1, 3), strides=(1, 3), data_format = "channels_first"))

# Dropout
cnn.add(Dropout(dropout_rate))

# Step 4 - Convolution Pool Block 4 : Standard Convolutional Pool Block

cnn.add(Conv2D(filters=200, kernel_size = [1,5], kernel_constraint = max_norm(3., axis = (0,1,2)),
            data_format = "channels_first"))

# BatchNormalization
cnn.add(BatchNormalization(axis=1, epsilon=1e-05, momentum=0.1))

# Activation - elu
cnn.add(Activation('elu'))

# MaxPooling(2D) - stride = 3x1
cnn.add(MaxPooling2D(pool_size=(1, 3), strides=(1, 3), data_format = "channels_first"))

# Dropout
cnn.add(Dropout(dropout_rate))

# Step 5 - Dense SoftMax Classification Layer

cnn.add(Flatten())

cnn.add(Dense(2, kernel_constraint = max_norm(0.5)))
cnn.add(Activation('softmax'))

# Part 3 - Training the CNN

# Compiling the CNN
cnn.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

# Training the CNN on the Training set and evaluating it on the Test set
cnn.fit(x_train, y_train, validation_data = (x_test,y_test), batch_size = 32, epochs = 50)








########################################Early Models Used for training##################################################
# import pandas as pd

# # Importing the dataset

# dataset_X = pd.read_csv('X.csv')
# dataset_y = pd.read_csv('y.csv')

# X = dataset_X.iloc[:, :-1].values
# y = dataset_y.iloc[:, -1].values

# from sklearn.pipeline import make_pipeline
# from mne.decoding import Vectorizer
# from sklearn.model_selection import cross_val_score
# from sklearn.linear_model import LogisticRegression

# from pyriemann.estimation import XdawnCovariances
# from pyriemann.classification import MDM
# #from pyriemann.utils.viz import plot_confusion_matrix

# from sklearn.model_selection import KFold
# from sklearn.metrics import classification_report


# n_components = 3  # pick some components

# # Define a monte-carlo cross-validation generator (reduce variance):
# cv = KFold(n_splits=10, random_state=42)
# pr = np.zeros(len(y))

# print('Multiclass classification with XDAWN + MDM')

# clf = make_pipeline(XdawnCovariances(n_components), MDM())

# for train_idx, test_idx in cv.split(X):
#     y_train, y_test =  y[train_idx], y[test_idx]

#     clf.fit(X[train_idx], y_train)
#     pr[test_idx] = clf.predict(X[test_idx])

# print(classification_report(y, pr))

# from sklearn.model_selection import train_test_split
# x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.75, random_state = 0)


# #Training the K-NN model on the Training set

# from sklearn.pipeline import make_pipeline
# from mne.decoding import Vectorizer
# from sklearn.svm import SVC
# from sklearn.metrics import classification_report
# from sklearn.preprocessing import StandardScaler

# classifier = make_pipeline(Vectorizer(), StandardScaler(), SVC(kernel = 'linear'))

# print('Multiclass classification : ')

# classifier.fit(x_train, y_train)

# y_pred = classifier.predict(x_test)
# print(np.concatenate((y_pred[:50], y_test[:50]),0))

# print(classification_report(y_test, y_pred))
###################################################################################################