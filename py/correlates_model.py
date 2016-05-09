__author__ = 'bptripp'

"""
The input to this network isn't depth maps, but rather a group of hand-engineered features.
"""

import csv
import numpy as np
from os.path import join
import scipy
import cPickle
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import Adam
from data import load_all_params

model = Sequential()
model.add(Dense(256, input_shape=[22]))
model.add(Activation('relu'))
model.add(Dense(256))
model.add(Activation('relu'))
model.add(Dropout(.1))
model.add(Dense(1))
model.add(Activation('sigmoid'))

adam = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
model.compile(loss='binary_crossentropy', optimizer=adam)

labels = []
features = []
with open('correlates.csv', 'rb') as csvfile:
    r = csv.reader(csvfile, delimiter=',')
    for row in r:
        labels.append(row[0])
        features.append(row[1:])

labels = np.array(labels).astype(float)
features = np.array(features).astype(float)

# normalize features ...
for i in range(features.shape[1]):
    features[:,i] = features[:,i] - np.mean(features[:,i])
    features[:,i] = features[:,i] / np.std(features[:,i])

n = len(labels)
validation_indices = np.random.randint(0, n, 500)
s = set(validation_indices)
train_indices = [x for x in range(n) if x not in s]

Y_valid = labels[validation_indices,np.newaxis]
X_valid = features[validation_indices,:]
Y_train = labels[train_indices,np.newaxis]
X_train = features[train_indices,:]

# print(Y_valid.shape)
# print(X_valid.shape)
# print(Y_train.shape)
# print(X_train.shape)

h = model.fit(X_train, Y_train, nb_epoch=500, validation_data=(X_valid, Y_valid))
