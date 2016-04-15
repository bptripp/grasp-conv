__author__ = 'bptripp'

# Convolutional network for grasp success prediction

import numpy as np
from keras.models import Sequential
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.optimizers import Adam
from data import AshleyDataSource

# imsize = (50,50)
imsize = (28,28)

model = Sequential()
model.add(Convolution2D(32, 7, 7, input_shape=(1,imsize[0],imsize[1]), init='glorot_normal'))
model.add(Activation('relu'))
model.add(Convolution2D(48, 5, 5, init='glorot_normal'))
model.add(Activation('relu'))
model.add(Convolution2D(64, 3, 3, init='glorot_normal'))
model.add(Activation('relu'))
model.add(Flatten())
model.add(Dense(256))
model.add(Activation('relu'))
model.add(Dropout(.5))
# model.add(Dense(128))
# model.add(Activation('relu'))
# model.add(Dropout(.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))

adam = Adam(lr=0.00001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
model.compile(loss='binary_crossentropy', optimizer=adam)

# source_valid = GraspDataSource('/Users/bptripp/code/grasp-conv/data/output_data.csv',
#                     '/Users/bptripp/code/grasp-conv/data/obj_files',
#                     range=(900,1000))
# X_valid, Y_valid = source_valid.get_XY(50)
#
# source_train = GraspDataSource('/Users/bptripp/code/grasp-conv/data/output_data.csv',
#                     '/Users/bptripp/code/grasp-conv/data/obj_files',
#                     range=(0,900))
# X_train, Y_train = source_train.get_XY(800)
# print(Y_valid)

from data import AshleyDataSource
source = AshleyDataSource()

#balance training data: first 50 successes and first 50 failures
n = 500
X_train = np.zeros((n,1,28,28))
Y_train = np.zeros(n)
fail_count = 0
success_count = 0
i = 0
while fail_count < n/2 or success_count < n/2:
    if source.Y[i] > .5 and success_count < n/2:
        X_train[fail_count+success_count,:,:,:] = source.X[i,:,:,:]
        Y_train[fail_count+success_count] = source.Y[i]
        success_count = success_count + 1
    if source.Y[i] < .5 and fail_count < n/2:
        X_train[fail_count+success_count,:,:,:] = source.X[i,:,:,:]
        Y_train[fail_count+success_count] = source.Y[i]
        fail_count = fail_count + 1
    i = i + 1
    print(i)

print(np.mean(Y_train))

# X_train = source.X[:100,:,:,:]
# Y_train = source.Y[:100]
X_valid = source.X[950:,:,:,:]
Y_valid = source.Y[950:]


h = model.fit(X_train, Y_train, batch_size=32, nb_epoch=500, show_accuracy=True, validation_data=(X_valid, Y_valid))
Y_predict = model.predict(X_valid)
print(Y_predict)
# model.save_weights('model_weights.h5')

import cPickle
f = open('model.pkl', 'wb')
cPickle.dump(model, f)
f.close()
