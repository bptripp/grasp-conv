__author__ = 'bptripp'

# Convolutional network for grasp success prediction

import numpy as np
from keras.models import Sequential
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.optimizers import Adam
import cPickle
from tuning import disp_tuning
from depthmap import get_distance


def get_bowls():
    shapes = [
        '24_bowl-16-Feb-2016-10-12-27',
        '24_bowl-17-Feb-2016-22-00-34',
        '24_bowl-24-Feb-2016-17-38-53',
        '24_bowl-26-Feb-2016-08-35-29',
        '24_bowl-27-Feb-2016-23-52-43',
        '24_bowl-29-Feb-2016-15-01-53']

    distances = []
    box_distances = []
    labels = []
    for shape in shapes:
        f = file('../data/depths/' + shape + '.pkl', 'rb')
        d, bd, l = cPickle.load(f)
        f.close()

        dist = get_distance(d, .2, 1.0)
        box_dist = get_distance(bd, .2, 1.0)

        distances.extend(dist.tolist())
        box_distances.extend(box_dist.tolist())
        labels.extend(l.tolist())

    return np.array(distances), np.array(box_distances), np.array(labels)


def make_datasets(distances, box_distances, labels):
    # try more like disparity with zero background
    distances = 1 - distances
    box_distances = 1 - box_distances

    indices = np.arange(len(labels))
    validation_flags = indices % 10 == 9
    training_flags = ~validation_flags

    n_train = len(labels[training_flags])
    n_validation = len(labels[validation_flags])
    X_train = np.zeros((n_train,2,distances.shape[1],distances.shape[2]))
    X_valid = np.zeros((n_validation,2,distances.shape[1],distances.shape[2]))
    X_train[:,0,:,:] = distances[training_flags,:,:]
    X_train[:,1,:,:] = box_distances[training_flags,:,:]
    X_valid[:,0,:,:] = distances[validation_flags,:,:]
    X_valid[:,1,:,:] = box_distances[validation_flags,:,:]
    Y_train = labels[training_flags]
    Y_valid = labels[validation_flags]

    return X_train, Y_train, X_valid, Y_valid


distances, box_distances, labels = get_bowls()
X_train, Y_train, X_valid, Y_valid = make_datasets(distances, box_distances, labels)
print(X_train.shape)
print(Y_train.shape)
print(X_valid.shape)
print(Y_valid.shape)


imsize = (X_train.shape[2],X_train.shape[3])

model = Sequential()
model.add(Convolution2D(32, 10, 10, input_shape=(2,imsize[0],imsize[1]), init='glorot_normal', border_mode='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
# model.add(Dropout(.5))
model.add(Convolution2D(32, 5, 5, init='glorot_normal', border_mode='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
# model.add(Dropout(.5))
model.add(Convolution2D(32, 5, 5, init='glorot_normal', border_mode='same'))
model.add(Activation('relu'))
# model.add(Dropout(.5))
model.add(Flatten())
model.add(Dense(256))
model.add(Activation('relu'))
model.add(Dropout(.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))

adam = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
model.compile(loss='mean_squared_error', optimizer=adam)

from cninit import init_model
init_model(model, X_train, Y_train)

h = model.fit(X_train, Y_train, batch_size=32, nb_epoch=200, show_accuracy=True, validation_data=(X_valid, Y_valid))

Y_predict = model.predict(X_valid)
print(Y_predict - Y_valid)
model.save_weights('model_weights.h5')
model.to_json('model.json')

