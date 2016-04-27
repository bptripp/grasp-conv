__author__ = 'bptripp'

# Convolutional network for grasp success prediction

import numpy as np
from keras.models import Sequential
from keras.layers.convolutional import Convolution2D, MaxPooling2D, AveragePooling2D
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.optimizers import Adam
import cPickle
from tuning import disp_tuning

def get_bowls():
    shapes = ['24_bowl-02-Mar-2016-07-03-29',
        '24_bowl-03-Mar-2016-22-54-50',
        '24_bowl-05-Mar-2016-13-53-41',
        '24_bowl-07-Mar-2016-05-06-04',
        '24_bowl-16-Feb-2016-10-12-27',
        '24_bowl-17-Feb-2016-22-00-34',
        '24_bowl-24-Feb-2016-17-38-53',
        '24_bowl-26-Feb-2016-08-35-29',
        '24_bowl-27-Feb-2016-23-52-43',
        '24_bowl-29-Feb-2016-15-01-53']

    depths = []
    labels = []
    for shape in shapes:
        f = open('../data/' + shape + '-random.pkl', 'rb')
        (d, l) = cPickle.load(f)
        f.close()
        depths.extend(d.tolist())
        labels.extend(l.tolist())

    return np.array(depths), np.array(labels)


imsize = (40,40)

model = Sequential()
model.add(Convolution2D(32, 15, 15, input_shape=(1,imsize[0],imsize[1]), init='glorot_normal', border_mode='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(.5))
model.add(Convolution2D(32, 5, 5, init='glorot_normal', border_mode='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(.5))
model.add(Convolution2D(32, 5, 5, init='glorot_normal', border_mode='same'))
model.add(Activation('relu'))
model.add(Dropout(.5))
model.add(Flatten())
model.add(Dense(256))
model.add(Activation('relu'))
model.add(Dropout(.5))
model.add(Dense(1))
#model.add(Activation('sigmoid'))

adam = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
model.compile(loss='mean_squared_error', optimizer=adam)

depths, labels = get_bowls()

depths = np.max(depths.flatten()) - depths # more like disparity; background zero
sigma = np.std(depths.flatten())
depths = depths / sigma

n_train = 90000

X_train = np.zeros((n_train, 1, imsize[0], imsize[1]))
X_train[:,0,:,:] = depths[:n_train,:,:]
Y_train = labels[:n_train]

n_valid = 1000
X_valid = np.zeros((n_valid, 1, imsize[0], imsize[1]))
X_valid[:,0,:,:] = depths[n_train:n_train+n_valid,:,:]
Y_valid = labels[n_train:n_train+n_valid]

h = model.fit(X_train, Y_train, batch_size=32, nb_epoch=200, show_accuracy=True, validation_data=(X_valid, Y_valid))

Y_predict = model.predict(X_valid)
print(Y_predict)
model.save_weights('model_weights.h5')
model.to_json('model.json')

