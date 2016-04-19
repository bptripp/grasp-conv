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
        f = open('../data/' + shape + '.pkl', 'rb')
        (d, l) = cPickle.load(f)
        f.close()
        depths.extend(d.tolist())
        labels.extend(l.tolist())

    return np.array(depths), np.array(labels)


imsize = (80,80)
centres = np.arange(1,1.9,.1)

model = Sequential()
#model.add(AveragePooling2D())
model.add(Convolution2D(32, 15, 15, input_shape=(len(centres),imsize[0],imsize[1]), init='glorot_normal'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(.5))
model.add(Convolution2D(32, 5, 5, init='glorot_normal'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(.5))
#model.add(Convolution2D(32, 5, 5, init='glorot_normal'))
#model.add(Activation('relu'))
#model.add(Dropout(.5))
model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))

adam = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
model.compile(loss='binary_crossentropy', optimizer=adam)

depths, labels = get_bowls()

depths = np.max(depths.flatten()) - depths # more like disparity; background zero
sigma = np.std(depths.flatten())
depths = depths / sigma

width = .1
n_train = 7000
X_train = np.zeros((n_train, len(centres), imsize[0], imsize[1]))
Y_train = np.zeros(n_train)
fail_count = 0
success_count = 0
i = 0
while fail_count < n_train/2 or success_count < n_train/2:
    if np.mod(i, 1000) < 900:
        if labels[i] > .5 and success_count < n_train/2:
            X_train[fail_count+success_count,:,:,:] = disp_tuning(depths[i,:,:], centres, width)
            Y_train[fail_count+success_count] = labels[i]
            success_count = success_count + 1
            #print('adding ' + str(i) + ' to train')
        if labels[i] <= .5 and fail_count < n_train/2:
            X_train[fail_count+success_count,:,:,:] = disp_tuning(depths[i,:,:], centres, width)
            Y_train[fail_count+success_count] = labels[i]
            fail_count = fail_count + 1
            #print('adding ' + str(i) + ' to train')
    i = i + 1


n_valid = 350
X_valid = np.zeros((n_valid, len(centres), imsize[0], imsize[1]))
Y_valid = np.zeros(n_valid)
valid_count = 0
for i in range(len(labels)):
    if valid_count < n_valid and np.mod(i, 1000) >= 900: 
        #print('adding ' + str(i) + ' to valid')
        X_valid[valid_count,:,:,:] = disp_tuning(depths[i,:,:], centres, width)
        Y_valid[valid_count] = labels[i]
        valid_count = valid_count + 1


# #balance training data: first 50 successes and first 50 failures
# n = 500
# X_train = np.zeros((n,1,28,28))
# Y_train = np.zeros(n)
# fail_count = 0
# success_count = 0
# i = 0
# while fail_count < n/2 or success_count < n/2:
#     if source.Y[i] > .5 and success_count < n/2:
#         X_train[fail_count+success_count,:,:,:] = source.X[i,:,:,:]
#         Y_train[fail_count+success_count] = source.Y[i]
#         success_count = success_count + 1
#     if source.Y[i] < .5 and fail_count < n/2:
#         X_train[fail_count+success_count,:,:,:] = source.X[i,:,:,:]
#         Y_train[fail_count+success_count] = source.Y[i]
#         fail_count = fail_count + 1
#     i = i + 1
#     print(i)
#
# print(np.mean(Y_train))
#
# # X_train = source.X[:100,:,:,:]
# # Y_train = source.Y[:100]
# X_valid = source.X[950:,:,:,:]
# Y_valid = source.Y[950:]


h = model.fit(X_train, Y_train, batch_size=32, nb_epoch=500, show_accuracy=True, validation_data=(X_valid, Y_valid))

Y_predict = model.predict(X_valid)
print(Y_predict)
model.to_json('model.json')
model.save_weights('model_weights.h5')

#
# import cPickle
# f = open('model.pkl', 'wb')
# cPickle.dump(model, f)
# f.close()
