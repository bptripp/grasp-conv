__author__ = 'bptripp'

"""
The input to this network isn't depth maps, but rather depth of overlap between
object / support and gripper finger trajectory.
"""

import numpy as np
from os.path import join
import scipy
import cPickle
from keras.models import Sequential
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.optimizers import Adam
from data import load_all_params

model = Sequential()
model.add(Convolution2D(32, 9, 9, input_shape=(2,80,16), init='glorot_normal', border_mode='same'))
model.add(Activation('relu'))
model.add(Convolution2D(32, 3, 3, init='glorot_normal', border_mode='same'))
model.add(Activation('relu'))
model.add(Flatten())
model.add(Dense(256))
model.add(Activation('relu'))
model.add(Dense(256))
model.add(Activation('relu'))
# model.add(Dropout(.1))
model.add(Dense(1))
model.add(Activation('sigmoid'))

adam = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
model.compile(loss='binary_crossentropy', optimizer=adam)

objects, gripper_pos, gripper_orient, labels = load_all_params('../../grasp-conv/data/output_data.csv')
seq_nums = np.arange(len(objects)) % 1000 #exactly 1000 per object in above file (dated March 18)

labels = np.array(labels)[:,np.newaxis]

n = len(objects)
validation_indices = np.random.randint(0, n, 500) #TODO: generalize across objects
s = set(validation_indices)
train_indices = [x for x in range(n) if x not in s]


def get_input(object, seq_num):
    image_file = object[:-4] + '-' + str(seq_num) + '-overlap.png'
    X = []

    image_dir = '../../grasp-conv/data/obj_overlap/'
    image = scipy.misc.imread(join(image_dir, image_file))
    X.append(image[:,32:48] / 255.0)

    image_dir = '../../grasp-conv/data/support_overlap/'
    image = scipy.misc.imread(join(image_dir, image_file))
    X.append(image[:,32:48] / 255.0)

    return np.array(X)


Y_valid = labels[validation_indices,:]
X_valid = []
for ind in validation_indices:
    X_valid.append(get_input(objects[ind], seq_nums[ind]))
X_valid = np.array(X_valid)


def generate_XY():
    while 1:
        ind = train_indices[np.random.randint(len(train_indices))]
        Y = np.zeros((1,1))
        Y[0,0] = labels[ind,0]
        X = get_input(objects[ind], seq_nums[ind])
        X = X[np.newaxis,:,:,:]
        yield (X, Y)


# X = get_input(objects[0], 0)
# import matplotlib.pyplot as plt
# fig = plt.figure(figsize=(10,5))
# ax = plt.subplot(1,2,1)
# plt.imshow(X[0,:,:])
# ax = plt.subplot(1,2,2)
# plt.imshow(X[1,:,:])
# plt.show()


h = model.fit_generator(generate_XY(),
    samples_per_epoch=500, nb_epoch=500,
    validation_data=(X_valid, Y_valid))
