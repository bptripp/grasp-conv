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
model.add(Convolution2D(64, 7, 7, input_shape=(2,80,16), init='glorot_normal', border_mode='same'))
model.add(Activation('relu'))
model.add(Convolution2D(64, 3, 3, init='glorot_normal', border_mode='same'))
model.add(Activation('relu'))
model.add(Dropout(.5))
model.add(Convolution2D(64, 3, 3, init='glorot_normal', border_mode='same'))
model.add(Activation('relu'))
model.add(Dropout(.5))
model.add(Flatten())
model.add(Dense(1024))
model.add(Activation('relu'))
model.add(Dropout(.5))
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))

adam = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
model.compile(loss='binary_crossentropy', optimizer=adam)

objects, gripper_pos, gripper_orient, labels = load_all_params('../../grasp-conv/data/output_data.csv')
seq_nums = np.arange(len(objects)) % 1000 #exactly 1000 per object in above file (dated March 18)

labels = np.array(labels)[:,np.newaxis]

n = len(objects)
validation_indices = np.random.randint(0, n, 5000) #TODO: generalize across objects
s = set(validation_indices)
train_indices = [x for x in range(n) if x not in s]

f = file('o-valid-ind.pkl', 'wb')
cPickle.dump(validation_indices, f)
f.close()

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
        batch_X = []
        batch_Y = []
        for i in range(32):
            ind = train_indices[np.random.randint(len(train_indices))]
            Y = np.zeros((1))
            Y[0] = labels[ind,0]
            X = get_input(objects[ind], seq_nums[ind])
            #X = X[np.newaxis,:,:,:]
            batch_X.append(X)
            batch_Y.append(Y)
        yield (np.array(batch_X), np.array(batch_Y))


# X = get_input(objects[0], 0)
# import matplotlib.pyplot as plt
# fig = plt.figure(figsize=(10,5))
# ax = plt.subplot(1,2,1)
# plt.imshow(X[0,:,:])
# ax = plt.subplot(1,2,2)
# plt.imshow(X[1,:,:])
# plt.show()


h = model.fit_generator(generate_XY(),
    samples_per_epoch=32768, nb_epoch=250,
    validation_data=(X_valid, Y_valid))

f = file('o-history.pkl', 'wb')
cPickle.dump(h.history, f)
f.close()

json_string = model.to_json()
open('o-model-architecture.json', 'w').write(json_string)
model.save_weights('o-model-weights.h5', overwrite=True)

