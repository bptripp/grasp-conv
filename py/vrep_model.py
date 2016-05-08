__author__ = 'bptripp'

import numpy as np
from os.path import join
import scipy
import cPickle
from keras.models import Sequential
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.optimizers import Adam, RMSprop
from data import load_all_params

im_width = 80

model = Sequential()
model.add(Convolution2D(32, 9, 9, input_shape=(2,im_width,im_width), init='glorot_normal', border_mode='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Convolution2D(32, 3, 3, init='glorot_normal', border_mode='same'))
model.add(Activation('relu'))
#model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Convolution2D(32, 3, 3, init='glorot_normal', border_mode='same'))
model.add(Activation('relu'))
model.add(Dropout(.25))
model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
#model.add(Dropout(.25))
model.add(Dense(512))
model.add(Activation('relu'))
#model.add(Dropout(.25))
model.add(Dense(64))
model.add(Activation('relu'))
#model.add(Dropout(.25))
model.add(Dense(1))
model.add(Activation('sigmoid'))

#from keras.models import model_from_json
#model = model_from_json(open('v-model-architecture.json').read())
#model.load_weights('v-model-weights.h5')

rmsp = RMSprop(lr=0.001, rho=0.9, epsilon=1e-06)
adam = Adam(lr=0.00001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
model.compile(loss='binary_crossentropy', optimizer=adam)

objects, gripper_pos, gripper_orient, labels = load_all_params('../../grasp-conv/data/output_data.csv')
seq_nums = np.arange(len(objects)) % 1000 #exactly 1000 per object in above file (dated March 18)

labels = np.array(labels)[:,np.newaxis]

n = len(objects)
validation_indices = np.random.randint(0, n, 500) #TODO: generalize across objects
s = set(validation_indices)
train_indices = [x for x in range(n) if x not in s]
#train_indices = train_indices[:1000]


def get_input(object, seq_num):
    image_file = object[:-4] + '-' + str(seq_num) + '.png'
    X = []

    image_dir = '../../grasp-conv/data/obj_depths/'
    image = scipy.misc.imread(join(image_dir, image_file))
    rescaled_distance = image / 255.0
    X.append(1.0 - rescaled_distance)

    image_dir = '../../grasp-conv/data/support_depths/'
    image = scipy.misc.imread(join(image_dir, image_file))
    rescaled_distance = image / 255.0
    X.append(1.0 - rescaled_distance)

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
        #print('ind ' + str(ind) + ' Y ' + str(Y))
        yield (X, Y)

h = model.fit_generator(generate_XY(),
    samples_per_epoch=500, nb_epoch=1500,
    validation_data=(X_valid, Y_valid))

f = file('v-history.pkl', 'wb')
cPickle.dump(h.history, f)
f.close()

json_string = model.to_json()
open('v-model-architecture.json', 'w').write(json_string)
model.save_weights('v-model-weights.h5', overwrite=True)

