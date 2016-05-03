__author__ = 'bptripp'

import numpy as np
import cPickle
from keras.models import Sequential
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.optimizers import Adam

im_width = 80

model = Sequential()
model.add(Convolution2D(32, 9, 9, input_shape=(1,im_width,im_width), init='glorot_normal', border_mode='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Convolution2D(32, 3, 3, init='glorot_normal', border_mode='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Convolution2D(32, 3, 3, init='glorot_normal', border_mode='same'))
model.add(Activation('relu'))
model.add(Flatten())
model.add(Dense(256))
model.add(Activation('relu'))
model.add(Dropout(.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))

adam = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
model.compile(loss='mean_squared_error', optimizer=adam)

f = file('../data/metrics.pkl', 'rb')
intersections, qualities, files = cPickle.load(f)
f.close()

#TODO: len(train_indices) + len(validation_indices) sometimes at random is slightly > len(files)
n = len(files)
validation_indices = np.random.randint(0, n, 500)
s = set(validation_indices)
train_indices = [x for x in range(n) if x not in s]

intersections = np.array(intersections)
collisions = np.where(intersections == np.array(None), 0, intersections)
collisions = np.minimum(np.max(collisions, axis=1), 1)

print(collisions.shape)

from os.path import join
import scipy
def get_input(image_file):
    image_dir=image_dir = '../../grasp-conv/data/support_depths/'
    image = scipy.misc.imread(join(image_dir, image_file))
    rescaled_distance = image / 255.0
    return 1.0 - rescaled_distance # I think this is a good closeup disparity-like representation

Y_valid = collisions[validation_indices]
X_valid = []
for ind in validation_indices:
    X_valid.append(get_input(files[ind]))
X_valid = np.array(X_valid)
X_valid = X_valid[:,np.newaxis,:,:]

def generate_XY():
    while 1:
        ind = train_indices[np.random.randint(len(train_indices))]
        Y = np.zeros((1,1))
        Y[0] = collisions[ind]
        X = get_input(files[ind])
        X = X[np.newaxis,np.newaxis,:,:]
        yield (X, Y)

h = model.fit_generator(generate_XY(),
    samples_per_epoch=500, nb_epoch=500,
    validation_data=(X_valid, Y_valid))

f = file('collision-history.pkl', 'wb')
cPickle.dump(h, f)
f.close()

