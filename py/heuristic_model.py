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
model.add(Dense(4))
# model.add(Activation('sigmoid'))

adam = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
model.compile(loss='mean_squared_error', optimizer=adam)

f = file('../data/metrics-objects.pkl', 'rb')
intersections, qualities, files = cPickle.load(f)
f.close()

n = len(files)
validation_indices = np.random.randint(0, n, 500)
s = set(validation_indices)
train_indices = [x for x in range(n) if x not in s]

intersections = np.array(intersections)
qualities = np.array(qualities)

mi = np.max(intersections)


def get_scores(fingers):
    # contact quality and symmetry scores for an opposing finger pair

    quality_scores = np.mean(qualities[:,fingers], axis=1)

    symmetry_scores = []
    for intersection in intersections:
        if intersection[fingers[0]] is not None and intersection[fingers[1]] is not None:
            symmetry_scores.append(1 - np.var(intersection[fingers] / mi)/.25)
        else:
            symmetry_scores.append(0)

    return quality_scores, np.array(symmetry_scores)


quality1, symmetry1 = get_scores([0, 2])
quality2, symmetry2 = get_scores([1, 2])

scores = np.concatenate(
    (quality1[:,np.newaxis],
     symmetry1[:,np.newaxis],
     quality2[:,np.newaxis],
     symmetry2[:,np.newaxis]), axis=1)

# print(scores.shape)
# print(np.mean(scores, axis=0))
# print(np.std(scores, axis=0))

# import matplotlib.pyplot as plt
# plt.plot(scores)
# plt.show()

# print(np.max(intersections)) 39
# print(np.min(intersections)) None
# print(np.max(qualities)) 1.0
# print(np.min(qualities)) 0.0

from os.path import join
import scipy
def get_input(image_file):
    image_dir = '../../grasp-conv/data/obj_depths/'
    image = scipy.misc.imread(join(image_dir, image_file))
    rescaled_distance = image / 255.0
    return 1.0 - rescaled_distance

Y_valid = scores[validation_indices,:]
X_valid = []
for ind in validation_indices:
    X_valid.append(get_input(files[ind]))
X_valid = np.array(X_valid)
X_valid = X_valid[:,np.newaxis,:,:]

def generate_XY():
    while 1:
        ind = train_indices[np.random.randint(len(train_indices))]
        Y = np.zeros((1,4))
        Y[0,:] = scores[ind,:]
        X = get_input(files[ind])
        X = X[np.newaxis,np.newaxis,:,:]
        yield (X, Y)

h = model.fit_generator(generate_XY(),
    samples_per_epoch=500, nb_epoch=500,
    validation_data=(X_valid, Y_valid))

print(h)
# f = file('collision-history.pkl', 'wb')
# cPickle.dump(h, f)
# f.close()

