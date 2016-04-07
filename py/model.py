__author__ = 'bptripp'

# Convolutional network for grasp success prediction

from keras.models import Sequential
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.optimizers import Adam
from data import GraspDataSource

# imsize = (50,50)
imsize = (28,28)

model = Sequential()
model.add(Convolution2D(32, 7, 7, input_shape=(1,imsize[0],imsize[1]), init='glorot_normal'))
model.add(Activation('relu'))
model.add(Convolution2D(32, 7, 7, init='glorot_normal'))
model.add(Activation('relu'))
model.add(Flatten())
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))

adam = Adam(lr=0.0000001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
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
X_train = source.X[:900,:,:,:]
Y_train = source.Y[:900]
X_valid = source.X[900:,:,:,:]
Y_valid = source.Y[900:]

h = model.fit(X_train, Y_train, batch_size=32, nb_epoch=80, show_accuracy=True, validation_data=(X_valid, Y_valid))
Y_predict = model.predict(X_valid)
print(Y_predict)
model.save_weights('model_weights.h5')
