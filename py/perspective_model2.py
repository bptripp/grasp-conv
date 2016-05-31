__author__ = 'bptripp'

# Just like perspective_model, but with ray-based metrics instead of depth map-based metrics as targets.
from os import listdir
from os.path import isfile, join
import cPickle
import numpy as np
import scipy
from keras.models import Sequential
from keras.layers.core import Dense, Flatten, Dropout, Activation
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import Adam
from keras.models import model_from_json
from keras.regularizers import activity_l2

num_outputs = 250

def get_model():
    model = Sequential()
    model.add(Convolution2D(64, 7, 7, input_shape=(1,80,80), init='glorot_normal', border_mode='same', activity_regularizer=activity_l2(0.000001)))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, 3, 3, init='glorot_normal', border_mode='same', activity_regularizer=activity_l2(0.000001)))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, 3, 3, init='glorot_normal', border_mode='same', activity_regularizer=activity_l2(0.000001)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2,2)))
    model.add(Flatten())
    model.add(Dense(256, activity_regularizer=activity_l2(0.000001)))
    model.add(Activation('relu'))
    model.add(Dropout(.5))
    model.add(Dense(1024, activity_regularizer=activity_l2(0.000001)))
    model.add(Activation('relu'))
    model.add(Dropout(.5))
    model.add(Dense(num_outputs))

    adam = Adam(lr=0.000001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    model.compile(loss='mse', optimizer=adam)

    return model


def load_model(weights_file='p2-model-weights.h5'):
    model = model_from_json(open('p2-model-architecture-big-reg.json').read())
    model.load_weights(weights_file)

    adam = Adam(lr=0.000001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    model.compile(loss='mse', optimizer=adam)

    return model


def load_data():
    import csv

    objects = []
    with open('../data/eye-perspectives-objects.csv') as f:
        reader = csv.reader(f, delimiter=',')
        for row in reader:
            objects.append(row[0])

    image_filenames = []
    for i in range(len(objects)):
        case = i % 200
        target = case / 20
        eye = case % 20
        fn = objects[i][:-4] + '-' + str(target) + '-' + str(eye) + '.png'
        image_filenames.append(fn)

    neuron_metrics = []
    with open('../data/ray-metrics.csv') as f:
        reader = csv.reader(f, delimiter=',')
        for row in reader:
            neuron_metrics.append(row)

    neuron_metrics = np.array(neuron_metrics)

    return image_filenames, neuron_metrics[:,:num_outputs]


def get_input(image_dir, image_file):
    image = scipy.misc.imread(join(image_dir, image_file))
    return np.array(image / 255.0)


def get_XY(image_dir, image_filenames, neuron_metrics, indices):
    X = []
    Y = []
    for ind in indices:
        X.append(get_input(image_dir, image_filenames[ind])[np.newaxis,:])
        Y.append(neuron_metrics[ind])
    return np.array(X), np.array(Y)


def train_model(model, image_dir, train_indices, valid_indices):
    image_filenames, neuron_metrics = load_data()
    print(len(image_filenames))

    X_valid, Y_valid = get_XY(image_dir, image_filenames, neuron_metrics, valid_indices)

    def generate_XY():
        while 1:
            batch_X = []
            batch_Y = []
            for i in range(32):
                ind = train_indices[np.random.randint(len(train_indices))]
                X = get_input(image_dir, image_filenames[ind])
                Y = neuron_metrics[ind]
                batch_X.append(X[np.newaxis,:])
                batch_Y.append(Y)
            yield (np.array(batch_X), np.array(batch_Y))

    for i in range(10):
        h = model.fit_generator(generate_XY(),
            samples_per_epoch=8192, nb_epoch=100,
            validation_data=(X_valid, Y_valid))

        with file('p2-history-big-reg-' + str(i) + '.pkl', 'wb') as f:
            cPickle.dump(h.history, f)

        json_string = model.to_json()
        open('p2-model-architecture-big-reg.json', 'w').write(json_string)
        model.save_weights('p2-model-weights-big-reg-' + str(i) + '.h5', overwrite=True)


def predict(model, image_dir, data_file, indices):
    image_filenames, neuron_metrics = load_data(data_file)
    X, Y = get_XY(image_dir, image_filenames, neuron_metrics, indices)
    return Y, model.predict(X, batch_size=32, verbose=0)


if __name__ == '__main__':
    # image_filenames, neuron_metrics = load_data()

    valid_indices = np.arange(0, 50000, 100)
    s = set(valid_indices)
    train_indices = [x for x in range(70000) if x not in s]

    model = get_model()
    # model = load_model(weights_file='p2-model-weights-big-reg-9.h5')
    train_model(model,
                '../../grasp-conv/data/eye-perspectives',
                train_indices,
                valid_indices)

    #model = load_model(weights_file='p-model-weights-big-9.h5')
    #targets, predictions = predict(model, 
    #             '../../grasp-conv/data/eye-perspectives',
    #             'perspective-data-big.pkl',
    #             valid_indices) 
    #with open('perspective-predictions-big-9.pkl', 'wb') as f:
    #    cPickle.dump((targets, predictions), f)


