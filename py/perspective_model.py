__author__ = 'bptripp'

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

num_outputs = 25

def merge_data(rel_dir):
    image_filenames = []
    neuron_metrics = []

    for f in listdir(rel_dir):
        rel_filename = join(rel_dir, f)
        if isfile(rel_filename) and f.endswith('.pkl') and not f == 'neuron-points.pkl':
            object_name = f[:-11]
            print('Processing ' + object_name)

            with open(rel_filename) as rel_file:
                target_indices, target_points, object_eye_points, object_eye_angles, neuron_metrics_for_object = cPickle.load(rel_file)

            s = neuron_metrics_for_object.shape
            for i in range(s[0]): #loop through target points
                for j in range(s[1]): #loop through eye perspectives
                    image_filename = object_name + '-' + str(i) + '-' + str(j) + '.png'
                    image_filenames.append(image_filename)
                    neuron_metrics.append(neuron_metrics_for_object[i,j,:])

    with open('perspective-data.pkl', 'wb') as f:
        cPickle.dump((image_filenames, np.array(neuron_metrics)), f)


def get_model():
    model = Sequential()
    model.add(Convolution2D(64, 7, 7, input_shape=(1,80,80), init='glorot_normal', border_mode='same'))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, 3, 3, init='glorot_normal', border_mode='same'))
    model.add(Activation('relu'))
    # model.add(Dropout(.5))
    model.add(Convolution2D(64, 3, 3, init='glorot_normal', border_mode='same'))
    model.add(Activation('relu'))
    # model.add(Dropout(.5))
    model.add(Flatten())
    model.add(Dense(1024))
    model.add(Activation('relu'))
    model.add(Dropout(.5))
    model.add(Dense(1024))
    model.add(Activation('relu'))
    model.add(Dropout(.5))
    model.add(Dense(num_outputs))

    adam = Adam(lr=0.000001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    model.compile(loss='mse', optimizer=adam)

    return model


def load_model():
    model = model_from_json(open('p-model-architecture.json').read())
    model.load_weights('p-model-weights.h5')

    adam = Adam(lr=0.000001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    model.compile(loss='mse', optimizer=adam)

    return model


def load_data(data_file):
    with open(data_file, 'rb') as f:
        image_filenames, neuron_metrics = cPickle.load(f)
    return image_filenames, neuron_metrics[:,:num_outputs]


def check_data():
    image_filenames, neuron_metrics = load_data('perspective-data-small.pkl')

    print(len(image_filenames))
    print(neuron_metrics.shape)

    print(image_filenames[0])
    print(neuron_metrics[0])
    print(np.min(neuron_metrics))
    print(np.max(neuron_metrics))
    print(np.mean(neuron_metrics))
    print(np.std(neuron_metrics))


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


def train_model(model, data_file, image_dir, train_indices, valid_indices):
    image_filenames, neuron_metrics = load_data(data_file)

    #X_valid = []
    #Y_valid = []
    #for ind in valid_indices:
    #    X_valid.append(get_input(image_dir, image_filenames[ind])[np.newaxis,:])
    #    Y_valid.append(neuron_metrics[ind])
    #X_valid = np.array(X_valid)
    #Y_valid = np.array(Y_valid)
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

    h = model.fit_generator(generate_XY(),
        samples_per_epoch=8192, nb_epoch=200,
        validation_data=(X_valid, Y_valid))

    with file('p-history.pkl', 'wb') as f:
        cPickle.dump(h.history, f)

    json_string = model.to_json()
    open('p-model-architecture.json', 'w').write(json_string)
    model.save_weights('p-model-weights.h5', overwrite=True)


def predict(model, image_dir, data_file, indices):
    image_filenames, neuron_metrics = load_data(data_file)
    X, Y = get_XY(image_dir, image_filenames, neuron_metrics, indices)
    return Y, model.predict(X, batch_size=32, verbose=0)


if __name__ == '__main__':
    # merge_data('/Volumes/TrainingData/grasp-conv/data/relative-small/')
    # check_data()

    valid_indices = np.arange(0, 50000, 100)
    s = set(valid_indices)
    train_indices = [x for x in range(75000) if x not in s]

    #model = get_model()
    #train_model(model,
    #            'perspective-data-big.pkl',
    #            '../../grasp-conv/data/eye-perspectives',
    #            train_indices,
    #            valid_indices)

    model = load_model()
    targets, predictions = predict(model, 
                 '../../grasp-conv/data/eye-perspectives',
                 'perspective-data-big.pkl',
                 valid_indices) 
    with open('perspective-predictions.pkl', 'wb') as f:
        cPickle.dump((targets, predictions), f)


