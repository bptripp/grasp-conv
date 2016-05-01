__author__ = 'bptripp'

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D

"""
Initialization of CNNs via clustering of inputs and convex optimization
of outputs.
"""

from scipy.optimize import curve_fit


def sigmoid(x, centre, gain):
     y = 1 / (1 + np.exp(-gain*(x-centre)))
     return y


def gaussian(x, mu, sigma):
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sigma, 2.)))


def get_sigmoid_params(false_samples, true_samples, do_plot=False):
    """
    Find gain and bias for sigmoid function that approximates probability
    of class memberships. Probability based on Bayes' rule & gaussian
    model of samples from each class.
    """
    false_mu = np.mean(false_samples)
    false_sigma = np.std(false_samples)
    true_mu = np.mean(true_samples)
    true_sigma = np.std(true_samples)

    lowest = np.minimum(np.min(false_samples), np.min(true_samples))
    highest = np.maximum(np.max(false_samples), np.max(true_samples))
    a = np.arange(lowest, highest, (highest-lowest)/25)

    p_x_false = gaussian(a, false_mu, false_sigma)
    p_x_true = gaussian(a, true_mu, true_sigma)
    p_x = p_x_true + p_x_false
    p_true = p_x_true / p_x

    popt, _ = curve_fit(sigmoid, a, p_true)
    centre, gain = popt[0], popt[1]

    if do_plot:
        # plt.figure()
        plt.hist(false_samples, a)
        plt.hist(true_samples, a)
        plt.plot(a, 100*sigmoid(a, centre, gain))
        plt.plot(a, 100*p_true)
        plt.title('centre: ' + str(centre) + ' gain: ' + str(gain))
        plt.show()

    return centre, gain


def check_sigmoid():
    n = 1000
    false_samples = 1 + .3*np.random.randn(n)
    true_samples = -1 + 1*np.random.randn(n)
    centre, gain = get_sigmoid_params(false_samples, true_samples, do_plot=True)


def get_convolutional_prototypes(samples, shape, patches_per_sample=5):
    assert len(samples.shape) == 4
    assert len(shape) == 4

    wiggle = (samples.shape[2]-shape[2], samples.shape[3]-shape[3])
    patches = []
    for sample in samples:
        for i in range(patches_per_sample):
            corner = (np.random.randint(0, wiggle[0]), np.random.randint(0, wiggle[1]))
            patches.append(sample[:,corner[0]:corner[0]+shape[2],corner[1]:corner[1]+shape[3]])
    patches = np.array(patches)

    flat = np.reshape(patches, (patches.shape[0], -1))
    km = KMeans(shape[0])
    km.fit(flat)
    return np.reshape(km.cluster_centers_, shape)


def get_dense_prototypes(samples, n):
    km = KMeans(n)
    km.fit(samples)
    return km.cluster_centers_


def check_get_prototypes():
    samples = np.random.rand(1000, 2, 28, 28)
    prototypes = get_convolutional_prototypes(samples, (20,2,5,5))
    print(prototypes.shape)

    samples = np.random.rand(900, 2592)
    prototypes = get_dense_prototypes(samples, 64)
    print(prototypes.shape)


def get_discriminant(samples, labels):
    lda = LinearDiscriminantAnalysis(solver='eigen', shrinkage='auto')
    lda.fit(samples, labels)
    return lda.coef_[0]


def check_discriminant():
    n = 1000
    labels = np.random.rand(n) < 0.5
    samples = np.zeros((n,2))
    for i in range(len(labels)):
        if labels[i] > 0.5:
            samples[i,:] = np.array([0,1]) + 1*np.random.randn(1,2)
        else:
            samples[i,:] = np.array([-2,-1]) + .5*np.random.randn(1,2)

    coeff = get_discriminant(samples, labels)

    plt.figure(figsize=(10,5))
    plt.subplot(1,2,1)
    plt.scatter(samples[labels>.5,0], samples[labels>.5,1], color='g')
    plt.scatter(samples[labels<.5,0], samples[labels<.5,1], color='r')
    plt.plot([-coeff[0], coeff[0]], [-coeff[1], coeff[1]], color='k')
    plt.subplot(1,2,2)
    get_sigmoid_params(np.dot(samples[labels<.5], coeff),
                       np.dot(samples[labels>.5], coeff),
                       do_plot=True)
    plt.show()


def init_model(model, X_train, Y_train):
    for i in range(len(model.layers)):
        print(model.layers[i])
        if i == len(model.layers) - 1:
            pass
        elif isinstance(model.layers[i], Convolution2D):
            inputs = get_inputs(model, X_train, i)
            w, b = model.layers[i].get_weights()
            w = get_convolutional_prototypes(inputs, w.shape)
            b = .1 * np.ones_like(b)
            model.layers[i].set_weights([w,b])
        elif isinstance(model.layers[i], Dense):
            inputs = get_inputs(model, X_train, i)
            w, b = model.layers[i].get_weights()
            print(w.shape)
            print(inputs.shape)
            w = get_dense_prototypes(inputs, w.shape[1])
            b = .1 * np.ones_like(b)
            model.layers[i].set_weights([w,b])



def get_inputs(model, X_train, layer):
    if layer == 0:
        return X_train
    else:
        partial_model = Sequential(layers=model.layers[:layer])
        partial_model.compile('sgd', 'mse')
        return partial_model.predict(X_train)


# def init_features(model, layer, inputs):
#     w, b = model.layers[layer].get_weights()
#     print(w.shape)
#     prototypes = get_prototypes(inputs, w.shape[0])
#     print(prototypes.shape)
#     # scale = 1./np.sum(prototypes)


def init_log_loss(model, X_train, Y_train):
    pass


if __name__ == '__main__':
    # check_sigmoid()
    check_get_prototypes()
    # check_discriminant()

    assert False

    import cPickle
    f = file('../data/depths/24_bowl-29-Feb-2016-15-01-53.pkl', 'rb')
    d, bd, l = cPickle.load(f)
    f.close()
    print(d.shape)
    print(bd.shape)
    print(l.shape)

    n = 100
    f = file('../data/bowl-test.pkl', 'wb')
    cPickle.dump((d[:n,:,:], bd[:n,:,:], l[:n]), f)
    f.close()

    import cPickle
    f = file('../data/bowl-test.pkl', 'rb')
    d, bd, l = cPickle.load(f)
    f.close()

    X_train = np.zeros((90,1,80,80))
    X_train[:,0,:,:] = d[:90,:,:]
    Y_train = l[:90]

    model = Sequential()
    model.add(Convolution2D(16,5,5,input_shape=(1,80,80)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D())
    model.add(Convolution2D(8,3,3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D())
    model.add(Flatten())
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    init_model(model, X_train, Y_train)

    #TODO: scale appropriately to output is standard normal