__author__ = 'bptripp'

import numpy as np
from scipy.optimize import curve_fit
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
    kernels = km.cluster_centers_

    # normalize product of centre and corresponding kernel
    for i in range(kernels.shape[0]):
        kernels[i,:] = kernels[i,:] / np.linalg.norm(kernels[i,:])

    return np.reshape(kernels, shape)


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
    if not (isinstance(model.layers[-1], Activation) \
            and model.layers[-1].activation.__name__ == 'sigmoid'\
            and isinstance(model.layers[-2], Dense)):
        raise Exception('This does not look like an LDA-compatible network, which is all we support')

    for i in range(len(model.layers)-2):
        if isinstance(model.layers[i], Convolution2D):
            inputs = get_inputs(model, X_train, i)
            w, b = model.layers[i].get_weights()
            w = get_convolutional_prototypes(inputs, w.shape)
            b = .1 * np.ones_like(b)
            model.layers[i].set_weights([w,b])
        if isinstance(model.layers[i], Dense):
            inputs = get_inputs(model, X_train, i)
            w, b = model.layers[i].get_weights()
            w = get_dense_prototypes(inputs, w.shape[1]).T
            b = .1 * np.ones_like(b)
            model.layers[i].set_weights([w,b])

    inputs = get_inputs(model, X_train, len(model.layers)-3)
    coeff = get_discriminant(inputs, Y_train)
    centre, gain = get_sigmoid_params(np.dot(inputs[Y_train<.5], coeff),
                       np.dot(inputs[Y_train>.5], coeff))
    w = coeff*gain
    w = w[:,np.newaxis]
    b = np.array([-centre])
    model.layers[-2].set_weights([w,b])
    sigmoid_inputs = get_inputs(model, X_train, len(model.layers)-1)

    plt.figure()
    plt.subplot(2,1,1)
    bins = np.arange(np.min(Y_train), np.max(Y_train))
    plt.hist(sigmoid_inputs[Y_train<.5])
    plt.subplot(2,1,2)
    plt.hist(sigmoid_inputs[Y_train>.5])
    plt.show()


def get_inputs(model, X_train, layer):
    if layer == 0:
        return X_train
    else:
        partial_model = Sequential(layers=model.layers[:layer])
        partial_model.compile('sgd', 'mse')
        return partial_model.predict(X_train)


if __name__ == '__main__':
    # check_sigmoid()
    # check_get_prototypes()
    # check_discriminant()

    import cPickle
    f = file('../data/bowl-test.pkl', 'rb')
    # f = file('../data/depths/24_bowl-29-Feb-2016-15-01-53.pkl', 'rb')
    d, bd, l = cPickle.load(f)
    f.close()

    d = d - np.mean(d.flatten())
    d = d / np.std(d.flatten())

    # n = 900
    n = 90
    X_train = np.zeros((n,1,80,80))
    X_train[:,0,:,:] = d[:n,:,:]
    Y_train = l[:n]

    model = Sequential()
    model.add(Convolution2D(64,9,9,input_shape=(1,80,80)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D())
    # model.add(Convolution2D(64,3,3))
    # model.add(Activation('relu'))
    # model.add(MaxPooling2D())
    model.add(Flatten())
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    init_model(model, X_train, Y_train)

    # from visualize import plot_kernels
    # plot_kernels(model.layers[0].get_weights()[0])
