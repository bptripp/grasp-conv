__author__ = 'bptripp'

from os.path import join
import cPickle
import matplotlib.pyplot as plt
import scipy.misc
import numpy as np
from perspective import get_rotation_matrix

def plot_correct_point_scatter():
    n_points = 200

    with open('../data/neuron-points.pkl', 'rb') as f:
        neuron_points, neuron_angles = cPickle.load(f)

    with open('perspective-data-small.pkl', 'rb') as f:
        image_files, metrics = cPickle.load(f)

    # print(metrics.shape)

    index = 600

    points = neuron_points[:,:n_points]
    offsets = np.zeros_like(points)
    for i in range(n_points):
        r = get_rotation_matrix(neuron_points[:,i], neuron_angles[:,i])
        offsets[:,i] = points[:,i] + np.dot(r, np.array([0,.005,0]))

    from mpl_toolkits.mplot3d import axes3d, Axes3D
    fig = plt.figure(figsize=(20,10))

    ax = fig.add_subplot(1,2,1,projection='3d')
    ax.scatter(points[0,:], points[1,:], points[2,:],
               c=metrics[index,:n_points], cmap='autumn', depthshade=False)
    ax.scatter(offsets[0,:], offsets[1,:], offsets[2,:],
               c=metrics[index,:n_points], cmap='autumn', depthshade=False, s=10)

    plt.subplot(1,2,2)
    image = scipy.misc.imread(join('../../grasp-conv/data/eye-perspectives', image_files[index]))
    plt.imshow(image)
    plt.title(image_files[index])
    plt.show()

    # print(neuron_points[:,:n_points])


def plot_predictions():
    with open('perspective-predictions.pkl') as f:
        targets, predictions = cPickle.load(f)
    # print(targets.shape)
    # print(predictions.shape)

    plt.figure()
    for i in range(25):
        plt.subplot(5,5,i)
        plt.scatter(targets[:,i], predictions[:,i])
        c = np.corrcoef(targets[:,i], predictions[:,i])[0,1]
        # plt.title(c)
        print(c)
    plt.show()

if __name__ == '__main__':
    # plot_correct_point_scatter()
    plot_predictions()