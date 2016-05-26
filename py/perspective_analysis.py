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
    with open('perspective-predictions-better.pkl') as f:
        targets, predictions = cPickle.load(f)
    # print(targets.shape)
    # print(predictions.shape)

    plt.figure(figsize=(9,6))
    for i in range(25):
        plt.subplot(5,5,i)
        plt.scatter(targets[:,i], predictions[:,i], s=1)
        c = np.corrcoef(targets[:,i], predictions[:,i])[0,1]
        plt.gca().axes.xaxis.set_ticks([])
        plt.gca().axes.yaxis.set_ticks([])
        print(c)
    plt.tight_layout()
    plt.show()


def plot_points_with_correlations():
    with open('perspective-predictions.pkl') as f:
        targets, predictions = cPickle.load(f)

    n_points = targets.shape[1]

    r = []
    for i in range(n_points):
        r.append(np.corrcoef(targets[:,i], predictions[:,i])[0,1])

    with open('perspective-predictions-better.pkl') as f:
        targets, predictions = cPickle.load(f)

    r_better = []
    for i in range(n_points):
        r_better.append(np.corrcoef(targets[:,i], predictions[:,i])[0,1])

    with open('../data/neuron-points.pkl', 'rb') as f:
        neuron_points, neuron_angles = cPickle.load(f)

    # from mpl_toolkits.mplot3d import axes3d, Axes3D
    # fig = plt.figure()
    # ax = fig.add_subplot(1,1,1,projection='3d')
    # ax.scatter(neuron_points[0,:n_points], neuron_points[1,:n_points], neuron_points[2,:n_points],
    #            c=r, cmap='autumn', depthshade=False, s=40)
    # plt.show()

    angles_from_vertical = []
    for i in range(n_points):
        # hor = np.sqrt(neuron_points[0,i]**2 + neuron_points[1,i]**2)
        # ver = neuron_points[2,i]
        # angles_from_vertical.append(np.arctan(hor / ver))
        norm = np.linalg.norm(neuron_points[:,i])
        angles_from_vertical.append(np.arccos(neuron_points[2,i] / norm))

    fig = plt.figure()
    plt.scatter(angles_from_vertical, r, color='r', s=10)
    plt.scatter(angles_from_vertical, r_better, color='b', s=50)
    plt.xlim([0,np.pi])
    plt.ylim([0,1])
    plt.tick_params(axis='both', labelsize=18)
    plt.xlabel('angle between eye and hand (rad)', fontsize=18)
    plt.ylabel('target-prediction correlation', fontsize=18)
    plt.show()


if __name__ == '__main__':
    # plot_correct_point_scatter()
    plot_predictions()
    # plot_points_with_correlations()