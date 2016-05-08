__author__ = 'bptripp'

import numpy as np
import matplotlib.pyplot as plt

def plot_kernels(weights):
    print(weights.shape)
    side = int(np.ceil(np.sqrt(weights.shape[0])))
    print(side)

    plt.figure()
    for i in range(weights.shape[0]):
        plt.subplot(side,side,i)
        plt.imshow(weights[i,0,:,:])
        plt.axis('off')
    plt.tight_layout()
    plt.show()


def plot_mesh(matrix):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import axes3d, Axes3D

    im_width = matrix.shape[0]
    fig = plt.figure()
    X = np.arange(0, im_width)
    Y = np.arange(0, im_width)
    X, Y = np.meshgrid(X, Y)
    ax = fig.add_subplot(1,1,1,projection='3d')
    ax.plot_wireframe(X, Y, matrix)
    plt.show()
