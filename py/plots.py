__author__ = 'bptripp'

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D
from data import get_prob_label, get_points
from depthmap import rot_matrix, loadOBJ

def plot_success_prob():
    shape = '24_bowl-02-Mar-2016-07-03-29'
    param_filename = '../data/params/' + shape + '.csv'
    points, labels = get_points(param_filename)

    # z = np.linspace(-4 * np.pi, 4 * np.pi, 300)
    # x = np.cos(z)
    # y = np.sin(z)
    # z = [1, 2]
    # x = [1, 2]
    # y = [1, 2]
    #
    # fig = plt.figure()
    # ax = fig.gca(projection='3d')
    # ax.plot(x, y, z,
    #         color = 'Blue',      # colour of the curve
    #         linewidth = 3,            # thickness of the line
    #         )
    fig = plt.figure()
    ax = Axes3D(fig)
    red = np.array([1.0,0.0,0.0])
    green = np.array([0.0,1.0,0.0])

    obj_filename = '../data/obj_files/' + shape + '.obj'
    verts, faces = loadOBJ(obj_filename)
    verts = np.array(verts)
    minz = np.min(verts, axis=0)[2]
    verts[:,2] = verts[:,2] + 0.2 - minz

    show_flags = np.random.rand(verts.shape[0]) < .25
    ax.scatter(verts[show_flags,0], verts[show_flags,1], verts[show_flags,2], c='b')

    for i in range(1000):
        point = points[i]
        prob, confidence = get_prob_label(points, labels, point, sigma_p=1.5*.001, sigma_a=1.5*(4*np.pi/180))
        goodness = prob * np.minimum(1, .5 + .15*confidence)

        rm = rot_matrix(point[3], point[4], point[5])
        pos = point[:3]
        front = pos + np.dot(rm,[0,0,.15])
        left = pos - np.dot(rm,[0.0,0.005,0])
        right = pos + np.dot(rm,[0.0,0.01,0])

        # if goodness > .85:
        if prob > .9:
            ax.plot([pos[0],front[0]], [pos[1],front[1]], [pos[2],front[2]],
                    color=(prob*green + (1-prob)*red),
                    linewidth=confidence)
            ax.plot([left[0],right[0]], [left[1],right[1]], [left[2],right[2]],
                    color=(prob*green + (1-prob)*red),
                    linewidth=confidence)
    plt.show()

if __name__ == '__main__':
    plot_success_prob()

