__author__ = 'bptripp'

# Experimenting with brain-inspired disparity tuning

import numpy as np
import cPickle
#import matplotlib.pyplot as plt

def disp_tuning(depth, centres, width):
    X = np.zeros((len(centres), depth.shape[0], depth.shape[1]))
    for i in range(len(centres)):
        X[i,:,:] = np.exp(-(depth-centres[i])**2/2/width**2)
    return X

if __name__ == '__main__':
    shapes = ['24_bowl-02-Mar-2016-07-03-29']
    f = file('../data/' + shapes[0] + '.pkl', 'rb')
    (depths, labels) = cPickle.load(f)
    f.close()

    # plt.hist(depths.flatten(), 50)
    # plt.show()

    centres = np.arange(1, 1.9, .1)
    width = .1

    # print(depths.shape)
    X = disp_tuning(depths[0,:,:], centres, width)
    print(X.shape)
