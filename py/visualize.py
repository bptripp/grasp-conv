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