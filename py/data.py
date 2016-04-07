__author__ = 'bptripp'

import os
import csv
import numpy as np
from itertools import islice
from depthmap import *
from PIL import Image

class GraspDataSource(object):

    def __init__(self, csv_file_name, obj_directory_name, range=None, imsize=(50,50)):
        self.obj_directory_name = obj_directory_name
        self.imsize = imsize

        self.objfiles = []
        self.orientations = []
        self.positions = []
        self.success = []

        with open(csv_file_name, 'rb') as csvfile:
            r = csv.reader(csvfile, delimiter=',')
            for row in islice(r, 1, None):
                self.objfiles.append(obj_directory_name + os.path.sep + row[0])
                self.orientations.append([float(row[1]), float(row[2]), float(row[3])])
                self.positions.append([float(row[4]), float(row[5]), float(row[6])])
                self.success.append(float(row[8]))

        if range is not None:
            self.objfiles = self.objfiles[range[0]:range[1]]
            self.orientations = self.orientations[range[0]:range[1]]
            self.positions = self.positions[range[0]:range[1]]
            self.success = self.success[range[0]:range[1]]

        self.display = Display(imsize=imsize)


    def get_XY(self, n):
        ind = np.random.randint(0, len(self.objfiles), n)
        X = np.zeros((n, 1, self.imsize[0], self.imsize[1]))
        Y = np.zeros(n)
        for i in range(n):
            # print(self.objfiles[ind[i]])
            verts, faces = loadOBJ(self.objfiles[ind[i]])
            new_verts = move_vertices(self.positions[ind[i]], self.orientations[ind[i]], verts)
            self.display.set_mesh(new_verts, faces)
            X[i,0,:,:] = self.display.read_depth()
            Y[i] = self.success[ind[i]]
        return np.array(X), np.array(Y)


class AshleyDataSource(object):
    def __init__(self):
        self.X = np.zeros((1000,1,28,28))
        for i in range(1000):
            # filename = '25_mug-02-Feb-2016-12-40-43.obj131001'
            filename = '../data/imgs/25_mug-02-Feb-2016-12-40-43.obj13' + str(1001+i) + '.png'
            im = Image.open(filename)
            self.X[i,0,:,:] = np.array(im.getdata()).reshape((28,28))

        for line in open('../data/labels.csv', "r"):
            vals = line.split(',')
            self.Y = map(int, vals)

        # normalize input
        self.X = (self.X - np.mean(self.X.flatten())) / np.std(self.X.flatten())


if __name__ == '__main__':
    # source = GraspDataSource('/Users/bptripp/code/grasp-conv/data/output_data.csv',
    #                     '/Users/bptripp/code/grasp-conv/data/obj_files',
    #                     range=(10000,10500))
    #
    # import time
    # start_time = time.time()
    # X, Y = source.get_XY(100)
    # print(time.time() - start_time)
    #
    # print(X.shape)
    # print(Y)

    source = AshleyDataSource()
    # print(source.labels)
    import numpy as np
    print(np.mean(source.Y))
    print(len(source.Y))
    print(source.X.shape)

    # import matplotlib.pyplot as plt
    # plt.imshow(source.X[0,0,:,:])
    # plt.show()
