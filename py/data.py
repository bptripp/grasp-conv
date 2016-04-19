__author__ = 'bptripp'

import os
import csv
import numpy as np
from itertools import islice
from depthmap import *
from PIL import Image
from depthmap import loadOBJ, Display
import cPickle

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


def make_depth_from_gripper(obj_filename, param_filename):
    """
    Make depth images from perspective of gripper.
    """
    verts, faces = loadOBJ(obj_filename)
    verts = np.array(verts)
    minz = np.min(verts, axis=0)[2]
    verts[:,2] = verts[:,2] + 0.2 - minz

    d = Display(imsize=(80,80))

    # positions = []
    # orientations = []
    labels = []
    depths = []
    c = 0
    for line in open(param_filename, "r"):
        print(c)
        if c == 100:
            break
        c = c + 1

        vals = line.split(',')
        gripper_pos = [float(vals[0]), float(vals[1]), float(vals[2])]
        gripper_orient = [float(vals[3]), float(vals[4]), float(vals[5])]
        labels.append(int(vals[6]))

        # new_verts = move_vertices(gripper_pos, gripper_orient, verts)
        d.set_camera_position(gripper_pos, gripper_orient, .3)
        # d.set_mesh(new_verts, faces)
        d.set_mesh(verts, faces) #this mut go after set_camera_position
        depth = d.read_depth()
        # depths.append(depth[100:300,100:300])
        depths.append(depth)

    d.close()
    return np.array(depths), np.array(labels)


def make_random_depths(obj_filename, param_filename, n, im_size=(40,40)):
    verts, faces = loadOBJ(obj_filename)
    verts = np.array(verts)
    minz = np.min(verts, axis=0)[2]
    verts[:,2] = verts[:,2] + 0.2 - minz

    points, labels = get_points(param_filename)

    d = Display(imsize=im_size)
    probs = []
    depths = []
    for i in range(n):
        point = get_interpolated_point(points)

        estimate, confidence = get_prob_label(points, labels, point)
        probs.append(estimate)

        gripper_pos = point[:3]
        gripper_orient = point[3:]
        d.set_camera_position(gripper_pos, gripper_orient, .3)
        d.set_mesh(verts, faces) #this must go after set_camera_position
        depth = d.read_depth()
        depths.append(depth)

    d.close()

    return np.array(depths), np.array(probs)


def get_interpolated_point(points):
    p1 = np.random.randint(0, len(points))
    p2 = get_closest_index(points, p1, prob_include=.5)
    mix_weight = np.random.rand()
    result = points[p1]*mix_weight + points[p2]*(1-mix_weight)
    # print(points[p1])
    # print(points[p2])
    # print(result)
    return result


def get_closest_index(points, index, prob_include=1):
    min_distance = 1000
    closest_index = []
    for i in range(len(points)):
        distance = np.linalg.norm(points[i] - points[index])
        if distance < min_distance and i != index and np.random.rand() < prob_include:
            min_distance = distance
            closest_index = i
    return closest_index


def get_points(param_filename):
    points = []
    labels = []
    for line in open(param_filename, "r"):
        vals = line.split(',')
        points.append([float(vals[0]), float(vals[1]), float(vals[2]),
            float(vals[3]), float(vals[4]), float(vals[5])])
        labels.append(int(vals[6]))
    return np.array(points), labels


def get_prob_label(points, labels, point, sigma_p=.001, sigma_a=(4*np.pi/180)):
    sigma_p_inv = sigma_p**-1
    sigma_a_inv = sigma_a**-1
    sigma_inv = np.diag([sigma_p_inv, sigma_p_inv, sigma_p_inv,
                         sigma_a_inv, sigma_a_inv, sigma_a_inv])
    differences = points - point
    weights = np.zeros(len(labels))
    for i in range(len(labels)):
        weights[i] = np.exp( -(1./2) * np.dot(differences[i,:], np.dot(sigma_inv, differences[i,:])) )
    estimate = np.sum(weights * np.array(labels).astype(float)) / np.sum(weights)
    confidence = np.sum(weights)
    # print(confidence)
    return estimate, confidence


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

    # source = AshleyDataSource()
    # # print(source.labels)
    # import numpy as np
    # print(np.mean(source.Y))
    # print(len(source.Y))
    # print(source.X.shape)

    # import matplotlib.pyplot as plt
    # plt.imshow(source.X[0,0,:,:])
    # plt.show()

    shapes = ['24_bowl-02-Mar-2016-07-03-29',
        '24_bowl-03-Mar-2016-22-54-50',
        '24_bowl-05-Mar-2016-13-53-41',
        '24_bowl-07-Mar-2016-05-06-04',
        '24_bowl-16-Feb-2016-10-12-27',
        '24_bowl-17-Feb-2016-22-00-34',
        '24_bowl-24-Feb-2016-17-38-53',
        '24_bowl-26-Feb-2016-08-35-29',
        '24_bowl-27-Feb-2016-23-52-43',
        '24_bowl-29-Feb-2016-15-01-53']

    # shapes = ['24_bowl-02-Mar-2016-07-03-29']

    n = 10 #TODO: check and increase
    for shape in shapes:
        depths, labels = make_random_depths('../data/obj_files/' + shape + '.obj',
                                            '../data/params/' + shape + '.csv',
                                            n, im_size=(40,40))

        f = file('../data/' + shape + '-random.pkl', 'wb')
        cPickle.dump((depths, labels), f)
        f.close()

    f = file('../data/' + shapes[0] + '-random.pkl', 'rb')
    (depths, labels) = cPickle.load(f)
    f.close()

    print(labels)

    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import axes3d, Axes3D

    depths = depths.astype(float)
    depths[depths > np.max(depths.flatten()) - 1] = np.NaN

    X = np.arange(0, depths.shape[1])
    Y = np.arange(0, depths.shape[2])
    X, Y = np.meshgrid(X, Y)
    fig = plt.figure(figsize=(12,6))
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    plt.xlabel('x')
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    plt.xlabel('x')
    # ax = Axes3D(fig)
    # for i in range(depths.shape[0]):
    s = 5 #pixel stride
    for i in range(n):
        if labels[i] > .5:
            color = 'g'
            ax = ax1
            ax.plot_wireframe(X[::s,::s], Y[::s,::s], depths[i,::s,::s], color=color)
        else:
            color = 'r'
            ax = ax2
            if np.random.rand(1) < .5:
                ax.plot_wireframe(X[::s,::s], Y[::s,::s], depths[i,::s,::s], color=color)
        # plt.title(str(i) + ': ' + str(labels[i]))
    plt.show()


    # shape = '24_bowl-02-Mar-2016-07-03-29'

    # points, labels = get_points('../data/params/' + shape + '.csv')
    # closest = get_closest_index(points, 0, prob_include=.5)
    # print(closest)

    # for i in range(20):
    #     point = get_interpolated_point(points)
    #     prob = get_prob_label(points, labels, point)
    #     print(prob)

    # print(points[0])
    # print(points[closest])

    # obj_filename = '../data/obj_files/' + shape + '.obj'
    # param_filename = '../data/params/' + shape + '.csv'
    # n = 15
    # depths, labels = make_random_depths(obj_filename, param_filename, n, im_size=(40,40))
    # print(labels)
    # for i in range(n):
    #     plt.subplot(3,n/3,i+1)
    #     plt.imshow(depths[i,:,:])
    #     plt.title('{:2.2f}'.format(labels[i]))
    # # plt.subplot(1,3,2)
    # # plt.imshow(depths[1,:,:])
    # # plt.subplot(1,3,3)
    # # plt.imshow(depths[2,:,:])
    # # print(labels)
    # plt.show()