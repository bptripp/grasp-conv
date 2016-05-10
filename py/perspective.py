__author__ = 'bptripp'

import numpy as np
import matplotlib.pyplot as plt
import cPickle
from scipy.optimize import bisect


def get_random_points(n, radius, surface=False):
    point_directions = np.random.randn(3, n)
    norms = np.sum(point_directions**2, axis=0)**.5
    points = radius * point_directions / norms

    if not surface:
        points = points * np.random.rand(n)**(1./3.)

    return points


def get_random_angles(n):
    """
    :param n: Number of angles needed
    :return: Random angles in restricted ranges, meant as deviations in perspective around
        looking staight at something.
    """
    angles = np.pi/8.*np.random.randn(3, n)
    angles[2,:] = 2*np.pi*np.random.rand(1, n)
    return angles


def get_rotation_matrix(point, angle):
    """
    :param point: Location of camera
    :param angle: Not what you expect: this is a list of angles relative to looking
        at (0,0,0), about world-z (azimuth), camera-y (elevation), and camera-z (roll).
        Random samples are produced by get_random_angles().
    :return: just what you expect
    """
    z = -point #location of (0,0,0) relative to point

    alpha = np.arctan(z[1]/z[0])
    if z[0] < 0: alpha = alpha + np.pi
    if alpha < 0: alpha = alpha + 2.*np.pi
    alpha = alpha + angle[0]

    # rotate by alpha about z
    Rz = np.array([[np.cos(alpha), -np.sin(alpha), 0], [np.sin(alpha), np.cos(alpha), 0], [0, 0, 1]])

    # find elevation in new coordinates
    beta = -np.arctan(np.sqrt(z[0]**2+z[1]**2)/z[2])
    if z[2] < 0: beta = beta + np.pi
    if beta < 0: beta = beta + 2.*np.pi
    beta = beta + angle[1]

    # rotate by beta about y
    Ry = np.array([[np.cos(beta), 0, -np.sin(beta)], [0, 1, 0], [np.sin(beta), 0, np.cos(beta)]])

    gamma = angle[2]
    Rz2 = np.array([[np.cos(-gamma), -np.sin(-gamma), 0], [np.sin(-gamma), np.cos(-gamma), 0], [0, 0, 1]])

    return np.dot(Rz, np.dot(Ry, Rz2))


def to_quaternion(rotation_matrix):
    # from Siciliano & Khatib pg. 12
    r = rotation_matrix
    e0 = .5 * np.sqrt(1 + r[0][0] + r[1][1] + r[2][2])
    e1 = (r[2][1] - r[1][2]) / (4*e0)
    e2 = (r[0][2] - r[2][0]) / (4*e0)
    e3 = (r[1][0] - r[0][1]) / (4*e0)
    return np.array([e0,e1,e2,e3])


def from_quaternion(e):
    # from http://www.euclideanspace.com/maths/geometry/rotations/conversions/quaternionToMatrix/
    rm = [[1-2*e[2]**2-2*e[3]**2, 2*e[1]*e[2]-2*e[3]*e[0], 	2*e[1]*e[3]+2*e[2]*e[0]],
          [2*e[1]*e[2]+2*e[3]*e[0], 1-2*e[1]**2-2*e[3]**2, 2*e[2]*e[3]-2*e[1]*e[0]],
          [2*e[1]*e[3]-2*e[2]*e[0], 2*e[2]*e[3]+2*e[1]*e[0], 1-2*e[1]**2-2*e[2]**2]]
    return np.array(rm)


def check_rotation_matrix(scatter=False):
    from mpl_toolkits.mplot3d import axes3d, Axes3D

    n = 6
    points = get_random_points(n, 2)
    angles = get_random_angles(n)

    # point = np.array([1,1e-6,1e-6])
    # point = np.array([1e-6,1,1e-6])

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1,projection='3d')
    for i in range(points.shape[1]):
        point = points[:,i]
        angle = angles[:,i]
        if not scatter:
            angle[0] = 0
            angle[1] = 0

        R = get_rotation_matrix(point, angle)
        ax.scatter(0, 0, 0, color='b')
        ax.scatter(point[0], point[1], point[2], color='r')
        x = np.dot(R, np.array([1,0,0]))
        y = np.dot(R, np.array([0,1,0]))
        z = np.dot(R, np.array([0,0,1]))
        ax.plot([point[0],point[0]+x[0]], [point[1],point[1]+x[1]], [point[2],point[2]+x[2]], color='r')
        ax.plot([point[0],point[0]+y[0]], [point[1],point[1]+y[1]], [point[2],point[2]+y[2]], color='g')
        ax.plot([point[0],point[0]+z[0]], [point[1],point[1]+z[1]], [point[2],point[2]+z[2]], color='b')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.ylabel('z')
    if not scatter:
        plt.title('blue axes should point AT blue dot (zero)')
    else:
        plt.title('blue axes should point NEAR blue dot (zero)')
    plt.show()


def check_quaternion():
    r = get_rotation_matrix(np.array([.1, -.2, .3]), np.array([.1, .4, 1.5]))
    error = r - from_quaternion(to_quaternion(r))
    assert np.std(error.flatten()) < 1e-6


def check_depth_from_random_perspective():
    from depthmap import loadOBJ, Display
    filename = '../data/obj_files/24_bowl-02-Mar-2016-07-03-29.obj'
    verts, faces = loadOBJ(filename)

    # put vertical centre at zero
    verts = np.array(verts)
    minz = np.min(verts, axis=0)[2]
    maxz = np.max(verts, axis=0)[2]
    verts[:,2] = verts[:,2] - (minz+maxz)/2

    n = 6
    points = get_random_points(n, .25)
    angles = get_random_angles(n)
    point = points[:,0]
    angle = angles[:,0]

    rot = get_rotation_matrix(point, angle)

    im_width = 80
    d = Display(imsize=(im_width,im_width))
    d.set_camera_position(point, rot, .5)
    d.set_mesh(verts, faces)
    depth = d.read_depth()
    d.close()

    X = np.arange(0, im_width)
    Y = np.arange(0, im_width)
    X, Y = np.meshgrid(X, Y)

    from mpl_toolkits.mplot3d import axes3d, Axes3D
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1,projection='3d')
    ax.plot_wireframe(X, Y, depth)
    ax.set_xlabel('x')
    plt.show()


def find_vertical(point):
    """
    Find new angle[2] so that camera-up points up. In terms of rotation matrix,
    R[2,0] should be 0 (x-axis horizontal) and R[2,1] should be positive (pointing
    up rather than down).
    """

    def f(gamma):
        return get_rotation_matrix(point, np.array([0, 0, gamma]))[2][0]

    gamma = bisect(f, 0, np.pi)

    if get_rotation_matrix(point, np.array([0, 0, gamma]))[2][1] < 0:
        gamma = gamma + np.pi

    return gamma


def check_find_vertical():
    n = 10
    points = get_random_points(n, .35, surface=True)
    for i in range(n):
        point = points[:,i]
        gamma = find_vertical(point)
        rot = get_rotation_matrix(point, np.array([0, 0, gamma]))
        if np.abs(rot[2,0] > 1e-6) or rot[2,1] < 0:
            print('error with gamma: ' + str(gamma) + ' should be 0: ' + str(rot[2,0]) + ' should be +ve: ' + str(rot[2,1]))


def plot_random_samples():
    n = 1000
    points = get_random_points(n, .25)
    angles = get_random_angles(n)

    from mpl_toolkits.mplot3d import axes3d, Axes3D
    fig = plt.figure(figsize=(10,5))
    ax = fig.add_subplot(1,2,1,projection='3d')
    ax.scatter(points[0,:], points[1,:], points[2,:])
    ax = fig.add_subplot(1,2,2,projection='3d')
    ax.scatter(angles[0,:], angles[1,:], angles[2,:])
    plt.show()


def get_perspectives(obj_filename, points, angles, im_width=80, near_clip=.25, far_clip=0.8, fov=45, camera_offset=.45):
    from depthmap import loadOBJ, Display, get_distance
    verts, faces = loadOBJ(obj_filename)

    # put vertical centre at zero
    verts = np.array(verts)
    min_bounding_box = np.min(verts, axis=0)
    max_bounding_box = np.max(verts, axis=0)

    # set bounding box centre to 0,0,0
    verts[:,0] = verts[:,0] - (min_bounding_box[0]+max_bounding_box[0])/2.
    verts[:,1] = verts[:,1] - (min_bounding_box[1]+max_bounding_box[1])/2.
    verts[:,2] = verts[:,2] - (min_bounding_box[2]+max_bounding_box[2])/2.

    d = Display(imsize=(im_width,im_width))
    d.set_perspective(fov=fov, near_clip=near_clip, far_clip=far_clip)
    perspectives = np.zeros((points.shape[1],im_width,im_width), dtype='float32')
    for i in range(points.shape[1]):
        point = points[:,i]
        angle = angles[:,i]
        rot = get_rotation_matrix(point, angle)
        d.set_camera_position(point, rot, camera_offset)
        d.set_mesh(verts, faces)
        depth = d.read_depth()
        distance = get_distance(depth, near_clip, far_clip)
        perspectives[i,:,:] = distance
    d.close()
    return perspectives


def process_directory(obj_dir, data_dir, n):
    from os import listdir
    from os.path import isfile, join
    import time

    for f in listdir(obj_dir):
        obj_filename = join(obj_dir, f)
        if isfile(obj_filename) and f.endswith('.obj'):
            data_filename = join(data_dir, f[:-4] + '.pkl')
            if isfile(data_filename):
                print('Skipping ' + f)
            else:
                print('Processing ' + f)
                start_time = time.time()
                points = get_random_points(n, .25)
                angles = get_random_angles(n)

                perspectives = get_perspectives(obj_filename, points, angles)

                f = open(data_filename, 'wb')
                cPickle.dump((points, angles, perspectives), f)
                f.close()
                print('   ' + str(time.time()-start_time) + 's')


def process_eye_directory(obj_dir, data_dir, n):
    #TODO: save image files here to allow random ordering during training

    from os import listdir
    from os.path import isfile, join
    import time

    for f in listdir(obj_dir):
        obj_filename = join(obj_dir, f)
        if isfile(obj_filename) and f.endswith('.obj'):
            data_filename = join(data_dir, f[:-4] + '.pkl')
            if isfile(data_filename):
                print('Skipping ' + f)
            else:
                print('Processing ' + f)
                start_time = time.time()
                points = get_random_points(n, .35, surface=True) #.75m with offset
                angles = np.zeros_like(points)

                # Set camera-up to vertical via third angle (angle needed is always
                # 3pi/4, but we'll find it numerically in case other parts of code
                # change while we're not looking).
                for i in range(n):
                    angles[2,i] = find_vertical(points[:,i])

                perspectives = get_perspectives(obj_filename, points, angles, near_clip=.4, fov=30)

                f = open(data_filename, 'wb')
                cPickle.dump((points, angles, perspectives), f)
                f.close()
                print('   ' + str(time.time()-start_time) + 's')


def check_maps(data_dir):
    """
    Checks pkl files in given directory to see if any of the depth maps they contain
    are empty. 
    """
    from os import listdir
    from os.path import isfile, join
    for f in listdir(data_dir):
        data_filename = join(data_dir, f)
        if isfile(data_filename) and f.endswith('.pkl'):
            print('Checking ' + f)
            f = open(data_filename, 'rb')
            (points, angles, perspectives) = cPickle.load(f)
            f.close()

            for i in range(perspectives.shape[0]):
                sd = np.std(perspectives[i,:,:].flatten())
                if sd < 1e-3:
                    print('   map ' + str(i) + ' is empty')


if __name__ == '__main__':
    # check_rotation_matrix(scatter=True)
    # check_quaternion()
    # check_depth_from_random_perspective()
    # plot_random_samples()
    # check_find_vertical()

    # process_directory('../data/obj_files/', '../data/perspectives/', 10)
    process_directory('../../grasp-conv/data/obj_files/', '../../grasp-conv/data/perspectives/', 3000)
    # process_eye_directory('../../grasp-conv/data/obj_files/', '../../grasp-conv/data/eye-perspectives/', 100)
    # check_maps('../../grasp-conv/data/perspectives/')

    # points = get_random_points(3, .35, surface=True)
    # print(points)
    # angles = np.zeros_like(points)
    # rot = get_rotation_matrix(np.array([-.047, -.249, -.241]), np.array([0, 0, -1.57+3.141]))
    # print(rot)
    # print(rot[0][0]**2 + rot[1][0]**2 + rot[2][0]**2)

    # import time

    # obj_name = '24_bowl-02-Mar-2016-07-03-29'
    # obj_filename = '../data/obj_files/' + obj_name + '.obj'
    # n = 100
    # points = get_random_points(n, .25)
    # angles = get_random_angles(n)
    #
    # # start_time = time.time()
    # perspectives = get_perspectives(obj_filename, points, angles)
    # # gen_time = time.time() - start_time
    # # print(gen_time)
    #
    # perspective_filename = '../data/perspectives/' + obj_name + '.pkl'
    # f = open(perspective_filename, 'wb')
    # cPickle.dump(perspectives, f)
    # f.close()
    # # save_time = time.time() - start_time - gen_time
    # # print(save_time)

    # X = np.arange(0, 80)
    # Y = np.arange(0, 80)
    # X, Y = np.meshgrid(X, Y)
    #
    # from mpl_toolkits.mplot3d import axes3d, Axes3D
    # fig = plt.figure()
    # ax = fig.add_subplot(1,1,1,projection='3d')
    # ax.plot_wireframe(X, Y, perspectives[0,:,:])
    # ax.set_xlabel('x')
    # plt.show()

