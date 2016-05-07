__author__ = 'bptripp'

import os
import csv
import numpy as np
from itertools import islice
from depthmap import *
from PIL import Image
import scipy
import scipy.misc
from depthmap import loadOBJ, Display
from heuristic import calculate_metric_map
import cPickle
import matplotlib.pyplot as plt


# class GraspDataSource(object):
#
#     def __init__(self, csv_file_name, obj_directory_name, range=None, imsize=(50,50)):
#         self.obj_directory_name = obj_directory_name
#         self.imsize = imsize
#
#         self.objfiles = []
#         self.orientations = []
#         self.positions = []
#         self.success = []
#
#         with open(csv_file_name, 'rb') as csvfile:
#             r = csv.reader(csvfile, delimiter=',')
#             for row in islice(r, 1, None):
#                 self.objfiles.append(obj_directory_name + os.path.sep + row[0])
#                 self.orientations.append([float(row[1]), float(row[2]), float(row[3])])
#                 self.positions.append([float(row[4]), float(row[5]), float(row[6])])
#                 self.success.append(float(row[8]))
#
#         if range is not None:
#             self.objfiles = self.objfiles[range[0]:range[1]]
#             self.orientations = self.orientations[range[0]:range[1]]
#             self.positions = self.positions[range[0]:range[1]]
#             self.success = self.success[range[0]:range[1]]
#
#         self.display = Display(imsize=imsize)
#
#
#     def get_XY(self, n):
#         ind = np.random.randint(0, len(self.objfiles), n)
#         X = np.zeros((n, 1, self.imsize[0], self.imsize[1]))
#         Y = np.zeros(n)
#         for i in range(n):
#             # print(self.objfiles[ind[i]])
#             verts, faces = loadOBJ(self.objfiles[ind[i]])
#             new_verts = move_vertices(self.positions[ind[i]], self.orientations[ind[i]], verts)
#             self.display.set_mesh(new_verts, faces)
#             X[i,0,:,:] = self.display.read_depth()
#             Y[i] = self.success[ind[i]]
#         return np.array(X), np.array(Y)


# class AshleyDataSource(object):
#     def __init__(self):
#         self.X = np.zeros((1000,1,28,28))
#         for i in range(1000):
#             # filename = '25_mug-02-Feb-2016-12-40-43.obj131001'
#             filename = '../data/imgs/25_mug-02-Feb-2016-12-40-43.obj13' + str(1001+i) + '.png'
#             im = Image.open(filename)
#             self.X[i,0,:,:] = np.array(im.getdata()).reshape((28,28))
#
#         for line in open('../data/labels.csv', "r"):
#             vals = line.split(',')
#             self.Y = map(int, vals)
#
#         # normalize input
#         self.X = (self.X - np.mean(self.X.flatten())) / np.std(self.X.flatten())


# def make_depth_from_gripper(obj_filename, param_filename, bottom=0.2):
#     """
#     Make depth images from perspective of gripper.
#     """
#     verts, faces = loadOBJ(obj_filename)
#     verts = np.array(verts)
#     min_bounding_box = np.min(verts, axis=0)
#     max_bounding_box = np.max(verts, axis=0)
#
#     # set bounding box horizontal centre to 0,0
#     verts[:,0] = verts[:,0] - (min_bounding_box[0]+max_bounding_box[0])/2.
#     verts[:,1] = verts[:,1] - (min_bounding_box[1]+max_bounding_box[1])/2.
#     # set bottom of bounding box to "bottom"
#     verts[:,2] = verts[:,2] + bottom - min_bounding_box[2]
#
#     d = Display(imsize=(80,80))
#
#     labels = []
#     depths = []
#     c = 0
#     for line in open(param_filename, "r"):
#         # print(c)
#         # if c == 100:
#         #     break
#         c = c + 1
#
#         vals = line.split(',')
#         gripper_pos = [float(vals[0]), float(vals[1]), float(vals[2])]
#         gripper_orient = [float(vals[3]), float(vals[4]), float(vals[5])]
#         rot = rot_matrix(gripper_orient[0], gripper_orient[1], gripper_orient[2])
#         labels.append(int(vals[6]))
#
#         d.set_camera_position(gripper_pos, rot, .4)
#         d.set_mesh(verts, faces) #this mut go after set_camera_position
#         depth = d.read_depth()
#         depths.append(depth)
#
#     d.close()
#     return np.array(depths), np.array(labels)


def load_all_params(param_filename):
    """
    Example line from file:
    "104_toaster_final-18-Dec-2015-13-56-59.obj",2.99894,0.034299705,0.4714164,0.09123467,0.0384472,0.5518384,0.0880979987086634,0.0
    """

    bad = [
        # Ashley says these are bad after looking through V-REP images ...
        '24_bowl-24-Feb-2016-17-38-53',
        '24_bowl-26-Feb-2016-08-35-29',
        '24_bowl-27-Feb-2016-23-52-43',
        '24_bowl-29-Feb-2016-15-01-53',
        '25_mug-11-Feb-2016-02-25-25',
        '28_Spatula_final-10-Mar-2016-18-31-08',
        '42_wineglass_final-01-Nov-2015-19-25-18',
        # These somehow have two objects in V-REP images ...
        '24_bowl-02-Mar-2016-07-03-29',
        '24_bowl-03-Mar-2016-22-54-50',
        '24_bowl-05-Mar-2016-13-53-41',
        '24_bowl-07-Mar-2016-05-06-04',
        # These ones may fall over a bit at simulation start (first has off depth maps, others don't) ...
        '55_hairdryer_final-18-Nov-2015-13-57-47',
        '55_hairdryer_final-15-Dec-2015-12-18-19',
        '55_hairdryer_final-09-Dec-2015-09-54-47',
        '55_hairdryer_final-19-Nov-2015-09-56-56',
        '55_hairdryer_final-21-Nov-2015-05-16-08',
        # These frequently do not have object at centre of depth map (various reasons possible) ...
        '33_pan_final-11-Mar-2016-17-41-49',
        '53_watertap_final-04-Dec-2015-01-28-24',
        '53_watertap_final-06-Dec-2015-04-20-45',
        '53_watertap_final-15-Nov-2015-05-08-46',
        '53_watertap_final-17-Nov-2015-00-26-39',
        '53_watertap_final-17-Nov-2015-15-57-57',
        '53_watertap_final-19-Jan-2016-04-32-52',
        '56_headphones_final-11-Nov-2015-14-14-02',
        '64_tongs_final-02-Dec-2015-12-22-36',
        '68_toy_final-05-Dec-2015-03-00-07',
        '68_toy_final-13-Nov-2015-10-50-34',
        '68_toy_final-18-Dec-2015-12-36-41',
        '68_toy_final-22-Nov-2015-08-51-12',
        '76_mirror_final-06-Dec-2015-03-46-18',
        '77_napkinholder_final-28-Nov-2015-13-06-17',
        '79_toy_dog_final-03-Dec-2015-08-15-04',
        '79_toy_dog_final-20-Jan-2016-06-55-00',
        '92_shell_final-26-Feb-2016-17-48-04',
        '94_weight_final-27-Feb-2016-15-40-40',
        '94_weight_final-29-Feb-2016-17-59-42',
        '95_boots_final-01-Mar-2016-16-02-15',
        '95_boots_final-01-Mar-2016-16-07-50',
        '95_boots_final-02-Mar-2016-13-46-24',
        '95_boots_final-02-Mar-2016-13-56-54',
        '95_boots_final-15-Nov-2015-06-30-07',
        '95_boots_final-20-Nov-2015-09-23-39',
        '95_boots_final-21-Nov-2015-04-00-35',
        '95_boots_final-23-Dec-2015-15-28-51',
        '95_boots_final-28-Feb-2016-18-58-13',
        '95_boots_final-28-Feb-2016-18-58-15',
        '98_faucet_final-28-Feb-2016-18-32-04',
        '98_faucet_final-28-Feb-2016-18-58-23',
        '98_faucet_final-28-Feb-2016-18-58-25'
    ]

    objects = []
    gripper_pos = []
    gripper_orient = []
    labels = []
    for line in open(param_filename, "r"):
        vals = line.translate(None, '"\n').split(',')
        if not (vals[0] == 'objfilename') and not vals[0][:-4] in bad:
            objects.append(vals[0])
            gripper_orient.append([float(vals[1]), float(vals[2]), float(vals[3])])
            gripper_pos.append([float(vals[4]), float(vals[5]), float(vals[6])])
            labels.append(int(float(vals[8])))
        # else:
        #     print('skipping ' + vals[0])

    return objects, gripper_pos, gripper_orient, labels


def make_depth_images(obj_name, pos, rot, obj_dir, image_dir, bottom=0.2, imsize=(80,80),
                      camera_offset=.45, near_clip=.25, far_clip=.8, support=False):
    """
    Saves depth images from perspective of gripper as image files. Default
    camera parameters make an exaggerated representation of region in front of hand.

    :param obj_name: Name corresponding to .obj file (without path or extension)
    :param pos: Positions of perspectives from which to make depth images
    :param rot: Rotation matrices of perspectives
    :param obj_dir: Directory where .obj files can be found
    :param image_dir: Directory in which to store images
    """
    obj_filename = obj_dir + obj_name + '.obj'

    if support:
        verts, faces = loadOBJ('../data/support-box.obj')
    else:
        verts, faces = loadOBJ(obj_filename)

    verts = np.array(verts)
    # minz = np.min(verts, axis=0)[2]
    # verts[:,2] = verts[:,2] + bottom - minz
    min_bounding_box = np.min(verts, axis=0)
    max_bounding_box = np.max(verts, axis=0)

    # set bounding box horizontal centre to 0,0
    verts[:,0] = verts[:,0] - (min_bounding_box[0]+max_bounding_box[0])/2.
    verts[:,1] = verts[:,1] - (min_bounding_box[1]+max_bounding_box[1])/2.
    # set bottom of bounding box to "bottom"
    verts[:,2] = verts[:,2] + bottom - min_bounding_box[2]

    d = Display(imsize=imsize)
    d.set_perspective(fov=45, near_clip=near_clip, far_clip=far_clip)

    for i in range(len(pos)):
        d.set_camera_position(pos[i], rot[i], camera_offset)
        d.set_mesh(verts, faces) #this must go after set_camera_position
        depth = d.read_depth()
        distance = get_distance(depth, near_clip, far_clip)
        rescaled_distance = np.maximum(0, (distance-camera_offset)/(far_clip-camera_offset))
        imfile = image_dir + obj_name + '-' + str(i) + '.png'
        Image.fromarray((255.0*rescaled_distance).astype('uint8')).save(imfile)
        # scipy.misc.toimage(depth, cmin=0.0, cmax=1.0).save(imfile)

    d.close()


def make_random_depths(obj_filename, param_filename, n, im_size=(40,40)):
    """
    Creates a dataset of depth maps and corresponding success probabilities
    at random interpolated gripper configurations.
    """
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

        estimate, confidence = get_prob_label(points, labels, point, sigma_p=2*.001, sigma_a=2*(4*np.pi/180))
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
    """
    Creates a random point that is interpolated between a pair of nearby points.
    """
    p1 = np.random.randint(0, len(points))
    p2 = get_closest_index(points, p1, prob_include=.5)
    mix_weight = np.random.rand()
    result = points[p1]*mix_weight + points[p2]*(1-mix_weight)
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
    """
    Gaussian kernel smoothing of success/failure to estimate success probability.
    """
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


# def make_random_bowl_depths():
#     shapes = ['24_bowl-02-Mar-2016-07-03-29',
#         '24_bowl-03-Mar-2016-22-54-50',
#         '24_bowl-05-Mar-2016-13-53-41',
#         '24_bowl-07-Mar-2016-05-06-04',
#         '24_bowl-16-Feb-2016-10-12-27',
#         '24_bowl-17-Feb-2016-22-00-34',
#         '24_bowl-24-Feb-2016-17-38-53',
#         '24_bowl-26-Feb-2016-08-35-29',
#         '24_bowl-27-Feb-2016-23-52-43',
#         '24_bowl-29-Feb-2016-15-01-53']
#
#     # shapes = ['24_bowl-02-Mar-2016-07-03-29']
#
#     n = 10000
#     for shape in shapes:
#         depths, labels = make_random_depths('../data/obj_files/' + shape + '.obj',
#                                             '../data/params/' + shape + '.csv',
#                                             n, im_size=(40,40))
#
#         f = file('../data/' + shape + '-random.pkl', 'wb')
#         cPickle.dump((depths, labels), f)
#         f.close()


# def check_random_bowl_depths():
#
#     # shapes = ['24_bowl-02-Mar-2016-07-03-29',
#     #     '24_bowl-03-Mar-2016-22-54-50',
#     #     '24_bowl-05-Mar-2016-13-53-41',
#     #     '24_bowl-07-Mar-2016-05-06-04',
#     #     '24_bowl-16-Feb-2016-10-12-27',
#     #     '24_bowl-17-Feb-2016-22-00-34',
#     #     '24_bowl-24-Feb-2016-17-38-53',
#     #     '24_bowl-26-Feb-2016-08-35-29',
#     #     '24_bowl-27-Feb-2016-23-52-43',
#     #     '24_bowl-29-Feb-2016-15-01-53']
#
#     shapes = ['24_bowl-02-Mar-2016-07-03-29']
#
#     f = file('../data/' + shapes[0] + '-random.pkl', 'rb')
#     (depths, labels) = cPickle.load(f)
#     f.close()
#
#     print(labels)
#
#     import matplotlib.pyplot as plt
#     from mpl_toolkits.mplot3d import axes3d, Axes3D
#
#     depths = depths.astype(float)
#     depths[depths > np.max(depths.flatten()) - 1] = np.NaN
#
#     X = np.arange(0, depths.shape[1])
#     Y = np.arange(0, depths.shape[2])
#     X, Y = np.meshgrid(X, Y)
#     fig = plt.figure(figsize=(12,6))
#     ax1 = fig.add_subplot(1, 2, 1, projection='3d')
#     plt.xlabel('x')
#     ax2 = fig.add_subplot(1, 2, 2, projection='3d')
#     plt.xlabel('x')
#     # ax = Axes3D(fig)
#     # for i in range(depths.shape[0]):
#     s = 5 #pixel stride
#     for i in range(n):
#         if labels[i] > .5:
#             color = 'g'
#             ax = ax1
#             ax.plot_wireframe(X[::s,::s], Y[::s,::s], depths[i,::s,::s], color=color)
#         else:
#             color = 'r'
#             ax = ax2
#             if np.random.rand(1) < .5:
#                 ax.plot_wireframe(X[::s,::s], Y[::s,::s], depths[i,::s,::s], color=color)
#         # plt.title(str(i) + ': ' + str(labels[i]))
#     plt.show()


def plot_box_corners():
    verts, faces = loadOBJ('../data/support-box.obj')
    verts = np.array(verts)
    print(verts.shape)

    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import axes3d, Axes3D
    fig = plt.figure()
    ax = fig.add_subplot(1, 2, 1, projection='3d')
    ax.scatter(verts[:,0], verts[:,1], verts[:,2])
    plt.show()


# def save_bowl_and_box_depths():
#     shapes = [
#         '24_bowl-16-Feb-2016-10-12-27',
#         '24_bowl-17-Feb-2016-22-00-34',
#         '24_bowl-24-Feb-2016-17-38-53',
#         '24_bowl-26-Feb-2016-08-35-29',
#         '24_bowl-27-Feb-2016-23-52-43',
#         '24_bowl-29-Feb-2016-15-01-53']
#
#     import time
#     for shape in shapes:
#         print('Processing ' + shape)
#         start_time = time.time()
#         depths, labels = make_depth_from_gripper('../data/obj_files/' + shape + '.obj',
#                                                 '../data/params/' + shape + '.csv',
#                                                 bottom=0.2)
#         box_depths, _ = make_depth_from_gripper('../data/support-box.obj',
#                                                 '../data/params/' + shape + '.csv',
#                                                 bottom=0)
#         f = file('../data/depths/' + shape + '.pkl', 'wb')
#         cPickle.dump((depths, box_depths, labels), f)
#         f.close()
#         print('    ' + str(time.time() - start_time) + 's')


def check_bowl_and_box_variance():
    f = file('../data/depths/' + '24_bowl-16-Feb-2016-10-12-27' + '.pkl', 'rb')
    depths, box_depths, labels = cPickle.load(f)
    f.close()

    # print(labels)
    # print(depths.shape)
    # print(box_depths.shape)

    obj_sd = []
    box_sd = []
    for i in range(len(labels)):
        obj_sd.append(np.std(depths[i,:,:].flatten()))
        box_sd.append(np.std(box_depths[i,:,:].flatten()))

    import matplotlib.pyplot as plt
    plt.plot(obj_sd)
    plt.plot(box_sd)
    plt.show()


def plot_bowl_and_box_distance_example():
    f = file('../data/depths/' + '24_bowl-16-Feb-2016-10-12-27' + '.pkl', 'rb')
    depths, box_depths, labels = cPickle.load(f)
    f.close()

    from depthmap import get_distance
    distances = get_distance(depths, .2, 1.0)
    box_distances = get_distance(box_depths, .2, 1.0)

    distances[distances > .99] = None
    box_distances[box_distances > .99] = None

    from mpl_toolkits.mplot3d import axes3d, Axes3D
    X = np.arange(0, depths.shape[1])
    Y = np.arange(0, depths.shape[2])
    X, Y = np.meshgrid(X, Y)
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    ax.plot_wireframe(X, Y, distances[205,:,:], color='b')
    ax.plot_wireframe(X, Y, box_distances[205,:,:], color='r')
    plt.show()


def load_depth_image(directory, object, index):
    imfile = directory + object + str(index) + '.png'
    image = scipy.ndimage.imread(imfile, flatten=True)


def get_pos_rot(objects, gripper_pos, gripper_orient, obj):
    """
    Get positions and rotation matrices for a given object, from a parameter list for
    all objects.
    """
    pos = []
    rot = []

    for i in range(len(objects)):
        if objects[i] == obj:
            pos.append(gripper_pos[i])
            rot.append(rot_matrix(gripper_orient[i][0], gripper_orient[i][1], gripper_orient[i][2]))

    return np.array(pos), np.array(rot)


def process_directory(obj_dir, image_dir, support=False):
    from os import listdir
    from os.path import isfile, join
    import time

    objects, gripper_pos, gripper_orient, labels = load_all_params('../../grasp-conv/data/output_data.csv')

    bottom = 0 if support else 0.2

    for f in listdir(obj_dir):
        obj_filename = join(obj_dir, f)
        if isfile(obj_filename) and f.endswith('.obj'):
            print('Processing ' + f)
            start_time = time.time()
            pos, rot = get_pos_rot(objects, gripper_pos, gripper_orient, f)
            make_depth_images(f[:-4], pos, rot, obj_dir, image_dir, bottom=bottom, support=support)
            print('   ' + str(time.time()-start_time) + 's')


def calculate_grasp_metrics_for_directory(image_dir, im_width=80,
                                          camera_offset=.45, near_clip=.25, far_clip=.8):
    from os import listdir
    from os.path import isfile, join
    import time
    from heuristic import finger_path_template
    from heuristic import calculate_grip_metrics

    template = finger_path_template(45.*np.pi/180., im_width, camera_offset)

    all_intersections = []
    all_qualities = []
    all_files = []
    for f in listdir(image_dir):
        image_filename = join(image_dir, f)
        if isfile(image_filename) and f.endswith('.png'):
            # print('Processing ' + image_filename)
            image = scipy.misc.imread(image_filename)
            rescaled_distance = image / 255.0
            distance = rescaled_distance*(far_clip-camera_offset)+camera_offset

            # from mpl_toolkits.mplot3d import axes3d, Axes3D
            # X = np.arange(0, im_width)
            # Y = np.arange(0, im_width)
            # X, Y = np.meshgrid(X, Y)
            # fig = plt.figure()
            # distance[distance > camera_offset + .3] = None
            # template[template < camera_offset] = None
            # ax = fig.add_subplot(1,1,1,projection='3d')
            # ax.plot_wireframe(X, Y, distance)
            # ax.plot_wireframe(X, Y, template, color='r')
            # ax.set_xlabel('x')
            # plt.show()

            intersections, qualities = calculate_grip_metrics(distance, template)
            # print(intersections)
            # print(qualities)
            all_intersections.append(intersections)
            all_qualities.append(qualities)
            all_files.append(f)
    return all_intersections, all_qualities, all_files


def calculate_grasp_metric_maps_for_directory(image_dir, dest_dir, im_width=80,
                                          camera_offset=.45, near_clip=.25, far_clip=.8):
    from os import listdir
    from os.path import isfile, join
    from heuristic import finger_path_template

    finger_path = finger_path_template(45.*np.pi/180., im_width, camera_offset)

    for f in listdir(image_dir):
        image_filename = join(image_dir, f)
        if isfile(image_filename) and f.endswith('.png'):
            print('Processing ' + image_filename)
            image = scipy.misc.imread(image_filename)
            rescaled_distance = image / 255.0
            distance = rescaled_distance*(far_clip-camera_offset)+camera_offset

            mm = calculate_metric_map(distance, finger_path, 1)
            imfile = dest_dir + f[:-4] + '-map' + '.png'
            Image.fromarray((255.0*mm).astype('uint8')).save(imfile)


def compress_images(directory, extension):
    """
    We need this to transfer data to server.
    """

    from os import listdir
    from os.path import isfile, join
    from zipfile import ZipFile

    n_per_zip = 50000

    # with ZipFile('zip-all.zip', 'w') as zf:
    #     for f in listdir(directory):
    #         image_filename = join(directory, f)
    #         if isfile(image_filename) and f.endswith(extension):
    #             zf.write(image_filename)

    zip_index = 0
    file_index = 0
    for f in listdir(directory):
        image_filename = join(directory, f)
        if isfile(image_filename) and f.endswith(extension):
            if file_index == 0:
                zf = ZipFile('obj' + str(zip_index) + '.zip', 'w')
            zf.write(image_filename)
            file_index += 1
            if file_index == n_per_zip:
                print('writing file ' + str(zip_index))
                file_index = 0
                zf.close()
                zip_index += 1

    zf.close()


if __name__ == '__main__':
    # save_bowl_and_box_depths()
    # plot_bowl_and_box_distance_example()

    # compress_images('../../grasp-conv/data/support_depths/', '.png')
    compress_images('../../grasp-conv/data/obj_depths/', '.png')
    # compress_images('../../grasp-conv/data/obj_mm/', '.png')

    # calculate_grasp_metric_maps_for_directory('../../grasp-conv/data/obj_depths/', '../../grasp-conv/data/obj_mm/')
    # image = scipy.misc.imread('../../grasp-conv/data/obj_mm/104_toaster_final-18-Dec-2015-13-56-59-0-map.png')
    # mm = image / 255.0
    # print(np.min(mm))
    # print(np.max(mm))

    # objects, gripper_pos, gripper_orient, labels = load_all_params('../../grasp-conv/data/output_data.csv')

    # obj_dir = '../../grasp-conv/data/obj_files/'
    # process_directory(obj_dir, '../../grasp-conv/data/obj_depths/')
    # # process_directory(obj_dir, '../../grasp-conv/data/support_depths/', support=True)
    #
    # # intersections, qualities, files = calculate_grasp_metrics_for_directory('../../grasp-conv/data/support_depths/')
    # intersections, qualities, files = calculate_grasp_metrics_for_directory('../../grasp-conv/data/obj_depths/')
    # f = file('../data/metrics-objects.pkl', 'wb')
    # cPickle.dump((intersections, qualities, files), f)
    # f.close()

    # f = file('metrics.pkl', 'rb')
    # intersections, qualities = cPickle.load(f)
    # f.close()
    # print(intersections)
    # print(qualities)

    # from mpl_toolkits.mplot3d import axes3d, Axes3D
    # X = np.arange(0, 80)
    # Y = np.arange(0, 80)
    # X, Y = np.meshgrid(X, Y)
    # fig = plt.figure()
    # ax = fig.add_subplot(1,1,1,projection='3d')
    # ax.plot_wireframe(X, Y, foo)
    # ax.set_xlabel('x')
    # plt.show()
