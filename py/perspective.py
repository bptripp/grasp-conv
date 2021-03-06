__author__ = 'bptripp'

from os import listdir
from os.path import isfile, join
import time
import numpy as np
import matplotlib.pyplot as plt
import cPickle
from PIL import Image
from scipy.optimize import bisect
from quaternion import angle_between_quaterions, to_quaternion


def get_random_points(n, radius, surface=False):
    point_directions = np.random.randn(3, n)
    norms = np.sum(point_directions**2, axis=0)**.5
    points = radius * point_directions / norms

    if not surface:
        # points = points * np.random.rand(n)**(1./3.)
        palm = .035 #TODO: whoops, this is in metres, not fraction of radius
        points = points * (palm + (1-palm)*np.random.rand(n))

    return points


def get_random_angles(n, std=np.pi/8.):
    """
    :param n: Number of angles needed
    :return: Random angles in restricted ranges, meant as deviations in perspective around
        looking staight at something.
    """
    angles = std*np.random.randn(3, n)
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

    # if get_rotation_matrix(point, np.array([0, 0, gamma]))[2][1] < 0:
    if get_rotation_matrix(point, np.array([0, 0, gamma]))[2][1] > 0:
        gamma = gamma + np.pi

    return gamma


def check_find_vertical():
    n = 3
    points = get_random_points(n, .35, surface=True)
    for i in range(n):
        point = points[:,i]
        gamma = find_vertical(point)
        rot = get_rotation_matrix(point, np.array([0, 0, gamma]))
        print(rot)
        # if np.abs(rot[2,0] > 1e-6) or rot[2,1] < 0:
        if np.abs(rot[2,0] > 1e-6) or rot[2,1] > 0:
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


def get_perspectives(obj_filename, points, angles, im_width=80, near_clip=.25, far_clip=0.8, fov=45, camera_offset=.45, target_point=None):
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

    if target_point is not None:
        verts[:,0] = verts[:,0] - target_point[0]
        verts[:,1] = verts[:,1] - target_point[1]
        verts[:,2] = verts[:,2] - target_point[2]

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
                points = get_random_points(n, .15)
                angles = get_random_angles(n, std=0)
                print(angles)

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


def calculate_metrics(perspectives, im_width=80, fov=45.0, camera_offset=.45):
    """
    :param perspectives: numpy array of depth images of object from gripper perspective
    """
    asymmetry_scale = 13.0 #TODO: calculate from camera params (13 pixels is ~5cm with default params)

    from heuristic import finger_path_template, calculate_grip_metrics
    finger_path = finger_path_template(fov*np.pi/180., im_width, camera_offset)

    collision_template = np.zeros_like(finger_path)
    collision_template[finger_path > 0] = camera_offset + 0.033

    # print(np.max(collision_template))
    # print(np.max(finger_path))
    # plt.imshow(collision_template)
    # plt.show()

    metrics = []
    collisions = []
    for perspective in perspectives:
        intersections, qualities = calculate_grip_metrics(perspective, finger_path)
        q1 = qualities[0]
        q2 = qualities[1]
        q3 = qualities[2]

        if intersections[0] is None or intersections[2] is None:
            a1 = 1
        else:
            a1 = ((intersections[0]-intersections[2])/asymmetry_scale)**2

        if intersections[1] is None or intersections[2] is None:
            a2 = 1
        else:
            a2 = ((intersections[1]-intersections[2])/asymmetry_scale)**2

        m = np.minimum((q1+q2)/1.5, q3) / (1 + (q1*a1+q2*a2) / (q1+q2+1e-6))

        collision = np.max(collision_template - perspective) > 0
        collisions.append(collision)
        # if collision:
        #     m = 0

        metrics.append(m)

        # plt.subplot(1,2,1)
        # plt.imshow(perspective)
        # plt.subplot(1,2,2)
        # plt.imshow(np.maximum(0, finger_path-perspective))
        # print(collision)
        # print((a1,a2))
        # print(intersections)
        # print(qualities)
        # print('metric: ' + str(m))
        # plt.show()

        # print((intersections, qualities))
    return metrics, collisions


def get_quaternion_distance(points, angles):
    """
    Get new representation of camera/gripper configurations as rotation quaternions and
    distances from origin, rather than 3D points and rotations about axis pointing to origin.
    """
    # print(points)
    # print(angles)
    quaternions = []
    distances = []
    for point, angle in zip(points.T, angles.T):
        distances.append(np.linalg.norm(point))
        quaternions.append(to_quaternion(get_rotation_matrix(point, angle)))

    return np.array(quaternions), np.array(distances)


def smooth_metrics(quaternions, distances, metrics):
    from interpolate import interpolate

    smoothed = []
    for i in range(len(metrics)):
        # print(i)
        interpolated = interpolate(quaternions[i], distances[i], quaternions, distances, metrics,
                                   sigma_d=.02, sigma_a=(16*np.pi/180))
        smoothed.append(interpolated)
        # print(interpolated - metrics[one])

    return smoothed


def load_target_points(filename):
    objects = []
    indices = []
    points = []
    for line in open(filename, "r"):
        vals = line.translate(None, '"\n').split(',')
        assert len(vals) == 5
        objects.append(vals[0])
        indices.append(int(vals[1]))
        points.append([float(vals[2]), float(vals[3]), float(vals[4])])

    return objects, indices, points


def get_target_points_for_object(objects, indices, points, object):
    indices_for_object = []
    points_for_object = []
    for o, i, p in zip(objects, indices, points):
        if o == object:
            indices_for_object.append(i)
            points_for_object.append(p)
    return np.array(indices_for_object), np.array(points_for_object)


def check_target_points():
    objects, indices, points = load_target_points('../../grasp-conv/data/obj-points.csv')
    print(objects)
    print(indices)
    print(points)
    indices, points = get_target_points_for_object(objects, indices, points, '28_Spatula_final-11-Nov-2015-14-22-01.obj')
    print(indices)
    print(points)


def check_metrics():
    # points, angles, metrics, collisions = calculate_metrics('../../grasp-conv/data/perspectives/28_Spatula_final-11-Nov-2015-14-22-01.pkl')
    # with open('spatula-perspectives.pkl', 'wb') as f:
    #     cPickle.dump((points, angles, metrics, collisions), f)

    with open('spatula-perspectives.pkl', 'rb') as f:
        (points, angles, metrics, collisions) = cPickle.load(f)
    metrics = np.array(metrics)

    smoothed = smooth_metrics(points, angles, metrics)
    with open('spatula-perspectives-smoothed.pkl', 'wb') as f:
        cPickle.dump((points, angles, metrics, collisions, smoothed), f)

    plt.hist(metrics, bins=50)
    plt.show()


def make_grip_perspective_depths(obj_dir, data_dir, target_points_file, n=1000):
    objects, indices, points = load_target_points(target_points_file)

    for f in listdir(obj_dir):
        obj_filename = join(obj_dir, f)
        if isfile(obj_filename) and f.endswith('.obj'):
            data_filename = join(data_dir, f[:-4] + '.pkl')
            if isfile(data_filename):
                print('Skipping ' + f)
            else:
                print('Processing ' + f)
                target_indices, target_points = get_target_points_for_object(objects, indices, points, f)

                start_time = time.time()
                #TODO: is there any reason to make points & angles these the same or different across targets?
                gripper_points = get_random_points(n, .15)
                gripper_angles = get_random_angles(n, std=0)

                perspectives = []
                for target_point in target_points:
                    print('   ' + str(target_point))

                    p = get_perspectives(obj_filename, gripper_points, gripper_angles, target_point=target_point)
                    perspectives.append(p)

                f = open(data_filename, 'wb')
                cPickle.dump((gripper_points, gripper_angles, target_indices, target_points, perspectives), f)
                f.close()
                print('   ' + str(time.time()-start_time) + 's')


def make_metrics(perspective_dir, metric_dir):
    """
    We'll store in separate pkl files per object to allow incremental processing, even through results
    won't take much memory.
    """

    for f in listdir(perspective_dir):
        perspective_filename = join(perspective_dir, f)
        if isfile(perspective_filename) and f.endswith('.pkl'):
            metric_filename = join(metric_dir, f[:-4] + '-metrics.pkl')
            if isfile(metric_filename):
                print('Skipping ' + f)
            else:
                print('Processing ' + f)

                start_time = time.time()
                with open(perspective_filename) as perspective_file:
                    gripper_points, gripper_angles, target_indices, target_points, perspectives = cPickle.load(perspective_file)
                print('unpickle: ' + str(time.time() - start_time))

                quaternions, distances = get_quaternion_distance(gripper_points, gripper_angles)

                collisions = []
                free_smoothed = []
                coll_smoothed = []

                for p in perspectives: # one per target point
                    fm, c = calculate_metrics(p)
                    fm = np.array(fm)
                    c = np.array(c)
                    fs = smooth_metrics(quaternions, distances, fm)
                    cm = fm * (1-c)
                    cs = smooth_metrics(quaternions, distances, cm)

                    collisions.append(c)
                    free_smoothed.append(fs)
                    coll_smoothed.append(cs)

                f = open(metric_filename, 'wb')
                cPickle.dump((gripper_points, gripper_angles, target_indices, target_points,
                              collisions, free_smoothed, coll_smoothed), f)
                f.close()


def make_eye_perspective_depths(obj_dir, data_dir, target_points_file, n=20):
    all_objects, all_target_indices, all_target_points = load_target_points(target_points_file)

    camera_offset=.45
    near_clip=.6
    far_clip=1.0

    eye_points = []
    eye_angles = []
    objects = []
    target_indices = []
    target_points = []
    for f in listdir(obj_dir):
        obj_filename = join(obj_dir, f)
        if isfile(obj_filename) and f.endswith('.obj'): #and int(f[0]) >= 0: #TODO: set f[0] range here
            print('Processing ' + f)
            ti, tp= get_target_points_for_object(all_objects, all_target_indices, all_target_points, f)
            objects.append(f)
            target_indices.append(ti)
            target_points.append(tp)

            start_time = time.time()

            points = get_random_points(n, .35, surface=True) #.8m with offset
            # points = np.array([[ 0.00001], [0.35], [0.00001]]) # TODO: bug with x=0 and with z=0

            print(points)

            angles = np.zeros_like(points)
            eye_points.append(points)
            eye_angles.append(angles)

            # Set camera-up to vertical via third angle (angle needed is always
            # 3pi/4, but we'll find it numerically in case other parts of code
            # change while we're not looking).
            for i in range(n):
                angles[2,i] = find_vertical(points[:,i])

            perspectives = []
            for target_index, target_point in zip(ti, tp):
                print('   ' + str(target_point))
                perspectives = get_perspectives(obj_filename, points, angles,
                                                near_clip=near_clip, far_clip=far_clip, camera_offset=camera_offset,
                                                fov=30, target_point=target_point)

                for i in range(len(perspectives)):
                    distance = perspectives[i]
                    rescaled_distance = np.maximum(0, (distance-camera_offset)/(far_clip-camera_offset))
                    imfile = data_dir + f[:-4] + '-' + str(target_index) + '-' + str(i) + '.png'
                    Image.fromarray((255.0*rescaled_distance).astype('uint8')).save(imfile)

            print('   ' + str(time.time()-start_time) + 's')

    data_filename = join(data_dir, 'eye-perspectives-murata.pkl')
    f = open(data_filename, 'wb')
    cPickle.dump((objects, target_indices, target_points, eye_points, eye_angles), f)
    f.close()


def merge_eye_perspectives(data_dir):
    # make_eye_perspective_depths mysteriously does not run all the way through, so I've
    # done it in two parts which are merged here:

    files = ['eye-perspectives1.pkl', 'eye-perspectives2.pkl']
    objects = []
    target_indices = []
    target_points = []
    eye_points = []
    eye_angles = []
    for file in files:
        with open(join(data_dir, file), 'rb') as f:
            o, ti, tp, ep, ea = cPickle.load(f)
            objects.extend(o)
            target_indices.extend(ti)
            target_points.extend(tp)
            eye_points.extend(ep)
            eye_angles.extend(ea)

    with open(join(data_dir, 'eye-perspectives.pkl'),'wb') as f:
        cPickle.dump((objects, target_indices, target_points, eye_points, eye_angles), f)


def export_neuron_perspectives():
    import csv

    with open('../data/neuron-points.pkl', 'rb') as f:
        neuron_points, neuron_angles = cPickle.load(f)

    with open('neuron-perspectives.csv', 'wb') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')

        for point, angle in zip(neuron_points.T, neuron_angles.T):
            R = get_rotation_matrix(point, angle)
            row = list(point)
            row.extend(R.flatten())
            writer.writerow(row)


def export_eye_perspectives(eye_perspectives_file):
    import csv

    with open(eye_perspectives_file) as f:
        objects, target_indices, target_points, eye_points, eye_angles = cPickle.load(f)

    with open('eye-perspectives.csv', 'wb') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')

        for object, ep, ea, tp in zip(objects, eye_points, eye_angles, target_points):
            print('Processing ' + object)
            for target_point in tp:
                for eye_point, eye_angle in zip(ep.T, ea.T):
                    eye_R = get_rotation_matrix(eye_point, eye_angle)
                    # row = [object]
                    row = []
                    row.extend(target_point)
                    row.extend(eye_point)
                    row.extend(eye_R.flatten())
                    writer.writerow(row)


def make_relative_metrics(eye_perspectives_file, metrics_dir, result_dir, n=500, neuron_points=None, neuron_angles=None):
    from quaternion import difference_between_quaternions
    from interpolate import interpolate

    # each of these points/angles will correspond to an output neuron ...
    if neuron_points is None or neuron_angles is None:
        neuron_points = get_random_points(n, .15)
        neuron_angles = get_random_angles(n, std=0)
        with open(join(result_dir, 'neuron-points.pkl'), 'wb') as f:
            cPickle.dump((neuron_points, neuron_angles), f)

    neuron_quaternions, neuron_distances = get_quaternion_distance(neuron_points, neuron_angles)

    with open(eye_perspectives_file) as f:
        objects, target_indices, target_points, eye_points, eye_angles = cPickle.load(f)

    for object, object_eye_points, object_eye_angles in zip(objects, eye_points, eye_angles):
        print('Processing ' + object)
        start_time = time.time()
        eye_quaternions, eye_distances = get_quaternion_distance(object_eye_points, object_eye_angles)

        metrics_file = join(metrics_dir, object[:-4] + '-metrics.pkl')
        with open(metrics_file) as f:
            gripper_points, gripper_angles, target_indices, target_points, collisions, free_smoothed, coll_smoothed = cPickle.load(f)
        gripper_quaternions, gripper_distances = get_quaternion_distance(gripper_points, gripper_angles)

        # note that for each object, gripper configs are the same relative to each target point
        #TODO: do we want coll_smoothed instead / as well?
        metrics = free_smoothed

        # interpolate relative to each eye point
        neuron_metrics_for_object = []
        for target_index, target_metrics in zip(target_indices, metrics):
            print('    target ' + str(target_index))
            neuron_metrics_for_target = []
            for eye_quaternion in eye_quaternions:
                rel_quaternions = []
                for gripper_quaternion in gripper_quaternions:
                    rel_quaternions.append(difference_between_quaternions(eye_quaternion, gripper_quaternion))
                rel_quaternions = np.array(rel_quaternions)

                #interpolate ...
                neuron_metrics = []
                for neuron_quaternion, neuron_distance in zip(neuron_quaternions, neuron_distances):
                    interpolated = interpolate(neuron_quaternion, neuron_distance, rel_quaternions, gripper_distances, target_metrics,
                                       sigma_d=.02, sigma_a=(16*np.pi/180))
                    neuron_metrics.append(interpolated)

                neuron_metrics_for_target.append(neuron_metrics)
            neuron_metrics_for_object.append(neuron_metrics_for_target)
        neuron_metrics_for_object = np.array(neuron_metrics_for_object)

        result_file = join(result_dir, object[:-4] + '-neuron.pkl')
        with open(result_file, 'wb') as f:
            cPickle.dump((target_indices, target_points, object_eye_points, object_eye_angles, neuron_metrics_for_object), f)
        print('    ' + str(time.time() - start_time) + 's')


def make_XY():
    eye_image_files = []
    metrics = []
    return eye_image_files, metrics


if __name__ == '__main__':
    # check_rotation_matrix(scatter=True)
    # check_depth_from_random_perspective()
    # plot_random_samples()
    # check_find_vertical()
    # check_target_points()
    # check_metrics()

    # make_grip_perspective_depths('../../grasp-conv/data/obj_tmp/',
    #                              '../../grasp-conv/data/perspectives/',
    #                              '../../grasp-conv/data/obj-points.csv')

    # make_grip_perspective_depths('../../grasp-conv/data/obj_files/',
    #                              '/Volumes/TrainingData/grasp-conv/data/perspectives/',
    #                              '../../grasp-conv/data/obj-points.csv')

    # make_eye_perspective_depths('../../grasp-conv/data/obj_tmp/',
    #                             '../../grasp-conv/data/eye-tmp/',
    #                             '../../grasp-conv/data/obj-points.csv')

    make_eye_perspective_depths('../../grasp-conv/data/obj_files_murata/',
                                '../../grasp-conv/data/eye-perspectives-murata/',
                                '../../grasp-conv/data/obj-points-murata.csv',
                                n=1)

    # make_eye_perspective_depths('../../grasp-conv/data/obj_files/',
    #                             '/Volumes/TrainingData/grasp-conv/data/eye-perspectives/',
    #                             '../../grasp-conv/data/obj-points.csv')

    # merge_eye_perspectives('/Volumes/TrainingData/grasp-conv/data/eye-perspectives/')
    # with open('/Volumes/TrainingData/grasp-conv/data/eye-perspectives/eye-perspectives.pkl','rb') as f:
    #     objects, target_indices, target_points, eye_points, eye_angles = cPickle.load(f)
    # print(objects)
    # print(target_indices)
    # print(target_points)
    # print(eye_angles)

    # make_relative_metrics('/Volumes/TrainingData/grasp-conv/data/eye-perspectives/eye-perspectives.pkl',
    #                       '/Volumes/TrainingData/grasp-conv/data/metrics/',
    #                       '/Volumes/TrainingData/grasp-conv/data/relative/')

    # export_neuron_perspectives()
    # export_eye_perspectives('/Volumes/TrainingData/grasp-conv/data/eye-perspectives/eye-perspectives.pkl')

    # import scipy
    # image = scipy.misc.imread('../../grasp-conv/data/eye-tmp/1_Coffeecup_final-03-Mar-2016-18-50-40-0-7.png')
    # plt.imshow(image)
    # plt.show()

    # with open('../../grasp-conv/data/eye-tmp/eye-perspectives.pkl') as f:
    #     objects, target_indices, target_points, eye_points, eye_angles = cPickle.load(f)
    # print(objects)
    # print(target_indices)
    # print(target_points)
    # print(np.array(eye_points))
    # print(np.array(eye_angles))


    # with open('spatula-perspectives.pkl', 'rb') as f:
    #     gripper_points, gripper_angles, target_indices, target_points, perspectives = cPickle.load(f)

    # make_metrics('../../grasp-conv/data/perspectives/', '../../grasp-conv/data/metrics/')

    # make_relative_metrics('../../grasp-conv/data/eye-tmp/eye-perspectives.pkl',
    #                       '../../grasp-conv/data/metrics/',
    #                       '../../grasp-conv/data/relative/')

    # checking files look OK ...
    # with open('../../grasp-conv/data/relative/neuron-points.pkl', 'rb') as f:
    #     neuron_points, neuron_angles = cPickle.load(f)
    # print(neuron_angles.shape)
    # with open('../../grasp-conv/data/relative/1_Coffeecup_final-03-Mar-2016-18-50-40-neuron.pkl', 'rb') as f:
    #     target_indices, target_points, object_eye_points, object_eye_angles, neuron_metrics_for_object = cPickle.load(f)
    # print(neuron_metrics_for_object.shape)
    # print(np.min(neuron_metrics_for_object))
    # print(np.max(neuron_metrics_for_object))
    # print(np.std(neuron_metrics_for_object))

    # make_metrics('/Volumes/TrainingData/grasp-conv/data/perspectives/',
    #              '/Volumes/TrainingData/grasp-conv/data/metrics/')

    # process_eye_directory('../../grasp-conv/data/obj_tmp/', '../../grasp-conv/data/eye-perspectives-tmp/', 100)

    # process_directory('../data/obj_files/', '../data/perspectives/', 10)
    # process_directory('../../grasp-conv/data/obj_tmp/', '../../grasp-conv/data/perspectives/', 5000)
    # check_maps('../../grasp-conv/data/perspectives/')

