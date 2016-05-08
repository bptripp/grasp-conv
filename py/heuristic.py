__author__ = 'bptripp'

import numpy as np
from scipy.optimize import newton
from scipy.signal import convolve2d
import matplotlib.pyplot as plt

# Barrett hand dimensions from http://www.barrett.com/images/HandDime4.gif
# Fingers don't extend fully, max 40deg from straight. First segment .07m; second .058m
# I have estimated the hand origin in MeshLab from a mesh exported from V-REP


def finger_path_template(fov, im_width, camera_offset, finger_width=.025):
    """
    :param fov: camera field of view (radians!!!)
    :param im_width: pixels
    :param camera_offset: distance of camera behind hand
    :return: distance image of the intersection volume of Barrett hand
    """
    # pixels = range(im_width/2)
    pixels = range(-im_width/4, im_width/2) # cross centre
    rads_per_pixel = fov / im_width;
    angles = rads_per_pixel * (np.array(pixels).astype(float) + 0.5)

    single_finger_xyz = (0.,.0396,.0302)
    double_finger_xyz = (0.025,.0604,.0302)

    single_depths = [] #lone finger
    double_depths = [] #pair of fingers on other side
    for angle in angles:
        single_depths.append(finger_depth(angle, camera_offset, finger_yz=single_finger_xyz[1:]))
        double_depths.append(finger_depth(angle, camera_offset, finger_yz=double_finger_xyz[1:]))

    template = np.zeros((im_width,im_width))

    for i in range(len(pixels)):
        if single_depths[i] > 0:
            finger_half_width_rad = np.arctan(finger_width/2./single_depths[i])
            finger_half_width_pixels = finger_half_width_rad / rads_per_pixel

            min_finger = int(np.floor(im_width/2-finger_half_width_pixels+.5))
            max_finger = int(np.ceil(im_width/2+finger_half_width_pixels+.5))
            template[im_width/2-1-pixels[i],min_finger:max_finger] = single_depths[i] #TODO: clean up offset

        if double_depths[i] > 0:
            finger_x_pixels = np.arctan(double_finger_xyz[0]/double_depths[i]) / rads_per_pixel # x offset of paired fingers

            min_finger = int(np.floor(im_width/2+finger_x_pixels-finger_half_width_pixels+.5))
            max_finger = int(np.ceil(im_width/2+finger_x_pixels+finger_half_width_pixels+.5))
            template[im_width/2+pixels[i],min_finger:max_finger] = double_depths[i]

            min_finger = int(np.floor(im_width/2-finger_x_pixels-finger_half_width_pixels+.5))
            max_finger = int(np.ceil(im_width/2-finger_x_pixels+finger_half_width_pixels+.5))
            template[im_width/2+pixels[i],min_finger:max_finger] = double_depths[i]

    return template


def finger_depth(camera_angle, camera_offset, finger_length=.12, finger_yz=(.05,0.013), finger_ext=.31):
    # finger_ext: angle from finger CoR to tip at max extension
    y0 = finger_yz[0]
    z0 = finger_yz[1]
    l = finger_length

    max_angle = np.arctan((y0+l*np.cos(finger_ext))/(z0+camera_offset+l*np.sin(finger_ext)))
    # this seems like a reasonable place to stop
    min_angle = np.arctan((y0+l*np.cos(0.75*np.pi))/(z0+camera_offset+l*np.sin(0.75*np.pi)))

    result = 0
    # if camera_angle > -1e-6 and camera_angle < max_angle:
    if camera_angle > min_angle and camera_angle < max_angle:
        # find corresponding finger angle
        f = lambda b: (y0+l*np.sin(b))/(z0+camera_offset+l*np.cos(b)) - np.tan(camera_angle)
        b = newton(f, np.pi/4.)
        z = z0 + camera_offset + l*np.cos(b)
        y = y0 + l*np.sin(b)
        result = np.sqrt(y**2+z**2)
    return result


def calculate_metric_map(depth_map, finger_path, direction, saturation_distance=.02, box_size=3):
    """
    An intermediate step in calculation of grip metrics. It isn't necessary to calculate
    this result separately except that decomposing calculation of metrics in this way
    may simplify deep network training.
    The same metrics could be calculated from various related maps. This one is chosen because
    it has fairly local features that should be easy for a network to approximate.
    """

    finger_width = np.round(np.mean(np.sum(finger_path > 0, axis=0)))

    overlap = np.maximum(0, finger_path - depth_map)
    overlap = np.minimum(saturation_distance, overlap)
    overlap = overlap[::direction,:]

    running_max = np.zeros_like(overlap)
    for i in range(overlap.shape[0]):
        start = np.maximum(0,i)
        finish = np.minimum(overlap.shape[0],i+1)
        running_max[i,:] = np.max(overlap[start:finish,:], axis=0)

    window = np.ones((box_size,finger_width))
    window = window / np.sum(window*saturation_distance) # normalize so that max convolution result is 1
    result = convolve2d(running_max, window, mode='same')

    return result[::direction,:]


def calculate_grip_metrics(depth_map, finger_path, saturation_distance=.02, box_size=3):
    overlap = np.maximum(0, finger_path - depth_map)

    # find first overlap from outside to centre in three regions
    s = depth_map.shape
    regions = [[s[0],s[0]/2,0,s[1]/2],
               [s[0],s[0]/2,s[1]/2,s[1]],
               [0,s[0]/2,s[1]/4,3*s[1]/4]]
    close_directions = [-1,-1,1]

    intersections = []
    qualities = []
    for region, direction in zip(regions, close_directions):
        region_overlap = overlap[region[0]:region[1]:direction,region[2]:region[3]]

        #running max to avoid penalizing grasping outside of concave shape ...
        # region_overlap = np.maximum.accumulate(region_overlap, axis=1)
        region_overlap = np.maximum.accumulate(region_overlap, axis=0)

        # p = np.sum(region_overlap, axis=0)
        p = np.sum(region_overlap, axis=1)
        # print(p)
        if True in (p>0).tolist():
            intersection = (p>0).tolist().index(True)
        else:
            intersection = None
        intersections.append(intersection)

        region_finger = finger_path[region[0]:region[1]:direction,region[2]:region[3]]
        region_finger = saturation_distance * np.array(region_finger > 0).astype(float)

        region_overlap = np.minimum(region_overlap, saturation_distance)
        if intersection is None:
            quality = 0
        else:
            # sub_region_overlap = region_overlap[:,intersection:intersection+box_size]
            # sub_region_finger = region_finger[:,intersection:intersection+box_size]
            sub_region_overlap = region_overlap[intersection:intersection+box_size,:]
            sub_region_finger = region_finger[intersection:intersection+box_size,:]
            quality = np.sum(sub_region_overlap.flatten()) / np.sum(sub_region_finger.flatten())

        # plt.imshow(sub_region_finger)
        # plt.show()

        qualities.append(quality)

    # print(intersections)
    # print(qualities)
    #
    # from mpl_toolkits.mplot3d import axes3d, Axes3D
    # X = np.arange(0, len(depth_map))
    # Y = np.arange(0, len(depth_map))
    # X, Y = np.meshgrid(X, Y)
    # fig = plt.figure()
    # ax = fig.add_subplot(1,1,1,projection='3d')
    # ax.plot_wireframe(X, Y, overlap)
    # plt.show()

    return intersections, qualities


def check_overlap_range(image_dir='../../grasp-conv/data/obj_depths', im_width=80, camera_offset=.45, far_clip=.8):
    from data import load_all_params
    import scipy
    objects, gripper_pos, gripper_orient, labels = load_all_params('../../grasp-conv/data/output_data.csv')
    seq_nums = np.arange(len(objects)) % 1000

    from os import listdir
    from os.path import isfile, join
    from heuristic import finger_path_template

    finger_path = finger_path_template(45.*np.pi/180., im_width, camera_offset)

    max_overlaps = []
    for object, seq_num in zip(objects, seq_nums):
        filename = join(image_dir, object[:-4] + '-' + str(seq_num) + '.png')
        image = scipy.misc.imread(filename)
        rescaled_distance = image / 255.0
        distance = rescaled_distance*(far_clip-camera_offset)+camera_offset
        max_overlaps.append(np.max(finger_path - distance))

    return max_overlaps, labels


if __name__ == '__main__':
    camera_offset=.45
    near_clip=.25
    far_clip=.8

    import scipy
    image = scipy.misc.imread('../../grasp-conv/data/obj_depths/1_Coffeecup_final-03-Mar-2016-18-50-40-1.png')
    rescaled_distance = image / 255.0
    distance = rescaled_distance*(far_clip-camera_offset)+camera_offset

    finger_path = finger_path_template(45.*np.pi/180., 80, camera_offset)
    import time
    start_time = time.time()
    for i in range(100):
        mm = calculate_metric_map(distance, finger_path, 1)
    print('elapsed: ' + str(time.time() - start_time))

    print(np.min(mm))
    print(np.max(mm))

    intersections, qualities = calculate_grip_metrics(distance, finger_path)

    print(intersections)

    # plt.imshow(mm)
    # plt.imshow(finger_path)
    # plt.show()

    from visualize import plot_mesh
    plot_mesh(finger_path)

    # angles = np.arange(0, np.pi/6, np.pi/160)
    # depths = []
    # for angle in angles:
    #     depths.append(finger_depth(angle, .3))
    #
    # import matplotlib.pyplot as plt
    # plt.plot(angles, depths)
    # plt.show()

    # x = .07 + .058*np.cos(40.*np.pi/180.)
    # y = .058*np.sin(40.*np.pi/180.)
    # print(np.sqrt(x**2+y**2))
    # print(np.arctan(.037/.114))