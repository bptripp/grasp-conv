__author__ = 'bptripp'

import numpy as np
from scipy.optimize import newton
import matplotlib.pyplot as plt

# Barrett hand dimensions from http://www.barrett.com/images/HandDime4.gif
# Fingers don't extend fully, max 40deg from straight. First segment .07m; second .058m
# I have estimated the hand origin in MeshLab from a mesh exported from V-REP


def finger_path_template(fov, im_width, camera_offset, finger_width=.025, finger_xyz=(.025,.05,.013)):
    """
    :param fov: camera field of view (radians!!!)
    :param im_width: pixels
    :param camera_offset: distance of camera behind hand
    :return: distance image of the intersection volume of Barrett hand
    """
    pixels = range(im_width/2)
    rads_per_pixel = fov / im_width;
    angles = rads_per_pixel * (np.array(pixels).astype(float) + 0.5)

    depths = []
    for angle in angles:
        depths.append(finger_depth(angle, camera_offset, finger_yz=finger_xyz[1:]))

    template = np.zeros((im_width,im_width))

    for pixel in pixels:
        if depths[pixel] > 0:
            finger_half_width_rad = np.arctan(finger_width/2./depths[pixel])
            finger_half_width_pixels = finger_half_width_rad / rads_per_pixel

            min_finger = int(np.floor(im_width/2-finger_half_width_pixels+.5))
            max_finger = int(np.ceil(im_width/2+finger_half_width_pixels+.5))
            template[im_width/2-1-pixel,min_finger:max_finger] = depths[pixel] #TODO: clean up offset

            finger_x_pixels = np.arctan(finger_xyz[0]/depths[pixel]) /rads_per_pixel # x offset of paired fingers

            min_finger = int(np.floor(im_width/2+finger_x_pixels-finger_half_width_pixels+.5))
            max_finger = int(np.ceil(im_width/2+finger_x_pixels+finger_half_width_pixels+.5))
            template[im_width/2+pixel,min_finger:max_finger] = depths[pixel]

            min_finger = int(np.floor(im_width/2-finger_x_pixels-finger_half_width_pixels+.5))
            max_finger = int(np.ceil(im_width/2-finger_x_pixels+finger_half_width_pixels+.5))
            template[im_width/2+pixel,min_finger:max_finger] = depths[pixel]

    # plt.imshow(template)
    # plt.show()
    return template


def finger_depth(camera_angle, camera_offset, finger_length=.12, finger_yz=(.05,0.013), finger_ext=.31):
    # finger_ext: angle from finger CoR to tip at max extension
    y0 = finger_yz[0]
    z0 = finger_yz[1]
    l = finger_length

    max_angle = np.arctan((y0+l*np.cos(finger_ext))/(z0+camera_offset+l*np.sin(finger_ext)))

    result = 0
    if camera_angle > -1e-6 and camera_angle < max_angle:
        # find corresponding finger angle
        f = lambda b: (y0+l*np.sin(b))/(z0+camera_offset+l*np.cos(b)) - np.tan(camera_angle)
        b = newton(f, np.pi/4.)
        z = z0 + camera_offset + l*np.cos(b)
        y = y0 + l*np.sin(b)
        result = np.sqrt(y**2+z**2)
    return result


def calculate_grip_metrics(depth_map, finger_path, saturation_distance=.02, box_size=3):
    overlap = np.maximum(0, finger_path - depth_map)

    # X = np.arange(0, len(depth_map))
    # Y = np.arange(0, len(depth_map))
    # X, Y = np.meshgrid(X, Y)
    # fig = plt.figure()
    # ax = fig.add_subplot(1,1,1,projection='3d')
    # ax.plot_wireframe(X, Y, overlap)
    # # ax.plot_wireframe(X, Y, template, color='r')
    # # ax.set_xlabel('x')
    # # plt.show()
    # # plt.imshow(overlap)
    # plt.show()

    # TODO: update this if template orientation wrong

    # find first overlap from outside to centre in three regions
    s = depth_map.shape
    # regions = [[0,s[0]/2,0,s[1]/2],
    #         [s[0]/2,s[0],0,s[1]/2],
    #         [s[0]/4,3*s[0]/4,s[1],s[1]/2]]
    # regions = [[0,s[0]/2,s[1],s[1]/2],
    #         [s[0]/2,s[0],s[1],s[1]/2],
    #         [s[0]/4,3*s[0]/4,0,s[1]/2]]
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
        # print(region_overlap)

        # p = np.sum(region_overlap, axis=0)
        p = np.sum(region_overlap, axis=1)
        print(p)
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
            sub_region_overlap = region_overlap[:,intersection:intersection+box_size]
            sub_region_finger = region_finger[:,intersection:intersection+box_size]
            quality = np.sum(sub_region_overlap.flatten()) / np.sum(sub_region_finger.flatten())

        qualities.append(quality)

    return intersections, qualities


if __name__ == '__main__':
    finger_path_template(45.*np.pi/180., 40, .3)

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