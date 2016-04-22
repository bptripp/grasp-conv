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

            min_finger = np.floor(im_width/2-finger_half_width_pixels+.5)
            max_finger = np.ceil(im_width/2+finger_half_width_pixels+.5)
            template[min_finger:max_finger,im_width/2+pixel] = depths[pixel] #TODO: clean up offset

            finger_x_pixels = np.arctan(finger_xyz[0]/depths[pixel]) /rads_per_pixel # x offset of paired fingers

            min_finger = np.floor(im_width/2+finger_x_pixels-finger_half_width_pixels+.5)
            max_finger = np.ceil(im_width/2+finger_x_pixels+finger_half_width_pixels+.5)
            template[min_finger:max_finger,im_width/2-1-pixel] = depths[pixel]

            min_finger = np.floor(im_width/2-finger_x_pixels-finger_half_width_pixels+.5)
            max_finger = np.ceil(im_width/2-finger_x_pixels+finger_half_width_pixels+.5)
            template[min_finger:max_finger,im_width/2-1-pixel] = depths[pixel]

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