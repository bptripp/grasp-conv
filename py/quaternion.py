__author__ = 'bptripp'

import numpy as np


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


def quaterion_product(e1, e2):
    # from https://en.wikipedia.org/wiki/Quaternion
    result = [
        e1[0]*e2[0] - e1[1]*e2[1] - e1[2]*e2[2] - e1[3]*e2[3],
        e1[0]*e2[1] + e1[1]*e2[0] + e1[2]*e2[3] - e1[3]*e2[2],
        e1[0]*e2[2] - e1[1]*e2[3] + e1[2]*e2[0] + e1[3]*e2[1],
        e1[0]*e2[3] + e1[1]*e2[2] - e1[2]*e2[1] + e1[3]*e2[0]
    ]
    return np.array(result)


def quaternion_conj(e):
    return np.array([e[0], -e[1], -e[2], -e[3]])


def angle_between_quaterions(e1, e2):
    # from http://math.stackexchange.com/questions/90081/quaternion-distance
    # return np.arccos(2*(e1[0]*e2[0]+e1[1]*e2[1]+e1[2]*e2[2]+e1[3]*e2[3])-1)

    # from http://math.stackexchange.com/questions/167827/compute-angle-between-quaternions-in-matlab
    z = quaterion_product(e1, quaternion_conj(e2))
    # print(z[0])
    return 2*np.arccos(np.clip(z[0], -1, 1))


def check_quaternion():
    r = np.array([[0.86230895, 0.20974727, -0.46090059],
        [ 0.50269225, -0.4642552,   0.72922398],
        [-0.06102276, -0.86050752, -0.50576974]])

    error = r - from_quaternion(to_quaternion(r))
    assert np.std(error.flatten()) < 1e-6


if __name__ == '__main__':
    check_quaternion()
