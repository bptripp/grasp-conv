__author__ = 'bptripp'

import numpy as np


def to_quaternion(rotation_matrix):
    # from Siciliano & Khatib pg. 12 and quaternion.m by Tincknell
    r = rotation_matrix
    # e0 = .5 * np.sqrt(1 + r[0][0] + r[1][1] + r[2][2])
    e0 = .5 * np.sqrt(np.maximum(0, r[0][0] + r[1][1] + r[2][2] + 1))
    if e0 == 0:
        e1 = np.sqrt(np.maximum(0, -0.5 * (r[1][1] + r[2][2]))) * np.sign(-r[1][2])
        e2 = np.sqrt(np.maximum(0, -0.5 * (r[0][0] + r[2][2]))) * np.sign(-r[0][2])
        e3 = np.sqrt(np.maximum(0, -0.5 * (r[0][0] + r[1][1]))) * np.sign(-r[0][1])
    else:
        e1 = (r[2][1] - r[1][2]) / (4*e0)
        e2 = (r[0][2] - r[2][0]) / (4*e0)
        e3 = (r[1][0] - r[0][1]) / (4*e0)
    return np.array([e0,e1,e2,e3])

"""
This is from quaternion.m by Mark Tincknell

function eout = RotMat2e( R )
% function eout = RotMat2e( R )
% One Rotation Matrix -> one quaternion
eout    = zeros(4,1);
if ~all( all( R == 0 ))
    eout(1) = 0.5 * sqrt( max( 0, R(1,1) + R(2,2) + R(3,3) + 1 ));
    if eout(1) == 0
        eout(2) = sqrt( max( 0, -0.5 *( R(2,2) + R(3,3) ))) * sgn( -R(2,3) );
        eout(3) = sqrt( max( 0, -0.5 *( R(1,1) + R(3,3) ))) * sgn( -R(1,3) );
        eout(4) = sqrt( max( 0, -0.5 *( R(1,1) + R(2,2) ))) * sgn( -R(1,2) );
    else
        eout(2) = 0.25 *( R(3,2) - R(2,3) )/ eout(1);
        eout(3) = 0.25 *( R(1,3) - R(3,1) )/ eout(1);
        eout(4) = 0.25 *( R(2,1) - R(1,2) )/ eout(1);
    end
end
end % RotMat2e
"""

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


def difference_between_quaternions(e1, e2):
    """
    TODO: look this up when internet is back online
    :param e1: From here
    :param e2: To here
    :return: Quaternion of rotation
    """
    # print(e1)
    # print(e2)

    guess = quaterion_product(quaternion_conj(e1), e2)
    # print(guess)

    r1 = from_quaternion(e1)
    r2 = from_quaternion(e2)
    guess2 = to_quaternion(np.dot(np.linalg.inv(r1), r2))
    # print(guess2)

    p = np.array([0, .2, -.4, 1.1])
    # print(quaterion_product(guess, quaterion_product(p, quaternion_conj(guess))))
    # print(quaterion_product(guess2, quaterion_product(p, quaternion_conj(guess2))))

    return guess2


def equal(e1, e2, tol=1e-6):
    # TODO: account for double cover
    equal = True
    for i in range(4):
        if e1[i]-e2[i] > tol:
            equal = False
    return equal


def check_quaternion():
    r = np.array([[0.86230895, 0.20974727, -0.46090059],
        [ 0.50269225, -0.4642552,   0.72922398],
        [-0.06102276, -0.86050752, -0.50576974]])

    error = r - from_quaternion(to_quaternion(r))
    assert np.std(error.flatten()) < 1e-6


def check_difference():
    # r1 = np.array([[0.86230895, 0.20974727, -0.46090059],
    #     [ 0.50269225, -0.4642552,   0.72922398],
    #     [-0.06102276, -0.86050752, -0.50576974]])
    r1 = np.array([[0.31578947, -0.9486833, -0.01664357],
        [-0.94736842, -0.31622777, 0.0499307],
        [-0.05263158, 0., -0.998614]])
    r2 = np.array([[1, 0, 0], [0, np.cos(np.pi), -np.sin(np.pi)], [0, np.sin(np.pi), np.cos(np.pi)]])
    # print(r2)
    difference_between_quaternions(to_quaternion(r1), to_quaternion(r2))

    r3 = np.array([[1, 0, 0], [0, np.cos(np.pi/2), -np.sin(np.pi/2)], [0, np.sin(np.pi/2), np.cos(np.pi/2)]])
    d = difference_between_quaternions(to_quaternion(r3), to_quaternion(r2))
    # print(d)
    # print(to_quaternion(r3))
    assert equal(d, to_quaternion(r3))


if __name__ == '__main__':
    # check_quaternion()
    check_difference()

    # r1 = np.array([[0.86230895, 0.20974727, -0.46090059],
    #     [ 0.50269225, -0.4642552,   0.72922398],
    #     [-0.06102276, -0.86050752, -0.50576974]])
    # r2 = np.array([[1, 0, 0], [0, np.cos(np.pi), -np.sin(np.pi)], [0, np.sin(np.pi), np.cos(np.pi)]])
    # # a = .5*np.pi
    # r3 = np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]])
    # e1 = to_quaternion(r1)
    # e2 = to_quaternion(r2)
    # e3 = to_quaternion(r3)
    # # print(e1)
    # # print(e2)
    # # print(np.linalg.norm(e1))
    #
    # print(np.dot(e2,e3))

