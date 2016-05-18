__author__ = 'bptripp'

import numpy as np
import matplotlib.pyplot as plt
import cPickle
from quaternion import angle_between_quaterions

# def interpolate(point, angle, points, angles, values, sigma_p=.01, sigma_a=(4*np.pi/180)):
#     """
#     Gaussian kernel smoothing.
#     """
#     # q = to_quaternion(get_rotation_matrix(point, angle))
#     # print(angle)
#
#     weights = np.zeros(len(values))
#     # foo = np.zeros(len(values))
#     # bar = np.zeros(len(values))
#     for i in range(len(values)):
#         # q_i = to_quaternion(get_rotation_matrix(points[:,i], angles[:,i]))
#
#         # print(q_i)
#
#         # angle = angle_between_quaterions(q, q_i)
#         # print(angle)
#
#         position_distance = np.linalg.norm(point - points[:,i])
#         angle_distance = angle[2] - angles[2,i];
#
#         # weights[i] = np.exp( -(angle**2/2/sigma_a**2) )
#         weights[i] = np.exp( -(angle_distance**2/2/sigma_a**2 + position_distance**2/2/sigma_p**2) )
#         # weights[i] = np.exp( -(angle**2/2/sigma_a**2 + distance**2/2/sigma_p**2) )
#         # foo[i] = np.exp( -(angle**2/2/sigma_a**2) )
#         # bar[i] = np.exp( -(distance**2/2/sigma_p**2) )
#
#     # print(weights)
#     # print(np.sum(weights))
#     # print(np.sum(foo))
#     # print(np.sum(bar))
#     return np.sum(weights * np.array(values)) / np.sum(weights)


def interpolate(quaternion, distance, quaternions, distances, values, sigma_a=(4*np.pi/180), sigma_d=.01):
    """
    Gaussian kernel smoothing.
    """
    weights = np.zeros(len(values))

    angle_threshold = np.cos(1.25*sigma_a) # I think this corresponds to twice this angle between quaternions
    distance_threshold = 2.5*sigma_d

    # attempt fast estimate (only considering within-threshold points) ...
    c = 0
    for i in range(len(values)):
        distance_difference = np.abs(distance - distances[i])
        if distance_difference < distance_threshold and np.dot(quaternion, quaternions[i]) > angle_threshold:
            c += 1
            angle_difference = np.abs(angle_between_quaterions(quaternion, quaternions[i]))
            weights[i] = np.exp( -(angle_difference**2/2/sigma_a**2 + distance_difference**2/2/sigma_d**2) )

    # slow estimate if not enough matches ...
    if c <= 3:
        print('slow estimate')
        for i in range(len(values)):
            distance_difference = np.abs(distance - distances[i])
            angle_difference = np.abs(angle_between_quaterions(quaternion, quaternions[i]))
            weights[i] = np.exp( -(angle_difference**2/2/sigma_a**2 + distance_difference**2/2/sigma_d**2) )

    # print(weights)
    # print(values)
    return np.sum(weights * np.array(values)) / np.sum(weights)


def check_interpolate():
    from perspective import get_quaternion_distance

    point = np.array([1e-6,.1,.1])
    angle = np.array([0,0,.9])
    points = np.array([[1e-6,.1,.1], [1e-6,.12,.1]]).T
    angles = np.array([[0,0,1], [0,0,1]]).T
    values = np.array([0,1])
    quaternion, distance = get_quaternion_distance(point[:,np.newaxis], angle[:,np.newaxis])
    quaternions, distances = get_quaternion_distance(points, angles)

    # print(quaternion)
    # print(distance)
    # print(quaternions)
    # print(distances)

    # estimate = interpolate(point, angle, points, angles, values, sigma_p=.01, sigma_a=(4*np.pi/180))
    estimate = interpolate(quaternion[0], distance[0], quaternions, distances, values, sigma_d=.01, sigma_a=(4*np.pi/180))
    print(estimate)


def test_interpolation_accuracy(points, angles, metrics, n_examples):
    """
    Compare interpolated vs. actual metrics by leaving random
    examples out of interpolation set and estimating them.
    """
    from perspective import get_quaternion_distance

    quaternions, distances = get_quaternion_distance(points, angles)

    actuals = []
    interpolateds = []
    for i in range(n_examples):
        print(i)
        one = np.random.randint(0, len(metrics))
        others = range(one)
        others.extend(range(one+1, len(metrics)))
        others = np.array(others)

        actuals.append(metrics[one])

        interpolated = interpolate(quaternions[one,:], distances[one], quaternions[others,:], distances[others], metrics[others],
                                   sigma_d=.01, sigma_a=(8*np.pi/180))
        interpolateds.append(interpolated)
        # print(interpolated - metrics[one])

    # print(np.corrcoef(actuals, interpolateds))
    return actuals, interpolateds


def plot_interp_error_vs_density():
    with open('spatula-perspectives-smoothed.pkl', 'rb') as f:
        (points, angles, metrics, collisions, smoothed) = cPickle.load(f)
    metrics = np.array(metrics)
    smoothed = np.array(smoothed)

    numbers = [250, 500, 1000, 2000, 4000]
    metric_errors = []
    smoothed_errors = []
    for n in numbers:
        actuals, interpolateds = test_interpolation_accuracy(points[:,:n], angles[:,:n], metrics[:n], 500)
        metric_errors.append(np.mean( (np.array(actuals)-np.array(interpolateds))**2 )**.5)

        actuals, interpolateds = test_interpolation_accuracy(points[:,:n], angles[:,:n], smoothed[:n], 500)
        smoothed_errors.append(np.mean( (np.array(actuals)-np.array(interpolateds))**2 )**.5)

    plt.plot(numbers, smoothed_errors)
    plt.plot(numbers, metric_errors)
    plt.show()


if __name__ == '__main__':
    # check_interpolate()
    plot_interp_error_vs_density()
