__author__ = 'bptripp'

from os.path import join
import cPickle
import matplotlib.pyplot as plt
import scipy.misc
import numpy as np
from perspective import get_rotation_matrix

def plot_correct_point_scatter():
    n_points = 200

    with open('../data/neuron-points.pkl', 'rb') as f:
        neuron_points, neuron_angles = cPickle.load(f)

    with open('perspective-data-small.pkl', 'rb') as f:
        image_files, metrics = cPickle.load(f)

    # print(metrics.shape)

    index = 600

    points = neuron_points[:,:n_points]
    offsets = np.zeros_like(points)
    for i in range(n_points):
        r = get_rotation_matrix(neuron_points[:,i], neuron_angles[:,i])
        offsets[:,i] = points[:,i] + np.dot(r, np.array([0,.005,0]))

    from mpl_toolkits.mplot3d import axes3d, Axes3D
    fig = plt.figure(figsize=(20,10))

    ax = fig.add_subplot(1,2,1,projection='3d')
    ax.scatter(points[0,:], points[1,:], points[2,:],
               c=metrics[index,:n_points], cmap='autumn', depthshade=False)
    ax.scatter(offsets[0,:], offsets[1,:], offsets[2,:],
               c=metrics[index,:n_points], cmap='autumn', depthshade=False, s=10)

    plt.subplot(1,2,2)
    image = scipy.misc.imread(join('../../grasp-conv/data/eye-perspectives', image_files[index]))
    plt.imshow(image)
    plt.title(image_files[index])
    plt.show()

    # print(neuron_points[:,:n_points])


def plot_predictions():
    with open('perspective-predictions-better.pkl') as f:
        targets, predictions = cPickle.load(f)
    # print(targets.shape)
    # print(predictions.shape)

    plt.figure(figsize=(9,6))
    for i in range(25):
        plt.subplot(5,5,i)
        plt.scatter(targets[:,i], predictions[:,i], s=1)
        c = np.corrcoef(targets[:,i], predictions[:,i])[0,1]
        plt.gca().axes.xaxis.set_ticks([])
        plt.gca().axes.yaxis.set_ticks([])
        print(c)
    plt.tight_layout()
    plt.show()


def plot_points_with_correlations():
    with open('perspective-predictions-big-0.pkl') as f:
        targets, predictions = cPickle.load(f)

    n_points = targets.shape[1]

    r = []
    for i in range(n_points):
        r.append(np.corrcoef(targets[:,i], predictions[:,i])[0,1])

    with open('perspective-predictions-big-9.pkl') as f:
        targets, predictions = cPickle.load(f)

    r_better = []
    for i in range(n_points):
        r_better.append(np.corrcoef(targets[:,i], predictions[:,i])[0,1])

    with open('../data/neuron-points.pkl', 'rb') as f:
        neuron_points, neuron_angles = cPickle.load(f)

    # from mpl_toolkits.mplot3d import axes3d, Axes3D
    # fig = plt.figure()
    # ax = fig.add_subplot(1,1,1,projection='3d')
    # ax.scatter(neuron_points[0,:n_points], neuron_points[1,:n_points], neuron_points[2,:n_points],
    #            c=r, cmap='autumn', depthshade=False, s=40)
    # plt.show()

    angles_from_vertical = []
    for i in range(n_points):
        # hor = np.sqrt(neuron_points[0,i]**2 + neuron_points[1,i]**2)
        # ver = neuron_points[2,i]
        # angles_from_vertical.append(np.arctan(hor / ver))
        norm = np.linalg.norm(neuron_points[:,i])
        angles_from_vertical.append(np.arccos(neuron_points[2,i] / norm))

    fig = plt.figure()
    plt.scatter(angles_from_vertical, r, s=30, facecolors='none', edgecolors='r')
    plt.scatter(angles_from_vertical, r_better, color='r', s=30)
    plt.xlim([0,np.pi])
    plt.ylim([0,1])
    plt.tick_params(axis='both', labelsize=18)
    plt.xlabel('angle between eye and hand (rad)', fontsize=18)
    plt.ylabel('target-prediction correlation', fontsize=18)
    plt.show()


def katsuyama_depths():
    K = [[-0.1500, -0.1500],
        [-0.1960, -0.0812],
        [-0.2121, 0],
        [-0.1960, 0.0812],
        [-0.1500, 0.1500],
        [-0.0812, 0.1960],
        [0.0000, 0.2121],
        [0.0812, 0.1960],
        [0.1500, 0.1500],
        [-0.2500, -0.2500],
        [-0.3266, -0.1353],
        [-0.3536, 0],
        [-0.3266, 0.1353],
        [-0.2500, 0.2500],
        [-0.1353, 0.3266],
        [0, 0.3536],
        [0.1353, 0.3266],
        [0.2500, 0.2500],
        [0, 0]]
    K = np.array(K)

    # Original numbers for x and y in cm, whereas we want m. This requires multiplying
    # by 10000, but such shapes are much sharper than objects, so we drop by 10x
    print(K)
    K = 1000. * K

    depths = []
    for i in range(K.shape[0]):
        depths.append(katsuyama_depth(K[i,0], K[i,1]))
    return np.array(depths)


def katsuyama_depth(K1, K2, c=.8, im_width=80, fov=30, near_clip=.6, far_clip=1):
    # note their display was about 36 degrees

    rads_per_pixel = (fov*np.pi/180.) / im_width
    hor_angles = np.arange(-im_width/2+.5, im_width/2+.5, 1) * rads_per_pixel
    ver_angles = hor_angles

    ta = np.tan(hor_angles)**2
    tb = np.tan(ver_angles)**2

    # solve quadratic equation ...
    b = -1
    depth = np.zeros((len(ver_angles),len(hor_angles)))
    for i in range(len(ver_angles)):
        for j in range(len(hor_angles)):
            a = .5 * (K1*tb[i] + K2*ta[j])

            if np.abs(a) < 1e-6:
                depth[i,j] = c
            elif b**2 - 4*a*c < 0:
                depth[i,j] = far_clip
            else:
                # we want the closer hit
                depth[i,j] = (-b - np.sqrt(b**2 - 4*a*c)) / (2*a)

    depth = np.minimum(far_clip, np.maximum(near_clip, depth))
    return depth


def get_truncated_model(structure_file, weights_file, n_layers):
    from keras.models import model_from_json
    from keras.optimizers import Adam
    from keras.models import Sequential

    model = model_from_json(open(structure_file).read())
    model.load_weights(weights_file)

    print(str(len(model.layers)) + ' layers')

    truncated = Sequential()
    for i in range(n_layers):
        truncated.add(model.layers[i])

    adam = Adam(lr=0.000001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    truncated.compile(loss='mse', optimizer=adam)

    return truncated


def save_katsutama_responses(structure_file, weights_file, n_layers):
    model = get_truncated_model(structure_file, weights_file, n_layers)
    x = katsuyama_depths()[:,np.newaxis,:,:]
    responses = model.predict_on_batch(x)
    print(responses.shape)

    with open('katsuyama-' + str(n_layers) + '.pkl', 'wb') as f:
        cPickle.dump(responses, f)


def plot_katsuyama_responses(n_layers):
    with open('../data/katsuyama-' + str(n_layers) + '.pkl') as f:
        data = cPickle.load(f)

    if len(data.shape) == 4:
        data = data[:,:,40,40]

    print(data - data[0,:])

    plt.plot(range(19), data - data[0,:])
    plt.show()

if __name__ == '__main__':
    # plot_correct_point_scatter()
    # plot_predictions()
    # plot_points_with_correlations()

    # depth = katsuyama_depth(200, -1.5, .8)

    #depths = katsuyama_depths()
    #for i in range(depths.shape[0]):
    #    plt.imshow(depths[i,:,:])
    #    print(depths[i,0,:])
    #    plt.show()

    layers = [2,4,6,9,12]
    for l in layers:
        print('running ' + str(l) + ' layers')
        save_katsutama_responses('p-model-architecture-big.json', 'p-model-weights-big-9.h5', l)

    # plot_katsuyama_responses(9)
