__author__ = 'bptripp'

from os.path import join
import matplotlib.pyplot as plt
import cPickle
import numpy as np
import scipy
from data import load_all_params

# objects, gripper_pos, gripper_orient, labels = load_all_params('../../grasp-conv/data/output_data.csv')

# f = file('../data/metrics-objects.pkl', 'rb')
# oi, oq, of = cPickle.load(f) # intersections, qualities, files
# f.close()

# f = file('../data/metrics-support.pkl', 'rb')
# si, sq, sf = cPickle.load(f)
# f.close()

def count_cases(correlates, labels):
    correlates = correlates.astype(int)
    cases = np.zeros((np.max(correlates)+1,2), dtype=int)
    for c, l in zip(correlates, labels):
        cases[c,l] += 1
    return cases


def get_collisions(oi, si):
    oi = np.array(oi)
    si = np.array(si)

    # collision if any si < oi and si is not None
    oi = np.where(oi == np.array(None), 900, oi)
    si = np.where(si == np.array(None), 1000, si)
    return np.sum(si < oi, axis=1) > 0


def check_objects_centred():
    """
    The hand is supposed to be aimed at a point on the object, so central
    pixel(s) should be on object, otherwise probably a bug (e.g. something
    wrong with how mesh imported into V-REP).
    """
    image_dir = '../../grasp-conv/data/obj_depths/'
    im_width = 80
    camera_offset=.45
    near_clip=.25
    far_clip=.8

    from os import listdir
    from os.path import isfile, join
    from heuristic import finger_path_template
    import scipy

    finger_path = finger_path_template(45.*np.pi/180., im_width, camera_offset)

    bad = []
    background_threshold = 254

    for f in listdir(image_dir):
        image_filename = join(image_dir, f)
        if isfile(image_filename) and f.endswith('.png'):
            image = scipy.misc.imread(image_filename)
            if np.min(image[39:41,39:41]) >= background_threshold:
                bad.append(f)
                print(f)

    print(len(bad))


def regions_at_intersections(distance, intersections,
                                    axis=0, directions=[-1,-1,1], length=5,
                                    spans=[[32,38],[42,48],[37,43]]):
    """
    :param distance: 2D distance map
    :param intersections: pixel coords of finger-surface intersections
    """
    regions = []
    for i in range(len(intersections)):
        direction = directions[i]
        span = spans[i]

        if axis==0:
            d = distance
        else:
            d = distance.T

        if direction < 0:
            d = d[::-1,:]

        if intersections[i] is None:
            region = np.zeros((length,span[1]-span[0]))
        else:
            region = d[intersections[i]:intersections[i]+length,span[0]:span[1]]

        regions.append(region)

    return regions


#TODO: watch for outliers due to numerical sensitivity at sharp edges
def surface_slope(distance_map, threshold):
    """
    Fits a plane to given points, ignoring values greater than a threshold distance
    (meant to be the background).
    """

    x = np.arange(0, distance_map.shape[1])
    y = np.arange(0, distance_map.shape[0])
    X, Y = np.meshgrid(x, y)

    features = np.vstack((X.ravel(), Y.ravel(), np.ones_like(Y).ravel())).T
    distances = distance_map.ravel()

    include = distances <= threshold

    if np.isnan(np.std(distances[include])):
        return [0,0,np.mean(distances[include])]
    else:
        x, residuals, rank, s = np.linalg.lstsq(features[include,:], distances[include])
        return x


def check_region_example():
    camera_offset=.45
    far_clip=.8

    import scipy
    image = scipy.misc.imread('../../grasp-conv/data/obj_depths/1_Coffeecup_final-03-Mar-2016-18-50-40-1.png')
    rescaled_distance = image / 255.0
    distance = rescaled_distance*(far_clip-camera_offset)+camera_offset

    intersections = [38,36,21]
    regions = regions_at_intersections(distance, intersections)
    print(np.array(regions))

    for region in regions:
        print(surface_slope(region, far_clip-.005))

    plt.imshow(distance)
    plt.show()


#TODO: test
def angle_from_vertical(gripper_orient):
    from depthmap import rot_matrix
    R = rot_matrix(gripper_orient[0], gripper_orient[1], gripper_orient[2])
    camera_vector = np.dot(R, np.array([0,0,1]))
    return np.arccos(-camera_vector[2])


def object_area(distance, threshold):
    return(np.sum(distance < threshold) / float(distance.size))


def object_centre(distance, threshold):
    object = distance < threshold
    x = np.arange(0, distance.shape[1])
    y = np.arange(0, distance.shape[0])
    X, Y = np.meshgrid(x, y)
    return np.mean(X[object]), np.mean(Y[object])


def check_correlates():
    data_file = '../../grasp-conv/data/output_data.csv'
    objects, gripper_pos, gripper_orient, labels, power_pinch = load_all_params(data_file, return_power_pinch=True)
    seq_nums = np.arange(len(objects)) % 1000

    camera_offset=.45
    far_clip=.8
    threshold = far_clip - .005

    for i in range(5):
        image_filename = join('../../grasp-conv/data/obj_depths', objects[i][:-4] + '-' + str(seq_nums[i]) + '.png')
        image = scipy.misc.imread(image_filename)
        rescaled_distance = image / 255.0
        distance = rescaled_distance*(far_clip-camera_offset)+camera_offset

        print('angle: ' + str(angle_from_vertical(gripper_orient[i])))
        print('area: ' + str(object_area(distance, threshold)))
        print('centre: ' + str(object_centre(distance, threshold)))

        plt.imshow(distance)
        plt.show()


def get_correlates():
    """
    Create a CSV file with a bunch of measures that may be correlates of grasp success.
    """
    from heuristic import finger_path_template, calculate_grip_metrics

    data_file = '../../grasp-conv/data/output_data.csv'
    objects, gripper_pos, gripper_orient, labels, power_pinch = load_all_params(data_file, return_power_pinch=True)
    seq_nums = np.arange(len(objects)) % 1000

    labels = np.array(labels)
    power_pinch = np.array(power_pinch)

    camera_offset=.45
    far_clip=.8
    fov = 45.*np.pi/180.
    im_width=80
    finger_path = finger_path_template(fov, im_width, camera_offset)

    background_threshold = far_clip - .005

    result = []
    for i in range(len(labels)):
        data = [labels[i], power_pinch[i]]

        image_filename = join('../../grasp-conv/data/obj_depths', objects[i][:-4] + '-' + str(seq_nums[i]) + '.png')
        distance_image = scipy.misc.imread(image_filename)
        rescaled_distance = distance_image / 255.0
        distance = rescaled_distance*(far_clip-camera_offset)+camera_offset
        intersections, qualities = calculate_grip_metrics(distance, finger_path)

        regions = regions_at_intersections(distance, intersections)
        for region in regions:
            slope = surface_slope(region, background_threshold)
            data.extend(slope[:2])

        image_filename = join('../../grasp-conv/data/obj_overlap', objects[i][:-4] + '-' + str(seq_nums[i]) + '-overlap.png')
        overlap_image = scipy.misc.imread(image_filename)
        regions = regions_at_intersections(overlap_image, intersections, length=3)
        for region in regions:
            data.append(np.sum(region > 0) / float(region.size))

        data.append(angle_from_vertical(gripper_orient[i]))
        data.append(object_area(distance, background_threshold))
        data.extend(object_centre(distance, background_threshold))

        # intersections and quality metrics
        intersections = np.where(intersections == np.array(None), im_width, intersections).astype(float)
        data.extend(intersections)
        data.extend(qualities)

        # symmetry
        rel_int = intersections / (im_width/2.)
        data.append((rel_int[0]-rel_int[2])**2)
        data.append((rel_int[1]-rel_int[2])**2)

        # intersections with support box
        image_filename = join('../../grasp-conv/data/support_depths', objects[i][:-4] + '-' + str(seq_nums[i]) + '.png')
        distance_image = scipy.misc.imread(image_filename)
        rescaled_distance = distance_image / 255.0
        distance = rescaled_distance*(far_clip-camera_offset)+camera_offset
        support_intersections, support_qualities = calculate_grip_metrics(distance, finger_path)
        support_intersections = np.where(support_intersections == np.array(None), im_width, support_intersections).astype(float)
        data.extend(support_intersections)

        result.append(data)

    # fig = plt.figure(figsize=(5,10))
    # fig.add_subplot(2,1,1)
    # plt.hist(power_pinch[labels<.5], range=(0, .20), bins=100)
    # fig.add_subplot(2,1,2)
    # plt.hist(power_pinch[labels>.5], range=(0, .20), bins=100)
    # plt.show()

    return np.array(result)

if __name__ == '__main__':
    # check_objects_centred()
    # save_correlates()

    # weights = surface_slope(np.random.rand(4,6))
    # print(weights)

    # check_region_example()
    #TODO: slopes and fractions of regions within finger path

    # import scipy
    # from os.path import join
    # image_dir = '../../grasp-conv/data/obj_overlap/'
    # image_file = '1_Coffeecup_final-03-Mar-2016-18-50-40-1-overlap.png'
    # image = scipy.misc.imread(join(image_dir, image_file))
    #
    # intersections = [38,36,21]
    # regions = regions_at_intersections(image, intersections, length=3)
    # regions = np.array(regions)
    # for region in regions:
    #     print(region)
    #     print(np.sum(region > 0) / float(region.size))
    #
    # plt.imshow(image)
    # plt.show()

    # check_correlates()

    correlates = get_correlates()

    import csv
    with open('correlates.csv', 'wb') as csvfile:
        cw = csv.writer(csvfile, delimiter=',')
        for c in correlates:
            cw.writerow(c)

