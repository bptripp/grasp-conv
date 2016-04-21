# The PyOpenGL code herein is loosely based on https://pythonprogramming.net/opengl-rotating-cube-example-pyopengl-tutorial/

import numpy as np
import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *


def loadOBJ(filename):
    """
    Adapted from http://www.nandnor.net/?p=86
    See also http://paulbourke.net/dataformats/obj/
    """
    verts = []
    faces = []
    for line in open(filename, "r"):
        vals = line.split()
        if vals[0] == "v":
            v = map(float, vals[1:4])
            verts.append(v)
        if vals[0] == "f":
            # f = map(int, vals[1:4])
            f = []
            for v in vals[1:]:
                w = v.split("/")
                f.append(int(w[0]))
            faces.append(f)
    return verts, faces


def rot_matrix(a, b, g):
    """
    :param a: alpha (rotation about x axis)
    :param b: beta (rotation about y axis)
    :param g: gamma (rotation about z axis)
    :return: matrix that does rotations in order gamma, beta, alpha in global reference frame
    """
    #TODO: maybe rotations are clockwise in opengl? let's see
    # a = -a
    b = -b
    # g = -g

    result = np.array([[np.cos(b)*np.cos(g), -np.cos(b)*np.sin(g), -np.sin(b)],
        [np.cos(a)*np.sin(g)-np.cos(g)*np.sin(a)*np.sin(b), np.cos(a)*np.cos(g)+np.sin(a)*np.sin(b)*np.sin(g), -np.cos(b)*np.sin(a)],
        [np.sin(a)*np.sin(g)+np.cos(a)*np.cos(g)*np.sin(b), np.cos(g)*np.sin(a)-np.cos(a)*np.sin(b)*np.sin(g), np.cos(a)*np.cos(b)]])

    A = np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]])
    B = np.array([[np.cos(b), 0, -np.sin(b)], [0, 1, 0], [np.sin(b), 0, np.cos(b)]])
    G = np.array([[np.cos(g), -np.sin(g), 0], [np.sin(g), np.cos(g), 0], [0, 0, 1]])
    # result2 = np.dot(A, np.dot(B, G))
    # result = np.dot(G, np.dot(B, A))
    # print(result - result2)

    return result

# def move_vertices(gripper_pos, gripper_orient, verts):
#     rm = rot_matrix(-gripper_orient[0], -gripper_orient[1], -gripper_orient[2])
#     # print(rm)
#
#     offset = np.array(gripper_pos)
#     # foo_verts = np.dot(rm, (verts + offset).T).T
#     new_verts = []
#     for v in verts:
#         new_verts.append(np.dot(rm, v - offset))
#
#     # print(foo_verts[0,:])
#     # print(new_verts[0])
#
#     return np.array(new_verts)


class Display(object):

    def __init__(self, imsize=(200,200)):
        self.imsize = imsize

        pygame.init()
        pygame.display.set_mode(imsize, DOUBLEBUF|OPENGL)

        # see https://open.gl/depthstencils
        glEnable(GL_DEPTH_TEST)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        self.set_perspective()
        # self.set_camera_position()

    def close(self):
        pygame.display.quit()

    # def __del__(self):
    #     self.close()

    def set_perspective(self, fov=45, near_clip=.2, far_clip=1.0):
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(fov, (self.imsize[0]/self.imsize[1]), near_clip, far_clip)

    def set_camera_position(self, pos, orient, offset):
        look_from = pos - np.dot(rot_matrix(orient[0], orient[1], orient[2]), [0, 0, offset])
        look_at = pos
        up = np.dot(rot_matrix(orient[0], orient[1], orient[2]), [0, 1, 0])
        # print(look_from)
        # print(look_at)
        # print(up)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        gluLookAt(look_from[0], look_from[1], look_from[2],
            look_at[0], look_at[1], look_at[2],
            up[0], up[1], up[2])

    # def set_camera_position(self):
    #     #TODO: user-specified position, orientation
    #     glMatrixMode(GL_MODELVIEW)
    #     glLoadIdentity()
    #     gluLookAt(0.0, 0.0, -.3,
    #       0.0, 0.0, 0.0,
    #       0.0, 1.0, 0.0)
    #     # glTranslatef(0.0,0.0,-0.4)

    def set_mesh(self, verts, faces):
        glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT)
        glColor3f(1, 0, 0)
        glBegin(GL_TRIANGLES)
        for face in faces:
            glVertex3fv(verts[face[0]-1])
            glVertex3fv(verts[face[1]-1])
            glVertex3fv(verts[face[2]-1])
        glEnd()

    def read_depth(self):
        pygame.display.flip()
        depth = glReadPixels(0, 0, self.imsize[0], self.imsize[1], GL_DEPTH_COMPONENT, GL_INT)
        # depth[depth == np.max(depth.flatten())] = 0
        return depth

# http://stackoverflow.com/questions/13685386/matplotlib-equal-unit-length-with-equal-aspect-ratio-z-axis-is-not-equal-to
def set_axes_equal(ax):
    '''Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    '''

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_mean = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_mean = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_mean = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5*max([x_range, y_range, z_range])

    ax.set_xlim3d([x_mean - plot_radius, x_mean + plot_radius])
    ax.set_ylim3d([y_mean - plot_radius, y_mean + plot_radius])
    ax.set_zlim3d([z_mean - plot_radius, z_mean + plot_radius])


# Adapted from http://web.archive.org/web/20130416194336/http://olivers.posterous.com/linear-depth-in-glsl-for-real
def get_distance(GL_depth, clip_near, clip_far):
    #TODO: scale to -1 to 1?
    z_b = GL_depth / 2147483647.0 # 2**32/2-1
    z_n = 2.0 * z_b - 1.0
    return 2.0 * clip_near * clip_far / (clip_far + clip_near - z_n * (clip_far - clip_near))


def check_gripper_points_to_example_object():
    filename = '../data/obj_files/24_bowl-02-Mar-2016-07-03-29.obj'
    verts, faces = loadOBJ(filename)
    gripper_pos = [0.12086, -0.026054, 0.40646]
    gripper_orient = [3.0652, -0.2162, -2.4642]
    verts = np.array(verts)
    minz = np.min(verts, axis=0)[2]
    verts[:,2] = verts[:,2] + 0.2 - minz

    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import axes3d, Axes3D

    rm = rot_matrix(gripper_orient[0], gripper_orient[1], gripper_orient[2])
    offset = np.dot(rm, [0, 0, .2])

    fig = plt.figure()
    ax = Axes3D(fig)
    show_flags = np.random.rand(verts.shape[0]) < .2
    ax.scatter(verts[show_flags,0], verts[show_flags,1], verts[show_flags,2], c='b')
    ax.scatter(gripper_pos[0], gripper_pos[1], gripper_pos[2], c='r')
    ax.plot([gripper_pos[0], gripper_pos[0]+offset[0]],
            [gripper_pos[1], gripper_pos[1]+offset[1]],
            [gripper_pos[2], gripper_pos[2]+offset[2]])

    # ax.scatter(new_verts[show_flags,0], new_verts[show_flags,1], new_verts[show_flags,2], c='r')
    ax.set_xlim(-.2, .2)
    ax.set_ylim(-.2, .2)
    ax.set_zlim(0, .4)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    plt.show()




if __name__ == '__main__':
    # check_gripper_points_to_example_object()

    filename = '../data/obj_files/24_bowl-02-Mar-2016-07-03-29.obj'
    verts, faces = loadOBJ(filename)
    # gripper_pos = [0.1171, -0.1033, 0.3716]
    # gripper_orient = [-2.5501, -0.2180, 0.6896]
    gripper_pos = [0.12086, -0.026054, 0.40646]
    gripper_orient = [3.0652, -0.2162, -2.4642]

    verts = np.array(verts)
    minz = np.min(verts, axis=0)[2]
    verts[:,2] = verts[:,2] + 0.2 - minz

    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import axes3d, Axes3D

    camera_offset = .3 #distance of camera behind hand
    d = Display(imsize=(100,100))
    d.set_camera_position(gripper_pos, gripper_orient, camera_offset)
    d.set_mesh(verts, faces)
    depth = d.read_depth()
    d.close()

    distance = get_distance(depth, .2, 1.0)

    fig = plt.figure()
    ax = Axes3D(fig)
    X = np.arange(0, 100)
    Y = np.arange(0, 100)
    X, Y = np.meshgrid(X, Y)
    ax.plot_wireframe(X, Y, distance-camera_offset)
    plt.show()

    # plt.imshow(depth, cmap='gray')
    # plt.savefig('test.png')
    # plt.show()




