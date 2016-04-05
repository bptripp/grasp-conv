# The PyOpenGL code herein is loosely based on https://pythonprogramming.net/opengl-rotating-cube-example-pyopengl-tutorial/

import numpy as np
import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *


def loadOBJ(filename):
    """
    Adapted from http://www.nandnor.net/?p=86
    """
    verts = []
    faces = []
    for line in open(filename, "r"):
        vals = line.split()
        if vals[0] == "v":
            v = map(float, vals[1:4])
            verts.append(v)
        if vals[0] == "f":
            f = map(int, vals[1:4])
            faces.append(f)
    return verts, faces


def rot_matrix(a, b, g):
    """
    :param a: alpha (rotation about x axis)
    :param b: beta (rotation about y axis)
    :param g: gamma (rotation about z axis)
    :return: matrix that does rotations in order gamma, beta, alpha in global reference frame
    """
    return np.array([[np.cos(b)*np.cos(g), -np.cos(b)*np.sin(g), -np.sin(b)],
        [np.cos(a)*np.sin(g)-np.cos(g)*np.sin(a)*np.sin(b), np.cos(a)*np.cos(g)+np.sin(a)*np.sin(b)*np.sin(g), -np.cos(b)*np.sin(a)],
        [np.sin(a)*np.sin(g)+np.cos(a)*np.cos(g)*np.sin(b), np.cos(g)*np.sin(a)-np.cos(a)*np.sin(b)*np.sin(g), np.cos(a)*np.cos(b)]])


def move_vertices(gripper_pos, gripper_orient, verts):
    rm = rot_matrix(-gripper_orient[0], -gripper_orient[1], -gripper_orient[2])

    offset = -np.array(gripper_pos)
    new_verts = []
    for v in verts:
        new_verts.append(np.dot(rm, v) + offset)
    return new_verts


class Display(object):

    def __init__(self, imsize=(200,200)):
        self.imsize = imsize

        pygame.init()
        pygame.display.set_mode(imsize, DOUBLEBUF|OPENGL)

        # see https://open.gl/depthstencils
        glEnable(GL_DEPTH_TEST)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        self.set_perspective()
        self.set_camera_position()

    def __del__(self):
        pygame.display.quit()

    def set_perspective(self, fov=60, near_clip=.1, far_clip=3.0):
        gluPerspective(fov, (self.imsize[0]/self.imsize[1]), near_clip, far_clip)

    def set_camera_position(self):
        #TODO: user-specified position, orientation
        # glTranslatef(0.0,0.0, -1)
        glTranslatef(0.0,0.0,0.0)

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
        return glReadPixels(0, 0, self.imsize[0], self.imsize[1], GL_DEPTH_COMPONENT, GL_INT)



if __name__ == '__main__':
    filename = '../data/obj_files/24_bowl-02-Mar-2016-07-03-29.obj'
    verts, faces = loadOBJ(filename)
    gripper_pos = [0.1171, -0.1033, 0.3716]
    gripper_orient = [-2.5501, -0.2180, 0.6896]

    new_verts = move_vertices(gripper_pos, gripper_orient, verts)

    d = Display()
    d.set_mesh(new_verts, faces)
    depth = d.read_depth()

    import numpy as np
    print(np.min(depth.flatten()))
    print(np.max(depth.flatten()))



