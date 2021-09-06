from numpy import sqrt

import numpy as np
import pygame
from matplotlib import cm
from pygame.locals import *

from OpenGL.GL import *
from OpenGL.GLU import *

from biobots2D.utils.tools import pyout

import os

os.environ['SDL_VIDEO_WINDOW_POS'] = "0,0"

vertices = {0: (0, 0, 0), 1: (.5, 0, -.5), 2: (.5, 0, .5), 3: (-.5, 0, .5), 4: (-.5, 0, -.5),
            5: (0, sqrt(3) / 2, 1), 6: (1, sqrt(3) / 2, 1), 7: (.5, sqrt(3), 1.5),
            8: (-.5, sqrt(3), 1.5), 9: (-1, sqrt(3) / 2, 1), 10: (-1, sqrt(3) / 2, 0),
            11: (-1.5, sqrt(3), 0.5), 12: (-1.5, sqrt(3), -0.5), 13: (-1, sqrt(3) / 2, -1),
            14: (0, sqrt(3) / 2, -1), 15: (-.5, sqrt(3), -1.5), 16: (.5, sqrt(3), -1.5),
            17: (1, sqrt(3) / 2, -1), 18: (1, sqrt(3) / 2, 0), 19: (1.5, sqrt(3), -.5),
            20: (1.5, sqrt(3), .5)
            }
edges = {0: (0, 1), 1: (1, 2), 2: (2, 0), 3: (0, 3), 4: (2, 3), 5: (0, 4), 6: (1, 4), 7: (3, 4),
         8: (2, 5), 9: (3, 5), 10: (2, 6), 11: (5, 6), 12: (5, 7), 13: (6, 7), 14: (5, 8),
         15: (7, 8), 16: (5, 9), 17: (8, 9), 18: (3, 9), 19: (3, 10), 20: (4, 10), 21: (9, 10),
         22: (9, 11), 23: (10, 11), 24: (10, 12), 25: (11, 12), 26: (10, 13),
         27: (12, 13), 28: (4, 13), 29: (1, 14), 30: (4, 14), 31: (13, 14), 32: (13, 15),
         33: (14, 15), 34: (14, 16), 35: (15, 16), 36: (14, 17), 37: (16, 17), 38: (1, 17),
         39: (6, 18), 40: (2, 18), 41: (1, 18), 42: (17, 18), 43: (18, 19), 44: (17, 19),
         45: (18, 20), 46: (19, 20), 47: (6, 20)
         }
surfaces = {0: (0, 1, 2), 1: (0, 2, 3), 2: (0, 3, 4), 3: (0, 4, 1), 4: (2, 5, 3), 5: (2, 6, 5),
            6: (5, 6, 7), 7: (5, 7, 8), 8: (5, 8, 9), 9: (3, 5, 9), 10: (3, 10, 4),
            11: (3, 9, 10), 12: (9, 11, 10), 13: (10, 11, 12), 14: (10, 12, 13), 15: (4, 10, 13),
            16: (1, 4, 14), 17: (4, 13, 14), 18: (13, 15, 14), 19: (14, 15, 16), 20: (14, 16, 17),
            21: (1, 14, 17), 22: (2, 18, 6), 23: (1, 18, 2), 24: (1, 17, 18), 25: (17, 19, 18),
            26: (18,19,20), 27: (6, 18, 20)
            }


def cmap(x):
    v = 0.4 + 0.4 * np.clip((1 - x) / 2, 0, 1)
    return cm.Greens(v)[:3]


def Cube():
    glBegin(GL_TRIANGLES)
    for ii, s in surfaces.items():
        a = np.array(vertices[s[1]]) - np.array(vertices[s[0]])
        b = np.array(vertices[s[2]]) - np.array(vertices[s[0]])
        axb = np.cross(a, b)  # get vector pointing straight up from surface
        frontwardness = np.dot(axb, np.array([0, 0, 1]))  # see how much it points forwards
        c = cmap(frontwardness)

        glColor3fv(c)
        for vertex in s:
            glVertex3fv(vertices[vertex])
    glEnd()

    glBegin(GL_LINES)
    glColor3fv((1, 1, 1))
    for ii, edge in edges.items():
        for vertex in edge:
            glVertex3fv(vertices[vertex])
    glEnd()


def main():
    pygame.init()
    display = (1920, 1050)
    # display = (800, 600)
    pygame.display.set_mode(display, DOUBLEBUF | OPENGL)

    gluPerspective(45, (display[0] / display[1]), 0.1, 50.0)

    glTranslatef(0.0, -2.5, -15)

    glEnable(GL_DEPTH_TEST)
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        # glRotatef(1, 3, 1, 1)
        glRotatef(1, 0, 1, 0)
        # glRotatef(0, 0, 1, 1)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        Cube()
        pygame.display.flip()
        pygame.time.wait(10)


main()
