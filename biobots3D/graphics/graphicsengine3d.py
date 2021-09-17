import atexit
import colorsys
import os

import cv2
import numpy as np
import pygame
# from OpenGL.raw.GL.VERSION.GL_1_0 import glTranslatef, glRotatef, glEnable, GL_DEPTH_TEST, \
#     glClear, \
#     GL_COLOR_BUFFER_BIT, GL_DEPTH_BUFFER_BIT, GL_TRIANGLES, glBegin, glEnd, glColor3fv,
#     glVertex3fv
# from OpenGL.raw.GLU import gluPerspective
from OpenGL.GL import glTranslatef, glRotatef, glEnable, glClear, glBegin, glColor3fv, \
    glVertex3fv, glEnd
from OpenGL.GL import GL_DEPTH_TEST, GL_COLOR_BUFFER_BIT, GL_DEPTH_BUFFER_BIT, GL_TRIANGLES
from OpenGL.GLU import gluPerspective
import cupy as cp
from pygame import DOUBLEBUF, OPENGL

from biobots3D.memory.gpusimdata import SimulationDataGPU
from utils.tools import pyout


class GraphicsEngine3D:
    LIGHTSOURCE = cp.array([1., 2., 0.])

    def __init__(self, gpu: SimulationDataGPU, mode: str = "display",
                 fname="res/videos/default.mp4"):
        os.environ['SDL_VIDEO_WINDOW_POS'] = "0,1080"

        self.LIGHTSOURCE /= cp.linalg.norm(self.LIGHTSOURCE)

        self.gpu = gpu
        self.mode = mode

        pygame.init()
        # display = (1920, 1080)
        display = (800, 600)
        pygame.display.set_mode(display, DOUBLEBUF | OPENGL)
        gluPerspective(45, (1920 / 1080), 0.1, 50.)
        glTranslatef(0., 0, -15)
        glRotatef(30, 1, 0, 0)
        glEnable(GL_DEPTH_TEST)

        if mode == 'save':
            self.video_writer = cv2.VideoWriter(f"res/videos/{fname}.mp4",
                                                cv2.VideoWriter_fourcc(*'MP4V'), 60, display)
            atexit.register(self.video_writer.release)

    def render_opengl(self):
        self.__controls()
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glRotatef(1, 0, 1, 0)

        v0 = self.gpu.N_pos[self.gpu.S_n_1] - self.gpu.N_pos[self.gpu.S_n_0]
        v1 = self.gpu.N_pos[self.gpu.S_n_2] - self.gpu.N_pos[self.gpu.S_n_0]
        normal = cp.cross(v0, v1)
        normal /= cp.linalg.norm(normal, axis=1, keepdims=True)
        face_value = cp.dot(normal, self.LIGHTSOURCE)
        face_value = (0.4 + 0.3 * (1 + face_value)).get()

        v0 = (self.gpu.N_pos[self.gpu.S_n_0]).get().astype(np.float32).tolist()
        v1 = (self.gpu.N_pos[self.gpu.S_n_1]).get().astype(np.float32).tolist()
        v2 = (self.gpu.N_pos[self.gpu.S_n_2]).get().astype(np.float32).tolist()

        glBegin(GL_TRIANGLES)
        for sii in range(self.gpu.S_ids.size):
            c = colorsys.hsv_to_rgb(0.33, 0.5, face_value[sii])
            glColor3fv(c)
            for v in [v0[sii], v1[sii], v2[sii]]:
                glVertex3fv(tuple(float(x) for x in v))
        glEnd()

        pygame.display.flip()
        pygame.time.wait(10)

        if self.mode == 'display':
            pygame.display.flip()
            pygame.time.wait(10)
        elif self.mode == 'save':


            view = pygame.surfarray.array3d(pygame.display.get_surface())
            view = view.transpose([1, 0, 2])
            img_bgr = cv2.cvtColor(view, cv2.COLOR_RGB2BGR)

            cv2.imshow("debug", img_bgr)
            cv2.waitKey(-1)

            self.video_writer.write(img_bgr)



    def __controls(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
