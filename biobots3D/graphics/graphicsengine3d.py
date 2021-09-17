import atexit
import colorsys
import os
import sys

import cv2
import numpy as np
import pygame
# from OpenGL.raw.GL.VERSION.GL_1_0 import glTranslatef, glRotatef, glEnable, GL_DEPTH_TEST, \
#     glClear, \
#     GL_COLOR_BUFFER_BIT, GL_DEPTH_BUFFER_BIT, GL_TRIANGLES, glBegin, glEnd, glColor3fv,
#     glVertex3fv
# from OpenGL.raw.GLU import gluPerspective
from OpenGL.GL import glTranslatef, glRotatef, glEnable, glClear, glBegin, glColor3fv, \
    glVertex3fv, glEnd, glReadPixels
from OpenGL.GL import GL_DEPTH_TEST, GL_COLOR_BUFFER_BIT, GL_DEPTH_BUFFER_BIT, GL_TRIANGLES
from OpenGL.GLU import gluPerspective
import cupy as cp
from OpenGL.GL import GL_RGB, GL_UNSIGNED_BYTE
from pygame import DOUBLEBUF, OPENGL

from biobots3D.memory.gpusimdata import SimulationDataGPU
from utils.tools import pyout


class GraphicsEngine3D:
    LIGHTSOURCE = cp.array([1., 2., 0.])

    def __init__(self, gpu: SimulationDataGPU, mode: str = "display",
                 fname="res/videos/default.mp4"):

        self.__init_display(fname)

        # OpenGL initialization
        gluPerspective(45, (self.display[0] / self.display[1]), 0.1, 50.)
        glTranslatef(0., 0, -15)
        glRotatef(30, 1, 0, 0)
        glEnable(GL_DEPTH_TEST)

        self.LIGHTSOURCE /= cp.linalg.norm(self.LIGHTSOURCE)

        self.gpu = gpu
        self.mode = mode

    def __init_display(self, fname):

        if sys.gettrace() is None:  # check for debug context
            import glfw  # glfw does not work in pycharm debugging context

            self.display = (1920, 1000)

            if not glfw.init():
                raise RuntimeError("glfw3 failed initialization")
            glfw.window_hint(glfw.VISIBLE, False)
            window = glfw.create_window(self.display[0], self.display[1], "hidden window", None,
                                        None)
            if not window:
                glfw.terminate()
                raise RuntimeError("glfw3 window failed creation")
            glfw.make_context_current(window)

            self.video_writer = cv2.VideoWriter(f"res/videos/{fname}.mp4",
                                                cv2.VideoWriter_fourcc(*'MP4V'), 60, self.display)
            atexit.register(self.video_writer.release)
            atexit.register(glfw.terminate)

            self.windowmode = "glfw"
        else:
            pygame.init()
            self.display = (800, 600)
            pygame.display.set_mode(self.display, DOUBLEBUF | OPENGL)
            self.windowmode = "pygame"

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

        img = np.empty(0)
        if self.windowmode == "glfw":
            img_buffer = glReadPixels(0, 0, *self.display, GL_RGB, GL_UNSIGNED_BYTE)
            img = np.frombuffer(img_buffer, dtype=np.uint8).reshape(*self.display[::-1], 3)
            img = img[::-1, :, :]

        if self.mode == 'display':
            if self.windowmode == "pygame":
                pygame.display.flip()
                pygame.time.wait(10)
            else:
                cv2.imshow("BioBots 3D", img)
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    sys.exit(0)
        elif self.mode == 'save':
            self.video_writer.write(img)

    def __controls(self):
        if self.windowmode == "pygame":
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    quit()
