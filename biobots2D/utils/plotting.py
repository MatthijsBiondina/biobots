import atexit
import sys
import time
from typing import List

import cupy as cp
import cv2
import numba
import numpy as np
from numba import cuda, vectorize
from numba.cuda import grid

from biobots2D.components.cell.abstractcell import AbstractCell
from biobots2D.components.simulation.cuda_memory import CudaMemory, synchronize
from biobots2D.utils.tools import pyout
import pygame
from pygame.locals import *
from pygame import gfxdraw


class Renderer:
    BACKGROUND = (45, 41, 131)
    CELL_COLORS = [(0, 119, 59),
                   (8, 171, 154),
                   (116, 206, 235),
                   (226, 202, 129),
                   (215, 98, 119),
                   (179, 68, 149),
                   (145, 30, 83)]

    def __init__(self, gpu: CudaMemory, figsize=(1080, 720)):
        self.gpu = gpu
        self.figsize = figsize

        pygame.init()

        self.fps = 60
        self.clock = pygame.time.Clock()

        self.display = pygame.display.set_mode(figsize)
        self.display.fill(self.BACKGROUND)
        pygame.display.set_caption("BioBots")

        self.ctypes = cp.asnumpy(self.gpu.C_type).astype(int)

        self.xmin, self.xmax, self.ymin, self.ymax = None, None, None, None

        self.videowriter = cv2.VideoWriter('res/gradient_following_biobot.mp4',
                                           cv2.VideoWriter_fourcc(*'MP4V'), 180, figsize)

        atexit.register(self.cleanup)

    def render(self):
        if not self.gpu.RENDER:
            return
        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit()
                sys.exit()

        basetone = (38,129,174)
        basetone = (200,200,200)
        # basetone = (89,75,109)
        pastel = lambda x: [int(0.75 * x[ii] + 0.25 * basetone[ii]) for ii in range(3)]
        dark = lambda x: [x[ii]//2 for ii in range(3)]

        self.display.fill(pastel(self.BACKGROUND))

        N = cp.asnumpy(self.gpu.N_pos[self.gpu.C_node_idxs])
        xmin, xmax, ymin, ymax = self.get_bounds(N)
        N[:, :, 0] = (N[:, :, 0] - xmin) / (xmax - xmin) * self.figsize[0]
        N[:, :, 1] = (N[:, :, 1] - ymin) / (ymax - ymin) * self.figsize[1]
        N = N.astype(np.int)

        for ii in range(N.shape[0]):
            color = self.CELL_COLORS[self.ctypes[ii]]

            color = pastel(color)

            points = N[ii].tolist()
            points = [[x[0], self.figsize[1]-x[1]] for x in points]
            pygame.draw.polygon(self.display, color, points)
            pygame.draw.polygon(self.display, dark(color), points, 5)

        pygame.display.update()
        self.clock.tick(self.fps)

        view = pygame.surfarray.array3d(self.display)
        view = view.transpose([1,0,2])
        img_bgr = cv2.cvtColor(view, cv2.COLOR_BGR2RGB)

        self.videowriter.write(img_bgr)

        # cv2.imshow("foo", img_bgr)
        # cv2.waitKey(-1)


        # pyout()

    def get_bounds(self, N):
        xmin_, xmax_ = np.min(N[:, :, 0]), np.max(N[:, :, 0])
        ymin_, ymax_ = np.min(N[:, :, 1]), np.max(N[:, :, 1])

        scale = max((xmax_ - xmin_) / self.figsize[0], (ymax_ - ymin_) / self.figsize[1])
        xmin = (xmax_ - 0.5 * (xmax_ - xmin_)) - 0.5 * self.figsize[0] * scale
        xmax = (xmax_ - 0.5 * (xmax_ - xmin_)) + 0.5 * self.figsize[0] * scale
        ymin = (ymax_ - 0.5 * (ymax_ - ymin_)) - 0.5 * self.figsize[1] * scale
        ymax = (ymax_ - 0.5 * (ymax_ - ymin_)) + 0.5 * self.figsize[1] * scale

        self.xmin = xmin if self.xmin is None else min(self.xmin, xmin)
        self.xmax = xmax if self.xmax is None else max(self.xmax, xmax)
        self.ymin = ymin if self.ymin is None else min(self.ymin, ymin)
        self.ymax = ymax if self.ymax is None else max(self.ymax, ymax)


        return self.xmin, self.xmax, self.ymin, self.ymax

    def cleanup(self):
        self.videowriter.release()