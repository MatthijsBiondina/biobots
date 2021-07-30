import sys
import time
from typing import List

import cupy as cp
import cv2
import numba
import numpy as np
from numba import cuda, vectorize
from numba.cuda import grid

from src.components.cell.abstractcell import AbstractCell
from src.components.simulation.cuda_memory import CudaMemory, synchronize
from src.utils.tools import pyout
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

    def render(self):
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
            pygame.draw.polygon(self.display, color, points)
            pygame.draw.polygon(self.display, dark(color), points, 5)
            # gfxdraw.aapolygon(self.display, points, color)
            # gfxdraw.filled_polygon(self.display, points, color)

        pygame.display.update()
        self.clock.tick(self.fps)

        # pyout()

    def get_bounds(self, N):
        xmin_, xmax_ = np.min(N[:, :, 0]), np.max(N[:, :, 0])
        ymin_, ymax_ = np.min(N[:, :, 1]), np.max(N[:, :, 1])

        scale = max((xmax_ - xmin_) / self.figsize[0], (ymax_ - ymin_) / self.figsize[1])
        xmin = (xmax_ - 0.5 * (xmax_ - xmin_)) - 0.5 * self.figsize[0] * scale
        xmax = (xmax_ - 0.5 * (xmax_ - xmin_)) + 0.5 * self.figsize[0] * scale
        ymin = (ymax_ - 0.5 * (ymax_ - ymin_)) - 0.5 * self.figsize[1] * scale
        ymax = (ymax_ - 0.5 * (ymax_ - ymin_)) + 0.5 * self.figsize[1] * scale

        return xmin, xmax, ymin, ymax
