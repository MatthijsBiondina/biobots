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
from src.utils.tools import pyout


class Renderer:
    def __init__(self, figsize=(480, 360)):
        self.figsize = figsize
        self.canvas = cuda.to_device(np.full(figsize[::-1] + (3,), 1.))

        self.n_cells = None
        self.N = None

    def render(self, cell_list: List[AbstractCell]):
        if self.n_cells is None or len(cell_list) != self.n_cells:
            self.n_cells = len(cell_list)
            self.N = cuda.to_device(np.array([len(c.node_list) for c in cell_list]))


        P = np.zeros((len(cell_list), max(self.N), 2))
        for ii, A in enumerate(
                [np.stack([np.array(n.position.numpy()) for n in c.node_list]) for c in cell_list]):
            P[ii] = A
        pyout(f"{time.time() - t0:.4f} ms")

        sys.exit(0)
        return

    def _get_inputs(self, P):
        xmin, xmax = np.min(P[:, :, 0]), np.max(P[:, :, 0])
        ymin, ymax = np.min(P[:, :, 1]), np.max(P[:, :, 1])
        scale = max((xmax - xmin) / self.figsize[0], (ymax - ymin) / self.figsize[1])
        xmin_ = (xmax - 0.5 * (xmax - xmin)) - 0.5 * self.figsize[0] * scale
        xmax_ = (xmax - 0.5 * (xmax - xmin)) + 0.5 * self.figsize[0] * scale
        ymin_ = (ymax - 0.5 * (ymax - ymin)) - 0.5 * self.figsize[1] * scale
        ymax_ = (ymax - 0.5 * (ymax - ymin)) + 0.5 * self.figsize[1] * scale

        X = cuda.to_device(np.linspace(xmin_, xmax_, self.figsize[0]))
        Y = cuda.to_device(np.linspace(ymin_, ymax_, self.figsize[1]))

        return X, Y




def render(cell_list: List[AbstractCell], figsize=(480, 360)):
    N = np.array([len(c.node_list) for c in cell_list])
    P = cp.zeros((len(cell_list), max(N), 2))
    for ii, A in enumerate(
            [cp.stack([cp.array(n.position.numpy()) for n in c.node_list]) for c in cell_list]):
        P[ii] = A
    C = cp.stack([cp.array(c.get_colour().numpy()) for c in cell_list], axis=0)

    xmin, xmax = cp.min(P[:, :, 0]), cp.max(P[:, :, 0])
    ymin, ymax = cp.min(P[:, :, 1]), cp.max(P[:, :, 1])
    scale = max((xmax - xmin) / figsize[0], (ymax - ymin) / figsize[1])
    xmin_ = (xmax - 0.5 * (xmax - xmin)) - 0.5 * figsize[0] * scale
    xmax_ = (xmax - 0.5 * (xmax - xmin)) + 0.5 * figsize[0] * scale
    ymin_ = (ymax - 0.5 * (ymax - ymin)) - 0.5 * figsize[1] * scale
    ymax_ = (ymax - 0.5 * (ymax - ymin)) + 0.5 * figsize[1] * scale
    xmin, xmax, ymin, ymax = xmin_, xmax_, ymin_, ymax_
    del xmin_, xmax_, ymin_, ymax_

    X = cp.linspace(xmin, xmax, figsize[0])
    Y = cp.linspace(ymin, ymax, figsize[1])
    ou = cp.full(figsize[::-1] + (3,), 1.)
    N = cp.array(N)

    for _ in range(10):
        X = cuda.to_device(X)
        Y = cuda.to_device(Y)
        P = cuda.to_device(P)
        N = cuda.to_device(N)
        C = cuda.to_device(C)
        ou = cuda.to_device(cp.full(figsize[::-1] + (3,), 1.))

        block_size = 64
        grid_size = (figsize[0] * figsize[1]) // block_size + 1
        cuda.synchronize()

        t0 = time.time()
        raytrace[grid_size, block_size](X, Y, P, N, C, ou)
        cuda.synchronize()
        pyout(time.time() - t0)

    # cv2.imshow("Foo", ou.copy_to_host())
    # cv2.waitKey(-1)

    return ou.copy_to_host()[:, :, ::-1]


@cuda.jit
def raytrace(X, Y, P, N, C, ou):
    """
    See https://www.geeksforgeeks.org/how-to-check-if-a-given-point-lies-inside-a-polygon/ and
    https://stackoverflow.com/questions/36399381/whats-the-fastest-way-of-checking-if-a-point-is-inside-a-polygon-in-python
    """
    ii_start, jj_start, kk_start = cuda.grid(3)
    ii_stride, jj_stride, kk_stride = cuda.gridsize(3)
    for kk in range(kk_start, P.shape[0], kk_stride):
        # dev
        # find middle of polygon
        p = P[kk, :N[kk]]
        c = C[kk]

        x_poly_mu, y_poly_mu = 0., 0.
        for i in range(N[kk]):
            x_poly_mu += p[i][0]
            y_poly_mu += p[i][1]
        x_poly_mu /= N[kk]
        y_poly_mu /= N[kk]

        # compute max distance
        poly_radius = 0.
        for i in range(N[kk]):
            poly_radius = max(poly_radius, abs(p[i][0] - x_poly_mu) + abs(p[i][1] - y_poly_mu))
        # dev

        for ii in range(ii_start, X.shape[0], ii_stride):
            for jj in range(jj_start, Y.shape[0], jj_stride):

                x = X[ii]
                y = Y[jj]

                if abs(x - x_poly_mu) + abs(y - y_poly_mu) <= poly_radius:

                    inside, on_edge = False, False
                    xints = 0.
                    p1x, p1y = p[N[kk] - 1]
                    for i in range(N[kk]):
                        p2x, p2y = p[i]
                        if min(p1y, p2y) < y <= max(p1y, p2y) and x <= max(p1x, p2x):
                            if p1y != p2y:
                                xints = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                            if p1x == p2x or x <= xints:
                                inside = not inside
                            if x == xints:
                                on_edge = True
                        p1x, p1y = p2x, p2y
                    if inside or on_edge:
                        for channel in range(3):
                            ou[jj, ii, channel] = c[channel]
