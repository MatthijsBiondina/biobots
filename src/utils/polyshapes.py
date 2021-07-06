from abc import ABC
from typing import Union

import torch
from numpy import pi
from torch import cos, sin, Tensor

from src.utils.errors import TodoException


class Polyshape:
    def __init__(self, x=None, y=None, P=None, X=None, Y=None):
        self.vertices, self._num_regions, self._num_holes = (None,) * 3

        if x is None and y is None and P is None and X is None and Y is None:
            # create an empty polyshape object
            raise TodoException
        elif x is not None and y is not None and P is None and X is None and Y is None:
            # create a polyshape from 2-D vertices defined by a vector of x-coordinates and a
            # vector of corresponding y-coordintates. x and y must be the same length with at
            # least three elements
            self.vertices = torch.stack((x, y), dim=1)
            # todo: debug remove rounding
            self.vertices[torch.abs(self.vertices) < 0.00001] = 0.
            self._num_regions = 1
            self._num_holes = 0
        elif x is None and y is None and P is not None and X is None and Y is None:
            # creates a polyshape from the 2-D vertices defined in the N-by-2 matrix P,
            # where N is the number of vertices. The first column of P defines the x-coordinates,
            # and the second column defines the y-coordinates
            raise TodoException
        elif x is None and y is None and P is None and X is not None and Y is not None:
            # where X and Y are 1-by-M lists of vectors for the x- and y-coordinates, creates a
            # polygon consisting of M boundaries. Each vector in X must have the same length as
            # the corresponding vector in Y, but the number of vertices can vary between boundaries.
            raise TodoException
        else:
            raise ValueError("Bad arguments: either no arguments, x and y, P, or X and Y "
                             "should be specified")

    @property
    def num_regions(self):
        return self._num_regions

    @property
    def num_holes(self):
        return self._num_holes


def nsidedpoly(N: int, mode: Union[None, str] = None, arg=1):
    """
    Creates a polyshape object that is a regular polygon with n sides of equal length.
    :param N: integer greaterthan or equal to 3
    :param args:
    :return:
    """
    if mode == 'radius':
        x = cos(torch.arange(0, 2 * pi, 2 * pi / N)) * arg
        y = sin(torch.arange(0, 2 * pi, 2 * pi / N,)) * arg
        return Polyshape(x=x, y=y)
    elif mode == 'centre':
        raise TodoException
    else:
        raise ValueError("Choose one: 'radius', 'centre'")


def polyarea(x: Tensor, y: Tensor):
    return 0.5 * torch.abs(torch.dot(x, y.roll(1)) - torch.dot(y, x.roll(1)))
