import os
from abc import ABC

import numpy as np
from numpy import pi, sqrt
# from pyquaternion import Quaternion
import quaternion

from utils.errors import TodoException
import utils.tetrakaidecahedron as tetrakaidecahedron
from utils.tools import pyout


class AbstractCell3D(ABC):
    def __init__(self, pos=(0, 0, 0), face=(0, 1, 0)):
        """
        Cell constructor builds default tetrakaidecahedron at specified location

        :param pos: gridpos
        :param face:
        """

        super(AbstractCell3D, self).__init__()
        self.pos = pos
        self.id = None

        self.node_pos = np.copy(tetrakaidecahedron.nodes)
        if True:  # pos[0] % 2 == 0:  # on even gridpoint rotate pi/4
            R = np.quaternion(np.cos(pi / 8), 0, np.sin(pi / 8), 0)
            self.node_pos = quaternion.rotate_vectors(R, self.node_pos)

        self.node_pos += np.array(pos) * sqrt(2)
        self.node_ids = np.arange(self.node_pos.shape[0])

        self.edge_n_0 = np.copy(tetrakaidecahedron.edges)[:, 0]
        self.edge_n_1 = np.copy(tetrakaidecahedron.edges)[:, 1]
        self.edge_ids = np.arange(self.edge_n_0.shape[0])

        self.surf_n_0 = np.copy(tetrakaidecahedron.surfaces)[:, 0]
        self.surf_n_1 = np.copy(tetrakaidecahedron.surfaces)[:, 1]
        self.surf_n_2 = np.copy(tetrakaidecahedron.surfaces)[:, 2]
        self.surf_ids = np.arange(self.surf_n_0.shape[0])

        self.target_volume: float = 0
