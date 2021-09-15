import random
from abc import ABC
from typing import List, Tuple

import cupy as cp
import numpy as np
from numpy import pi, sin, cos
import quaternion

from biobots3D.bots.abstractbot3d import AbstractBot3D
from utils.errors import TodoException
from utils.tools import pyout


class AbstractSimulation3D(ABC):
    def __init__(self):
        super(AbstractSimulation3D, self).__init__()
        self.seed = None
        self.bots: List[AbstractBot3D] = []

        self.next_bbot_id = 0
        self.next_cell_id = 0
        self.next_surf_id = 0
        self.next_edge_id = 0
        self.next_node_id = 0

    def set_seed(self, seed):
        """
        Set rng seed for all sources of randomness

        :param seed:
        :return:
        """
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)
        cp.random.seed(seed)

    def add_biobot(self,
                   bot: AbstractBot3D,
                   pos: np.ndarray = np.array([0, 0, 0]),
                   R: np.quaternion = np.quaternion(1., 0., 0., 0.)
                   ):
        bot.id = self.next_bbot_id
        self.next_bbot_id += 1

        for _, c in bot.cells.items():
            c.id += self.next_cell_id
            self.next_cell_id += 1

            c.surf_ids += self.next_surf_id
            self.next_surf_id += c.surf_ids.shape[0]

            c.edge_ids += self.next_edge_id
            self.next_edge_id += c.edge_ids.shape[0]

            c.node_ids += self.next_node_id
            self.next_node_id += c.node_ids.shape[0]

        # Translate and rotate
        if R is None:  # generate random quaternion
            ang = np.random.uniform(-pi, pi)
            vec = np.random.uniform(-1, 1, 3)
            vec /= np.linalg.norm(vec)
            R = np.quaternion(cos(ang), sin(ang) * vec[0], sin(ang) * vec[1], sin(ang) * vec[2])

        center = bot.pos
        for _, c in bot.cells.items():
            c.node_pos -= center
            c.node_pos = quaternion.rotate_vectors(R, c.node_pos)
            c.node_pos += pos

        self.bots.append(bot)
