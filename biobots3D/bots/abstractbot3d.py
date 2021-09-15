from abc import ABC, abstractmethod
from typing import Tuple, Dict

import numpy as np

from biobots3D.cells.abstractcell3d import AbstractCell3D
from utils.errors import TodoException
from utils.tools import pyout


class AbstractBot3D(ABC):
    def __init__(self):
        self.id = None
        self.cells: Dict[Tuple[int]:AbstractCell3D] = {}

        self.construct()
        self.give_cells_unique_ids()
        self.connect_cells()

    @abstractmethod
    def construct(self, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def connect_cells(self, **kwargs):
        raise NotImplementedError

    @property
    def pos(self):
        return np.mean(np.array([c.pos for _, c in self.cells.items()]), axis=0)

    def give_cells_unique_ids(self):
        """
        make sure every cell has a different id and each node, edge, and surface in each cell
        has a different id
        :return:
        """
        next_cell_id = 0
        next_surf_id = 0
        next_edge_id = 0
        next_node_id = 0

        for pos, c in self.cells.items():
            # Cell ID
            c.id = next_cell_id
            next_cell_id += 1

            # Surfaces
            c.surf_ids += next_surf_id
            next_surf_id += c.surf_ids.shape[0]

            # Edges
            c.edge_ids += next_edge_id
            next_edge_id += c.edge_ids.shape[0]

            # Nodes
            c.node_ids += next_node_id
            next_node_id += c.node_ids.shape[0]

    def add_cell(self, cell: AbstractCell3D, pos: Tuple[int, int, int]):
        if len(pos) != 3:
            raise ValueError(f"pos should be 3D coordinates, not {pos}")
        if not (isinstance(xyz, int) for xyz in pos):
            raise TypeError(f"Coordinates should be (int, int, int), not "
                            f"{(type(xyz) for xyz in pos)}")
        if not (pos[0] % 2 == pos[1] % 2 == pos[2] % 2):
            raise ValueError(f"coordinates should all be even or odd to fit on grid, got {pos}")
        if pos in self.cells:
            raise IndexError(f"coordinates {pos} already occupied")

        self.cells[pos] = cell
