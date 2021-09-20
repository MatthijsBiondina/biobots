from abc import ABC, abstractmethod
from typing import Tuple, Dict, List

import numpy as np
from numpy import int64

from biobots3D.cells.abstractcell3d import AbstractCell3D
from biobots3D.memory.cpubotdata import BBotDataCPU
from utils.errors import TodoException
from utils.tools import pyout


class AbstractBot3D(ABC):

    def __init__(self):
        self.id = None
        self.cells: Dict[Tuple[int]:AbstractCell3D] = {}

        self.construct()
        self.__give_components_unique_ids()
        self.connect_cells()
        self.__rename_merged_vertices()

        self.cpu = BBotDataCPU()
        self.__load_central_memory()

    @abstractmethod
    def construct(self, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def connect_cells(self, **kwargs):
        raise NotImplementedError

    # @property
    # def pos(self):
    #     """
    #     Two methods depending on whether central cpu data has been gathered
    #
    #     Note that methods will return slightly different values, since merged nodes are counted
    #     twice in the first method
    #     :return:
    #     """
    #     try:
    #         return np.mean(np.array([c.pos for _, c in self.cells.items()]), axis=0)
    #     except AttributeError:
    #         return np.mean(self.cpu.node_pos, axis=0)

    def __give_components_unique_ids(self):
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
            c.edge_n_0 += next_node_id
            c.edge_n_1 += next_node_id
            c.surf_n_0 += next_node_id
            c.surf_n_1 += next_node_id
            c.surf_n_2 += next_node_id
            next_node_id += c.node_ids.shape[0]

    def __rename_merged_vertices(self):
        new_ids = {}
        next_id = 0
        for _, c in self.cells.items():
            for id_ in c.node_ids:
                try:
                    new_ids[id_]
                except KeyError:
                    new_ids[id_] = next_id
                    next_id += 1
        for _, c in self.cells.items():
            for key, val in new_ids.items():
                c.node_ids[c.node_ids == key] = val
                c.edge_n_0[c.edge_n_0 == key] = val
                c.edge_n_1[c.edge_n_1 == key] = val
                c.surf_n_0[c.surf_n_0 == key] = val
                c.surf_n_1[c.surf_n_1 == key] = val
                c.surf_n_2[c.surf_n_2 == key] = val

    def add_cell(self, cell: AbstractCell3D):
        pos = cell.pos
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

    def __load_central_memory(self):
        D = self.cpu

        nr_of_cells = len(self.cells)
        nr_of_nodes = max([np.max(c.node_ids) for _, c in self.cells.items()]) + 1
        nr_of_edges = max([np.max(c.edge_ids) for _, c in self.cells.items()]) + 1
        nr_of_surfs = max([np.max(c.surf_ids) for _, c in self.cells.items()]) + 1

        D.cell_ids = np.arange(nr_of_cells)

        D.node_ids = np.arange(nr_of_nodes)
        D.cell_2_N = [np.empty(0)] * nr_of_cells
        D.cell_2_S = [np.empty(0)] * nr_of_cells
        D.node_pos = np.empty((nr_of_nodes, 3))
        D.edge_ids = np.arange(nr_of_edges)
        D.edge_n_0 = np.empty(nr_of_edges, dtype=int64)
        D.edge_n_1 = np.empty(nr_of_edges, dtype=int64)
        D.surf_ids = np.arange(nr_of_surfs)
        D.surf_n_0 = np.empty(nr_of_surfs, dtype=int64)
        D.surf_n_1 = np.empty(nr_of_surfs, dtype=int64)
        D.surf_n_2 = np.empty(nr_of_surfs, dtype=int64)

        del nr_of_nodes, nr_of_cells, nr_of_edges, nr_of_surfs

        for c_ii, (_, c) in enumerate(self.cells.items()):
            for n_ii, id_ in enumerate(c.node_ids):
                D.node_pos[D.node_ids == id_] = c.node_pos[n_ii]
                try:
                    D.cell_2_N[c_ii][n_ii] = id_
                except IndexError:
                    D.cell_2_N[c_ii] = np.full(c.node_ids.size, id_)
            del n_ii

            for e_ii, id_ in enumerate(c.edge_ids):
                central_ii = np.argwhere(D.edge_ids == id_).squeeze(0)
                D.edge_n_0[central_ii] = c.edge_n_0[e_ii]
                D.edge_n_1[central_ii] = c.edge_n_1[e_ii]
            del e_ii

            for s_ii, id_ in enumerate(c.surf_ids):
                central_ii = np.argwhere(D.surf_ids == id_).squeeze(0)
                D.surf_n_0[central_ii] = c.surf_n_0[s_ii]
                D.surf_n_1[central_ii] = c.surf_n_1[s_ii]
                D.surf_n_2[central_ii] = c.surf_n_2[s_ii]

                try:
                    D.cell_2_S[c_ii][s_ii] = id_
                except IndexError:
                    D.cell_2_S[c_ii] = np.full(c.surf_ids.size, id_)

        del c_ii

        del self.cells
