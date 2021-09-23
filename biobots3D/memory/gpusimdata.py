from typing import List

import cupy as cp
from numba import vectorize

from utils.tools import pyout


def cross_product(n0, n1, n2):
    """
    Compute the cross product of a surface defined by n0, n1, n2 in anti-clockwise order
    :param n0:
    :param n1:
    :param n2:
    :return:
    """
    return cp.cross(n1 - n0, n2 - n0)


def polyhedron_volume(origin, vertex, normal_vector, surface_area, C_2_S):
    origin_2_vertex = vertex - origin
    height = cp.sum(origin_2_vertex * normal_vector, axis=1)  # projection on normal
    surface_subdivision_volume = 1 / 3 * height * surface_area
    return cp.array([cp.sum(surface_subdivision_volume[c2s]) for c2s in C_2_S])


class SimulationDataGPU:
    def __init__(self):
        self.B_ids: cp.ndarray = cp.empty(0)  # (#bbots,)
        self.B_2_C: List[cp.ndarray] = []  # [(#bbots, #cells), ...]

        self.C_ids: cp.ndarray = cp.empty(0)  # (#cells,)
        self.C_v_0: cp.ndarray = cp.empty(0)  # (#cells,) target volume
        self.C_2_N: List[cp.ndarray] = []
        self.C_2_S: List[cp.ndarray] = []

        self.S_ids: cp.ndarray = cp.empty(0)
        self.S_n_0: cp.ndarray = cp.empty(0)
        self.S_n_1: cp.ndarray = cp.empty(0)
        self.S_n_2: cp.ndarray = cp.empty(0)
        self.S_2_C: cp.ndarray = cp.empty(0)

        self.E_ids: cp.ndarray = cp.empty(0)
        self.E_n_0: cp.ndarray = cp.empty(0)
        self.E_n_1: cp.ndarray = cp.empty(0)

        self.N_ids: cp.ndarray = cp.empty(0)
        self.N_pos: cp.ndarray = cp.empty(0)

        # dynamic variables
        self.N_for: cp.ndarray = cp.empty(0)
        self.S_area: cp.ndarray = cp.empty(0)  # Surface area
        self.S_cros: cp.ndarray = cp.empty(0)  # Surface Cross Product
        self.S_nrmv: cp.ndarray = cp.empty(0)  # Surface Normal Vector

        self.C_orig: cp.ndarray = cp.empty(0)
        self.C_vol: cp.ndarray = cp.empty(0)

        #
        self._neighbouring_surface_nodes = None

    def reset_dynamic_variables(self):
        self.N_for = cp.zeros_like(self.N_pos)
        self.S_cros = cross_product(self.N_pos[self.S_n_0], self.N_pos[self.S_n_1],
                                    self.N_pos[self.S_n_2])
        self.S_area = cp.linalg.norm(self.S_cros, axis=1) / 2
        self.S_nrmv = self.S_cros / cp.linalg.norm(self.S_cros, axis=1, keepdims=True)

        self.C_orig = cp.array([cp.mean(self.N_pos[c2n], axis=0) for c2n in self.C_2_N])
        self.C_vol = polyhedron_volume(self.C_orig[self.S_2_C], self.N_pos[self.S_n_0],
                                       self.S_nrmv, self.S_area, self.C_2_S)

    @property
    def neighbouring_surface_nodes_masks(self):
        if self._neighbouring_surface_nodes is not None:
            return self._neighbouring_surface_nodes

        M = cp.argwhere(self.S_n_0[None, :] == self.N_ids[:, None])
        M_ids = self.N_ids[M[:, 0]]
        M_n_1 = self.S_n_1[M[:, 1]]
        M_n_2 = self.S_n_2[M[:, 1]]

        M = cp.argwhere(self.S_n_1[None, :] == self.N_ids[:, None])
        M_ids = cp.concatenate((M_ids, self.N_ids[M[:, 0]]), axis=0)
        M_n_1 = cp.concatenate((M_n_1, self.S_n_2[M[:, 1]]), axis=0)
        M_n_2 = cp.concatenate((M_n_2, self.S_n_0[M[:, 1]]), axis=0)

        M = cp.argwhere(self.S_n_2[None, :] == self.N_ids[:, None])
        M_ids = cp.concatenate((M_ids, self.N_ids[M[:, 0]]), axis=0)
        M_n_1 = cp.concatenate((M_n_1, self.S_n_0[M[:, 1]]), axis=0)
        M_n_2 = cp.concatenate((M_n_2, self.S_n_1[M[:, 1]]), axis=0)

        index = cp.argsort(M_ids)
        M_ids = M_ids[index]
        M_n_1 = M_n_1[index]
        M_n_2 = M_n_2[index]

        count = cp.sum(M_ids[None, :] == M_ids[:, None], axis=1)

        M_ids_4 = M_ids[count == 4].reshape(-1, 4)[:, 0]
        M_n_1_4 = M_n_1[count == 4].reshape(-1, 4)
        M_n_2_4 = M_n_2[count == 4].reshape(-1, 4)

        M_ids_6 = M_ids[count == 6].reshape(-1, 6)[:, 0]
        M_n_1_6 = M_n_1[count == 6].reshape(-1, 6)
        M_n_2_6 = M_n_2[count == 6].reshape(-1, 6)

        self._neighbouring_surface_nodes = ((M_ids_4, M_n_1_4, M_n_2_4),
                                            (M_ids_6, M_n_1_6, M_n_2_6))

        return self._neighbouring_surface_nodes
