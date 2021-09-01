from typing import List, Union

import numpy as np
from numpy import pi

from biobots2D.components.cell.abstractcell import AbstractCell
from biobots2D.components.cell.element import Element
from biobots2D.components.node.node import Node
from biobots2D.components.simulation.cuda_memory import CudaMemory
from biobots2D.utils.tools import pyout


def dot(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    return np.einsum("...i,...i->...", A, B)


class CPUMemory:
    EXEC_CPU = False
    RENDER = False

    def __init__(self,
                 cell_list: List[AbstractCell],
                 element_list: List[Element],
                 node_list: List[Node],
                 d_limit: float,
                 gpu: CudaMemory
                 ):
        raise TypeError


        # hyperparameters
        self.d_limit = np.float32(d_limit)

        # Cell data
        self.C_node_ids = np.array([[n.id for n in c.node_list] for c in cell_list])
        self.C_node_idxs = self.__make_c_node_idxs(cell_list, node_list)
        self.C_element_idxs = np.array([[e.id for e in c.element_list] for c in cell_list])
        self.C_age = np.array([c.age for c in cell_list], dtype=np.float)
        self.C_type = np.array([c.cell_type for c in cell_list])
        self.C_grown_cell_target_area = np.array([c.grown_cell_target_area for c in cell_list])
        self.C_inhibitory = np.array([c.inhibitory for c in cell_list])

        # Cell type indices
        self.Ctype_0 = np.argwhere(self.C_type == 0).squeeze(1)
        self.Ctype_1 = np.argwhere(self.C_type == 1).squeeze(1)
        self.Ctype_2 = np.argwhere(self.C_type == 2).squeeze(1)
        self.Ctype_3 = np.argwhere(self.C_type == 3).squeeze(1)
        self.Ctype_4 = np.argwhere(self.C_type == 4).squeeze(1)

        # Node data
        self.N_id = np.array([n.id for n in node_list])
        self.N_id2idx = self.__make_id2idx()
        self.N_pos = np.array([n.position.numpy() for n in node_list])
        self.N_for = np.array([n.force.numpy() for n in node_list])
        self.N_pos_previous = np.copy(self.N_pos)
        self.N_for_previous = np.copy(self.N_for)
        self.N_eta = np.array([n.eta for n in node_list], np.float32)

        # Element data
        self.E_node_1_id = np.array([e.node_1.id for e in element_list])
        self.E_node_2_id = np.array([e.node_2.id for e in element_list])
        self.E_cell_idx = np.array([e.cell_list[0].id for e in element_list])
        self.E_node_1 = self.N_id2idx[self.E_node_1_id]
        self.E_node_2 = self.N_id2idx[self.E_node_2_id]
        self.E_internal = np.array([e.internal for e in element_list])
        self.E_cilia_direction = self.__determine_cilia_direction(element_list)

        # Broadcast matrices
        self.cell2node = self.__make_cell2node_matrix(cell_list)
        self.cell2element = self.__make_cell2element_matrix(cell_list)
        self.element2node = self.__make_element2node_matrix(element_list)
        self.rotate_clockwise_2d = np.array([[0., -1.], [1., 0.]], dtype=np.float32)

        # Masks
        self.node2element_mask = self.__make_node2element_mask(element_list)
        self.cell2element_mask = self.cell2element.astype(np.bool)

        # Dynamic memory
        self._vector_1_to_2 = None
        self._outward_normal = None
        self._element_length = None
        self._C_area = None
        self._C_perimeter = None
        self._C_target_area = None
        self._C_target_perimeter = None
        self._C_pos = None
        self._polygons = None
        self._E_length = None
        self._candidates = None

        self._t = 0.
        self.spice = 0.

    def clear_dynamic_memory(self, t):
        self._vector_1_to_2 = None
        self._outward_normal = None
        self._element_length = None
        self._C_area = None
        self._C_perimeter = None
        self._C_target_area = None
        self._C_target_perimeter = None
        self._C_pos = None
        self._polygons = None
        self._E_length = None
        self._candidates = None
        self._t = t

    @property
    def vector_1_to_2(self) -> np.ndarray:
        if self._vector_1_to_2 is None:
            direction_1_to_2 = self.N_pos[self.E_node_2] - self.N_pos[self.E_node_1]
            direction_1_to_2 /= np.linalg.norm(direction_1_to_2, axis=1)[:, None]
            self._vector_1_to_2 = direction_1_to_2
        return self._vector_1_to_2

    @property
    def outward_normal(self) -> np.ndarray:
        if self._outward_normal is None:
            self._outward_normal = self.vector_1_to_2 @ self.rotate_clockwise_2d
        return self._outward_normal

    @property
    def element_length(self) -> np.ndarray:
        if self._element_length is None:
            length = (self.N_pos[self.E_node_1] - self.N_pos[self.E_node_2]) ** 2
            length = np.sum(length, axis=1) ** .5
            self._element_length = length
        return self._element_length

    @property
    def C_area(self) -> np.ndarray:
        if self._C_area is None:
            x, y = self.polygons[:, :, 0], self.polygons[:, :, 1]
            self._C_area = 0.5 * np.abs(dot(x, np.roll(y, 1, axis=1))
                                        - dot(y, np.roll(x, 1, axis=1)))
        return self._C_area

    @property
    def C_target_area(self) -> np.ndarray:
        if self._C_target_area is None:
            self._C_target_area = np.empty_like(self._C_area)

            # Type 0 has constant target area
            self._C_target_area[:] = self.C_grown_cell_target_area[:]

            # Type 1 grows and shrinks as a polygon

            diff = 1.5

            self._C_target_area[self.Ctype_1] = \
                (1. + 0.5 * diff + 0.5 * diff * -np.sin(self._t * 5)) * \
                self.C_grown_cell_target_area[
                    self.Ctype_1]

            # Type 2 has constant target area
            # self._C_target_area[self.Ctype_2] = self.C_grown_cell_target_area[self.Ctype_2]
            # self._C_target_area[self.Ctype_3] = self.C_grown_cell_target_area[self.Ctype_3]

        return self._C_target_area

    @property
    def C_perimeter(self) -> np.ndarray:
        if self._C_perimeter is None:
            self._C_perimeter = np.sum(self.E_length[self.C_element_idxs], axis=1)
        return self._C_perimeter

    @property
    def C_target_perimeter(self) -> np.ndarray:
        if self._C_target_perimeter is None:
            self._C_target_perimeter = np.empty_like(self.C_target_area)

            # Type 0 wants to be a regular polygon
            n = self.C_element_idxs.shape[1]
            self._C_target_perimeter = \
                (4 * self.C_target_area * n * np.tan(pi / n)) ** .5
            self._C_target_perimeter[self.Ctype_1] = \
                (4 * self.C_target_area[self.Ctype_1] * n * np.tan(pi / n)) ** .5
        return self._C_target_perimeter

    @property
    def polygons(self):
        if self._polygons is None:
            self._polygons = self.N_pos[self.C_node_idxs]
        return self._polygons

    @property
    def E_length(self):
        if self._E_length is None:
            n1 = self.N_pos[self.E_node_1]
            n2 = self.N_pos[self.E_node_2]
            self._E_length = np.sum((n1 - n2) ** 2, axis=1) ** .5
        return self._E_length

    @property
    def candidates(self):
        if self._candidates is None:
            candidates = np.ones((self.N_pos.shape[0], self.E_node_1.shape[0]), dtype=np.bool)

            # find nodes in element interaction regions
            pnt = self.N_pos
            start = self.N_pos[self.E_node_1]
            end = self.N_pos[self.E_node_2]
            line_vec = end - start
            pnt_vec = pnt[:, None] - start[None, :]
            line_len = np.linalg.norm(line_vec, axis=1)
            line_unitvec = line_vec / line_len[:, None]
            pnt_vec_scaled = pnt_vec / line_len[None, :, None]
            t = np.sum(line_unitvec[None, :, :] * pnt_vec_scaled, axis=2)
            nearest = line_vec[None, :, :] * t[:, :, None]
            dist = np.linalg.norm(nearest - pnt_vec, axis=2)
            candidates = candidates & (0. <= t) & (t <= 1.) & (dist < self.d_limit)

            candidates = candidates & ~ self.node2element_mask
            self._candidates = candidates
        return self._candidates

    @property
    def C_pos(self):
        if self._C_pos is None:
            self._C_pos = np.mean(self.N_pos[self.C_node_idxs], axis=1)
        return self._C_pos

    def __make_c_node_idxs(self, clst: List[AbstractCell], nlst: List[Node]):
        n_id = np.array([n.id for n in nlst])

        c_node_idxs = np.zeros((len(clst), len(clst[0].node_list)), dtype=np.int64)

        for c_ii, c in enumerate(clst):
            for n_ii, n in enumerate(c.node_list):
                c_node_idxs[c_ii, n_ii] = np.where(n_id == n.id)[0][0]
        return c_node_idxs

    def __make_id2idx(self):
        N_id2idx = np.zeros((int(np.max(self.N_id)) + 1,), dtype=np.int64)
        for ii in range(self.N_id.shape[0]):
            N_id2idx[self.N_id[ii]] = ii
        return N_id2idx

    def __determine_cilia_direction(self, elst: List[Element]):
        cilia = np.zeros((len(elst),), dtype=np.float32)
        for ii, e in enumerate(elst):
            if e.pointing_forward is not None:
                cilia[ii] = 1 if e.pointing_forward else -1
        return cilia

    def __make_cell2node_matrix(self, clst: List[AbstractCell]):
        matrix = np.zeros(self.C_node_idxs.shape[:2] + (self.N_pos.shape[0],), dtype=np.float32)
        for c_ii, c in enumerate(clst):
            for n_ii, n in enumerate(c.node_list):
                matrix[c_ii, n_ii, self.C_node_idxs[c_ii, n_ii]] = 1.
        return matrix

    def __make_cell2element_matrix(self, clst: List[AbstractCell]):
        matrix = np.zeros((self.C_node_idxs.shape[0], self.E_node_1.shape[0]), dtype=np.float32)
        for c_ii, c in enumerate(clst):
            for e in c.element_list:
                matrix[c_ii, e.id] = 1.
        return matrix

    def __make_element2node_matrix(self, elst):
        matrix = np.zeros((self.E_node_1.shape[0], self.N_pos.shape[0], 2), dtype=np.float32)
        for eii, e in enumerate(elst):
            matrix[eii, self.N_id2idx[e.node_1.id], 0] = 1.
            matrix[eii, self.N_id2idx[e.node_2.id], 1] = 1.
        return matrix

    def __make_node2element_mask(self, elst: List[Element]):
        mask = np.zeros((self.N_pos.shape[0], self.E_node_1.shape[0]), dtype=np.bool)
        for eii, e in enumerate(elst):
            mask[self.N_id2idx[e.node_1.id], eii] = True
            mask[self.N_id2idx[e.node_2.id], eii] = True
        return mask
