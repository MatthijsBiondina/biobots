import time
from typing import List, Union

# import numpy as cp
import cupy as cp
# from numba import cuda, float32, vectorize, guvectorize, int64
import numpy as np

from biobots2D.components.cell.abstractcell import AbstractCell
from biobots2D.components.cell.element import Element
from biobots2D.components.node.node import Node


def list2cupy(lst: List, property_: Union[None, str] = None):
    if property_ is None:
        return cp.array(lst, dtype=cp.float32)
    else:
        return cp.array([getattr(item, property_) for item in lst], dtype=cp.float32)


def synchronize():
    cp.cuda.stream.get_current_stream().synchronize()


def dot(A: cp.ndarray, B: cp.ndarray) -> cp.ndarray:
    return cp.einsum('...i,...i->...', A, B)


class CudaMemory:
    EXEC_CPU = False
    RENDER = True

    def __init__(self,
                 cell_list: List[AbstractCell],
                 element_list: List[Element],
                 node_list: List[Node],
                 d_limit: float):
        # hyperparameters
        self.d_limit = cp.float32(d_limit)

        # Cell data
        self.C_node_ids = cp.array([[n.id for n in c.node_list] for c in cell_list])
        self.C_node_idxs = self.__make_c_node_idxs(cell_list, node_list)
        self.C_element_idxs = cp.array([[e.id for e in c.element_list] for c in cell_list])
        self.C_age = list2cupy([c.age for c in cell_list])
        self.C_type = list2cupy([c.cell_type for c in cell_list])
        self.C_grown_cell_target_area = list2cupy([c.grown_cell_target_area for c in cell_list])
        self.C_inhibitory = cp.array([c.inhibitory for c in cell_list])

        # Cell type indexes
        self.Ctype_0 = cp.argwhere(self.C_type == 0).squeeze(1)
        self.Ctype_1 = cp.argwhere(self.C_type == 1).squeeze(1)
        self.Ctype_2 = cp.argwhere(self.C_type == 2).squeeze(1)
        self.Ctype_3 = cp.argwhere(self.C_type == 3).squeeze(1)
        self.Ctype_4 = cp.argwhere(self.C_type == 4).squeeze(1)

        # Node data
        self.N_id = cp.array([n.id for n in node_list])
        self.N_id2idx = self.__make_id2idx()
        self.N_pos = cp.array([n.position.numpy() for n in node_list])
        self.N_for = cp.array([n.force.numpy() for n in node_list])
        self.N_pos_previous = None
        self.N_for_previous = None
        self.N_eta = cp.array([n.eta for n in node_list], dtype=cp.float32)

        # Element data
        self.E_node_1_id = cp.array([e.node_1.id for e in element_list])
        self.E_node_2_id = cp.array([e.node_2.id for e in element_list])
        self.E_cell_idx = cp.array([e.cell_list[0].id for e in element_list])
        self.E_node_1 = self.N_id2idx[self.E_node_1_id]
        self.E_node_2 = self.N_id2idx[self.E_node_2_id]
        self.E_internal = cp.array([e.internal for e in element_list])
        self.E_cilia_direction = cp.array(
            [0 if e.pointing_forward is None else
             1 if e.pointing_forward else -1 for e in element_list])

        # SpacePartition
        # self.boxes = CudaSpacePartition(space_partition)

        # Broadcast matrices
        self.cell2node = self.__make_cell2node_matrix(cell_list)
        self.cell2element = self.__make_cell2element_matrix(cell_list)
        self.element2node = self.__make_element2node_matrices(element_list)
        self.rotate_clockwise_2d = cp.array([[0., -1.], [1., .0]], dtype=cp.float32)

        # Masks
        self.node2element_mask = self.__make_node2element_mask(element_list)
        # self.cell2node_mask = self.cell2node.astype(cp.bool)
        self.cell2element_mask = self.cell2element.astype(cp.bool)

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

        # self._dmatrix_l2 = None
        # numerical placeholders
        self.pi = cp.float32(np.pi)

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

        # self._dmatrix_l2 = None

    @property
    def vector_1_to_2(self) -> cp.ndarray:
        if self._vector_1_to_2 is None:
            direction_1_to_2 = self.N_pos[self.E_node_2] - self.N_pos[self.E_node_1]
            direction_1_to_2 /= cp.linalg.norm(direction_1_to_2, axis=1)[:, None]
            self._vector_1_to_2 = direction_1_to_2
        return self._vector_1_to_2

    @property
    def outward_normal(self) -> cp.ndarray:
        if self._outward_normal is None:
            self._outward_normal = self.vector_1_to_2 @ self.rotate_clockwise_2d
        return self._outward_normal

    @property
    def element_length(self) -> cp.ndarray:
        if self._element_length is None:
            length = (self.N_pos[self.E_node_1] - self.N_pos[self.E_node_2]) ** 2
            length = cp.sum(length, axis=1) ** .5
            self._element_length = length
        return self._element_length

    @property
    def C_area(self) -> cp.ndarray:
        if self._C_area is None:
            x, y = self.polygons[:, :, 0], self.polygons[:, :, 1]
            self._C_area = 0.5 * cp.abs(dot(x, cp.roll(y, 1, axis=1))
                                        - dot(y, cp.roll(x, 1, axis=1)))
        return self._C_area

    @property
    def C_target_area(self) -> cp.ndarray:
        if self._C_target_area is None:
            self._C_target_area = cp.empty_like(self._C_area)

            # Type 0 has constant target area
            self._C_target_area[:] = self.C_grown_cell_target_area[:]

            # Type 1 grows and shrinks as a polygon

            diff = 1.5

            self._C_target_area[self.Ctype_1] = \
                (1. + 0.5 * diff + 0.5 * diff * -cp.sin(self._t * 5)) * \
                self.C_grown_cell_target_area[
                    self.Ctype_1]

            # Type 2 has constant target area
            # self._C_target_area[self.Ctype_2] = self.C_grown_cell_target_area[self.Ctype_2]
            # self._C_target_area[self.Ctype_3] = self.C_grown_cell_target_area[self.Ctype_3]

        return self._C_target_area

    @property
    def C_perimeter(self) -> cp.ndarray:
        if self._C_perimeter is None:
            self._C_perimeter = cp.sum(self.E_length[self.C_element_idxs], axis=1)
        return self._C_perimeter

    @property
    def C_target_perimeter(self) -> cp.ndarray:
        if self._C_target_perimeter is None:
            self._C_target_perimeter = cp.empty_like(self.C_target_area)

            # Type 0 wants to be a regular polygon
            n = self.C_element_idxs.shape[1]
            self._C_target_perimeter = \
                (4 * self.C_target_area * n * cp.tan(self.pi / n)) ** .5
            self._C_target_perimeter[self.Ctype_1] = \
                (4 * self.C_target_area[self.Ctype_1] * n * cp.tan(self.pi / n)) ** .5
            # self._C_target_perimeter[self.Ctype_2] = \
            #     (4 * self.C_target_area[self.Ctype_2] * n * cp.tan(self.pi / n)) ** .5
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
            self._E_length = cp.sum((n1 - n2) ** 2, axis=1) ** .5
        return self._E_length

    @property
    def candidates(self):
        if self._candidates is None:
            t0 = time.time()
            candidates = cp.ones((self.N_pos.shape[0], self.E_node_1.shape[0]), dtype=cp.bool)

            # find nodes in element interaction regions
            pnt = self.N_pos
            start = self.N_pos[self.E_node_1]
            end = self.N_pos[self.E_node_2]
            line_vec = end - start
            pnt_vec = pnt[:, None] - start[None, :]
            line_len = cp.linalg.norm(line_vec, axis=1)
            line_unitvec = line_vec / line_len[:, None]
            pnt_vec_scaled = pnt_vec / line_len[None, :, None]
            t = cp.sum(line_unitvec[None, :, :] * pnt_vec_scaled, axis=2)
            nearest = line_vec[None, :, :] * t[:, :, None]
            dist = cp.linalg.norm(nearest - pnt_vec, axis=2)
            candidates = candidates & (0. <= t) & (t <= 1.) & (dist < self.d_limit)

            candidates = candidates & ~ self.node2element_mask
            self._candidates = candidates
        return self._candidates

    @property
    def C_pos(self):
        if self._C_pos is None:
            self._C_pos = cp.mean(self.N_pos[self.C_node_idxs], axis=1)
        return self._C_pos

    def __make_cell2node_matrix(self, clst: List[AbstractCell]):
        matrix = cp.zeros((self.C_node_idxs.shape[0], self.C_node_idxs.shape[1],
                           self.N_pos.shape[0]))
        for cii, c in enumerate(clst):
            for nii, n in enumerate(c.node_list):
                matrix[cii][nii][self.C_node_idxs[cii, nii]] = 1.
        return matrix

    def __make_element2node_matrices(self, elst: List[Element]):
        matrix = cp.zeros((self.E_node_1_id.shape[0], self.N_pos.shape[0], 2), dtype=cp.float32)
        for eii, e in enumerate(elst):
            matrix[eii, self.N_id2idx[e.node_1.id], 0] = 1.
            matrix[eii, self.N_id2idx[e.node_2.id], 1] = 1.
        return matrix

    def __make_cell2element_matrix(self, cell_list: List[AbstractCell]):
        matrix = cp.zeros((self.C_node_idxs.shape[0], self.E_node_1.shape[0]), dtype=cp.float32)
        for c_ii, c in enumerate(cell_list):
            for e in c.element_list:
                matrix[c_ii, e.id] = 1.
        return matrix

    def __make_c_node_idxs(self, clst: List[AbstractCell], nlst: List[Node]):
        n_id = cp.array([n.id for n in nlst])

        c_node_idxs = cp.zeros((len(clst), len(clst[0].node_list)), dtype=cp.int64)

        for c_ii, c in enumerate(clst):
            for n_ii, n in enumerate(c.node_list):
                c_node_idxs[c_ii, n_ii] = cp.where(n_id == n.id)[0][0]
        return c_node_idxs

    def __make_id2idx(self):
        N_id2idx = cp.zeros((int(cp.max(self.N_id)) + 1,), dtype=cp.int64)
        for ii in range(self.N_id.shape[0]):
            N_id2idx[self.N_id[ii]] = ii
        return N_id2idx

    def __make_node2element_mask(self, elst: List[Element]):
        mask = cp.zeros((self.N_pos.shape[0], self.E_node_1.shape[0]), dtype=cp.bool)
        for eii, e in enumerate(elst):
            mask[self.N_id2idx[e.node_1.id], eii] = True
            mask[self.N_id2idx[e.node_2.id], eii] = True
        return mask
