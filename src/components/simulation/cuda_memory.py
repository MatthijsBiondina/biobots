from typing import List, Union

# import numpy as cp
import cupy as cp
# from numba import cuda, float32, vectorize, guvectorize, int64
import numpy as np
from numba.cuda.cudadrv.devicearray import DeviceNDArray as DNDA

from src.components.cell.abstractcell import AbstractCell
from src.components.cell.element import Element
from src.components.cudaspacepartition import CudaSpacePartition
from src.components.node.node import Node
from src.components.spacepartition import SpacePartition
from src.utils.tools import pyout


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

    def __init__(self,
                 cell_list: List[AbstractCell],
                 element_list: List[Element],
                 node_list: List[Node]):
        # Cell data
        self.C_node_ids = cp.array([[n.id for n in c.node_list] for c in cell_list])
        self.C_node_idxs = self.__make_c_node_idxs(cell_list, node_list)
        self.C_element_idxs = cp.array([[e.id for e in c.element_list] for c in cell_list])
        self.C_age = list2cupy([c.age for c in cell_list])
        self.C_type = list2cupy([c.cell_type for c in cell_list])
        self.C_grown_cell_target_area = list2cupy([c.grown_cell_target_area for c in cell_list])

        # Cell type indexes
        self.Ctype_0 = cp.argwhere(self.C_type == 0)
        self.Ctype_1 = cp.argwhere(self.C_type == 1)

        # Node data
        self.N_id = cp.array([n.id for n in node_list])
        self.N_cell_ids = cp.array([])
        self.N_pos = cp.array([n.position.numpy() for n in node_list])
        self.N_for = cp.array([n.force.numpy() for n in node_list])
        self.N_pos_previous = None
        self.N_for_previous = None
        self.N_eta = cp.array([n.eta for n in node_list], dtype=cp.float32)

        # Element data
        self.E_node_1 = cp.array([e.node_1.id for e in element_list])
        self.E_node_2 = cp.array([e.node_2.id for e in element_list])

        # SpacePartition
        # self.boxes = CudaSpacePartition(space_partition)

        # Broadcast matrices
        self.cell2node = self.__make_cell2node_matrix(cell_list)
        self.cell2element = self.__make_cell2element_matrix(cell_list)
        self.rotate_clockwise_2d = cp.array([[0., -1.], [1., .0]], dtype=cp.float32)

        # Masks
        # self.node2element_mask = self.__make_node2element_matrix(element_list)
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
        self._polygons = None
        self._E_length = None

        # self._dmatrix_l2 = None
        # numerical placeholders
        self.pi = cp.float32(np.pi)

    def clear_dynamic_memory(self):
        self._vector_1_to_2 = None
        self._outward_normal = None
        self._element_length = None
        self._C_area = None
        self._C_perimeter = None
        self._C_target_area = None
        self._C_target_perimeter = None
        self._polygons = None
        self._E_length = None

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
            self._C_target_area = self.C_grown_cell_target_area
            # todo: check thant c_grown_cell_target_area doesn't get set to None when dynamic
            # memory is cleared
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
            self._C_target_perimeter[self.Ctype_0] = \
                (4 * self.C_target_area[self.Ctype_0] * n * cp.tan(self.pi / n)) ** .5
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

    def __make_cell2node_matrix(self, clst: List[AbstractCell]):
        matrix = cp.zeros((self.C_node_idxs.shape[0], self.C_node_idxs.shape[1],
                           self.N_pos.shape[0]))
        for cii, c in enumerate(clst):
            for nii, n in enumerate(c.node_list):
                matrix[cii][nii][self.C_node_idxs[cii, nii]] = 1.

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
