# import cupy as cp
import numpy as cp
from biobots2D.components.forces.cellbasedforce.abstractcellbasedforce import AbstractCellBasedForce

from biobots2D.components.simulation.cuda_memory import CudaMemory


class CiliaPropagationForce(AbstractCellBasedForce):
    def __init__(self, propagation_magnitude):
        self.propagation_magnitude = cp.float32(propagation_magnitude)

    def add_cell_based_forces(self, cell_list: list, gpu: CudaMemory):
        """

        :param cell_list:
        :param gpu:
        :return:
        """
        candidates = gpu.candidates
        blocked = cp.any(candidates, axis=0)

        # magnitude = sigmoid(
        #     gpu.C_inhibitory[gpu.E_cell_idx] * gpu.spice) \
        #             * self.propagation_magnitude
        magnitude = self.propagation_magnitude


        F = gpu.vector_1_to_2 * gpu.E_cilia_direction[:, None] * magnitude
        F = cp.where(blocked[:,None], cp.zeros_like(F), F)
        if gpu.spice > 0.5:
            F = cp.where((gpu.C_inhibitory[gpu.E_cell_idx] == 1)[:,None], cp.zeros_like(F), F)
        else:
            F = cp.where((gpu.C_inhibitory[gpu.E_cell_idx] == -1)[:,None], cp.zeros_like(F), F)


        gpu.N_for[gpu.E_node_1] += F / 2
        gpu.N_for[gpu.E_node_2] += F / 2


