import sys
import time
from typing import List

import numpy as cp
# import cupy as cp
from biobots2D.components.cell.abstractcell import AbstractCell
from biobots2D.components.forces.cellbasedforce.abstractcellbasedforce import AbstractCellBasedForce
from biobots2D.components.simulation.cuda_memory import CudaMemory, synchronize
from biobots2D.utils.errors import TodoException
from biobots2D.utils.tools import pyout


class FreeCellPerimeterNormalisingForce(AbstractCellBasedForce):
    def __init__(self, spring_rate):
        """
        A normalising force to keep the edges around a free cell roughly the same size. It will
        push each edge to have length P/N, where P is the current perimeter and N the number of
        edges
        """
        self.spring_rate = spring_rate

        # CUDA placeholders
        self.spring_rate_cuda = cp.float32(spring_rate)

    def add_cell_based_forces(self, cell_list: List[AbstractCell], gpu: CudaMemory):
        """
        For each cell in the list, calculate the forces and add them to the nodes
        :param cell_list:
        :return:
        """
        if gpu.EXEC_CPU:
            for c in cell_list:
                self.apply_spring_force(c)

        self.apply_spring_force_cuda(gpu)

    def apply_spring_force(self, c: AbstractCell):
        """

        :param c:
        :return:
        """
        N = len(c.element_list)
        p = c.get_cell_perimeter() / N
        for e in c.element_list:
            unit_vector_1_to_2 = e.get_vector_1_to_2()

            l = e.get_length()
            mag = self.spring_rate * (p - l)

            force = mag * unit_vector_1_to_2

            e.node_1.add_force_contribution(-force)
            e.node_2.add_force_contribution(force)

    def apply_spring_force_cuda(self, gpu: CudaMemory):
        p = gpu.C_perimeter / gpu.C_node_idxs.shape[1]
        unit_vector_1_to_2 = gpu.vector_1_to_2
        l = gpu.element_length




        # mag = self.spring_rate_cuda * ((p @ gpu.cell2element) - l)

        mag = self.spring_rate_cuda * cp.log(l / (p @ gpu.cell2element)) / cp.log(0.5)


        force = unit_vector_1_to_2 * mag[:, None]
        gpu.N_for[gpu.E_node_1] -= force
        gpu.N_for[gpu.E_node_2] += force
