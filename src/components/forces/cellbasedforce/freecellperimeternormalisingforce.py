from typing import List

from src.components.cell.abstractcell import AbstractCell
from src.components.forces.cellbasedforce.abstractcellbasedforce import AbstractCellBasedForce
from src.components.simulation.cuda_memory import CudaMemory
from src.utils.errors import TodoException
from src.utils.tools import pyout


class FreeCellPerimeterNormalisingForce(AbstractCellBasedForce):
    def __init__(self, spring_rate):
        """
        A normalising force to keep the edges around a free cell roughly the same size. It will
        push each edge to have length P/N, where P is the current perimeter and N the number of
        edges
        """
        self.spring_rate = spring_rate

    def add_cell_based_forces(self, cell_list: List[AbstractCell], gpu: CudaMemory):
        """
        For each cell in the list, calculate the forces and add them to the nodes
        :param cell_list:
        :return:
        """
        for c in cell_list:
            self.apply_spring_force(c)


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
