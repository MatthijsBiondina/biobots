from pybrown.components.forces.cellbasedforce.abstractcellbasedforce import AbstractCellBasedForce
from pybrown.utils.errors import TodoException


class FreeCellPerimeterNormalisingForce(AbstractCellBasedForce):
    def __init__(self, spring_rate):
        """
        A normalising force to keep the edges around a free cell roughly the same size. It will
        push each edge to have length P/N, where P is the current perimeter and N the number of
        edges
        """
        self.spring_rate = spring_rate

    def add_cell_based_forces(self, cell_list: list):
        """
        For each cell in the list, calculate the forces and add them to the nodes
        :param cell_list:
        :return:
        """
        raise TodoException

    def apply_spring_force(self, c):
        """

        :param c:
        :return:
        """
        raise TodoException