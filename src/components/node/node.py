import warnings
from typing import List

import torch
from torch import Tensor

from src.components.node.nodedata.elementneighbours import ElementNeighbours
from src.utils.errors import TodoException
from src.utils.tools import pyout


class Node:
    def __init__(self, x: Tensor, y: Tensor, id_):
        """
        A class specifying the details about nodes
        """
        self.x = x
        self.y = y
        self.position = torch.tensor([x, y])
        # Need to give the node a previous position so elements can move to a new nox on the very
        # first time step
        self.previous_position = torch.tensor([x, y])
        self.id = id_
        self.force = torch.tensor([0., 0.])
        self.previous_force = torch.tensor([0., 0.])

        # This will be circular - each element will have two nodes. Each node can be part of
        # multiple elements, similarly for cells
        self.element_list = []
        self.cell_list: List = []
        self.is_top_node = None

        # Each node stores its local drag coefficient, so we can distinguish between different
        # regions in a tissue more easily
        self.eta = 1

        self.node_adjusted = False
        self.pre_adjusted_position = []

        self.node_data = [ElementNeighbours()]

    def delete(self):
        raise TodoException

    def add_force_contribution(self, force: Tensor):
        if torch.any(force.isnan() | force.isinf()):
            raise ValueError(f"Force is inf or NaN")
        else:
            self.force = self.force + force

    def move_node(self, pos):
        """
        This function is used to move the position due to time stepping so the force must be
        reset here. This is only to be used by the numerical integration
        :param pos:
        :return:
        """
        self.__new_position(pos)
        # Reset the force for the next time step
        self.previous_force = self.force
        self.force = torch.zeros_like(self.force)

    def adjust_position(self, pos):
        """
        Used when modifying the position manually. Doesn't affect previous position, or reset the
        force. But it will require fixing up the space partition
        :param pos:
        :return:
        """
        raise TodoException

    def add_element(self, e):
        if e in self.element_list:
            warnings.warn(f"Adding an element that already appears in element_list for node "
                          f"{self.id}. This has not been added.")
        else:
            self.element_list.append(e)

    def remove_element(self, e):
        raise TodoException

    def replace_element_list(self, e_list):
        raise TodoException

    def add_cell(self, c):
        raise TodoException

    def remove_cell(self, c):
        raise TodoException

    def replace_cell_list(self, c_list):
        """
        Used for CellFree to overwrite the existing cell. Does not modify any links in the cell,
        it assumes they are handled in the division or creation process
        :param c_list:
        :return:
        """
        raise TodoException

    def set_drag_coefficient(self, eta):
        """
        Use this to change the drag coefficient so that the associated elements have their
        properties updated
        :param eta:
        :return:
        """
        raise TodoException

    def get_neighbouring_elements(self, t):
        """
        Used to find nodes that are in proximity and are not explicitily connected to this
        node. The simulation managinf object must be passeed in because this function needs
        access to the space partition
        :param t:
        :return:
        """
        raise TodoException

    def __new_position(self, pos):
        """
        Should not be used directly, only as part of MoveNode
        :param pos:
        :return:
        """
        self.previous_position = self.position
        self.position = pos
        self.x, self.y = pos

    def __str__(self):
        return f"Node {self.id}"


if __name__ == '__main__':
    print("foo")
