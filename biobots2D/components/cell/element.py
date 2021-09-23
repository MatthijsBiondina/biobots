import torch
from torch import tensor

from biobots2D.components.node.node import Node
from utils.errors import TodoException


class Element:
    def __init__(self, node_1, node_2, id, pointing_forward=None):
        """
        All the initilising
        An element will always have a pair of nodes

        This ordering is important, because it defines the orientation of the element. If the
        element is completely free in space, then the orientation is arbitrary, but as soon as it
        is part of a cell (or membrane if I end up developing one), then 1 to 2 must be chosen so
        that if u = (Node2.position - Node1.position) / length, then the unit vector v = (u[2],
        -u[1]) points out of the tissue. Then, v will point to the right if forward is 1 to 2.
        Due to the choice of v, travelling 1 to 2 will go anticlockwise around the perimeter of
        the cell

        Note: v should be perpendicular, but I can't do that in ASCII art
                2 o
                 /
                /
               /\
              /  \v
            u/    \
            /
         1 o
        :param node_1:
        :param node_2:
        :param id:
        """
        # todo remove for debug. it's just a reminder
        if id == -1:
            raise ValueError("Update element id in MATLAB implementation")

        self.id = id

        # The 'total drag' for the element at the centre of drag. This will be constant for an
        # element, unless the coeficient of drag is changed for a node
        self.eta_d = None

        # This will be circular - each element will have two nodes. Each node can be part of
        # multiple elements
        self.node_1: Node = node_1
        self.node_2: Node = node_2

        # Used only when an element is modified during cell division. Keep track of the old node
        # to help with adjusting the element boxes
        self.old_node_1 = None
        self.old_node_2 = None

        self.modified_in_division = False

        self.natural_length = 1
        self.stiffness = 20

        self.minimum_length = 0.2

        self.node_list = [self.node_1, self.node_2]
        self.cell_list = []

        self.internal = False

        # Determines if an element is part of a membrane, hence should be subject to membrane
        # related forces
        self.is_membrane = False

        self.node_1.add_element(self)
        self.node_2.add_element(self)

        self.update_total_drag()

        self.pointing_forward = pointing_forward

    def delete(self):
        raise TodoException

    def update_total_drag(self):
        """
        Of the three important physical quantities total drag will only change if the drag
        coefficients are explicitly changed
        :return:
        """
        self.eta_d = self.node_1.eta + self.node_2.eta

    def get_moment_of_drag(self):
        """
        The length of the element will change at every time step, so ID needs to be calculated
        every time
        :return:
        """
        raise TodoException

    def set_natural_length(self, len_):
        raise TodoException

    def get_natural_length(self):
        raise TodoException

    def get_length(self):
        return torch.sum((self.node_1.position - self.node_2.position) ** 2) ** .5

    def get_vector_1_to_2(self):
        direction_1_to_2 = self.node_2.position - self.node_1.position
        direction_1_to_2 /= torch.linalg.norm(direction_1_to_2)
        return direction_1_to_2

    def get_outward_normal(self):
        """
        See constructor for discussion about why this is the outward normal
        :return:
        """
        u = self.get_vector_1_to_2()
        return u @ tensor([[0., -1.], [1., 0.]])

    def get_mid_point(self):
        raise TodoException

    def get_other_node(self, node: Node):
        if node == self.node_1:
            return self.node_2
        elif node == self.node_2:
            return self.node_1
        raise ValueError(f"Node {node.id} is not in Element {self.id}")

    def swap_nodes(self):
        """
        Used when the nodes are not anticlockwise 1 -> 2. An element has no concept of
        orientation by itself so this must be controlled from outside the element only
        :return:
        """
        raise TodoException

    def replace_node(self, old_node, new_node):
        """
        Removes the old node from the element, and replaces it with a new node. This is used in
        cell division primarily
        :param old_node:
        :param new_node:
        :return:
        """

    def add_cell(self, c):
        raise TodoException

    def get_other_cell(self, c):
        """
        Since we don't know what order the cells are in, we need a special way to grab the tutorials
        cell if we already know one of them. There will be two cases per simulation where there
        is no tutorials cell, so in these cases, return logical false

        :param c:
        :return:
        """
        raise TodoException

    def replace_cell(self, old_c, new_c):
        raise TodoException

    def replace_cell_list(self, cell_list):
        raise TodoException

    def remove_cel(self, c):
        raise TodoException

    def is_element_internal(self):
        return self.internal

    def __str__(self):
        return f"Element {self.id}"
