from biobots2D.components.cell.abstractcell import AbstractCell
from biobots2D.components.cell.celldata.cellarea import CellArea
from biobots2D.components.cell.celldata.cellcentre import CellCentre
from biobots2D.components.cell.celldata.cellperimeter import CellPerimeter
from biobots2D.components.cell.celldata.targetarea import TargetArea
from biobots2D.components.cell.celldata.targetperimeter import TargetPerimeter
from biobots2D.components.cell.divisionnode.abstractsplitnode import AbstractSplitNode
from biobots2D.components.cell.divisionnode.randomnode import RandomNode
from biobots2D.components.cell.element import Element
from biobots2D.utils.errors import TodoException


class CellFree(AbstractCell):
    def __init__(self, *args):
        super().__init__()
        if len(args) == 3:
            cycle, node_list, id_ = args
            element_list = None
        else:
            cycle, node_list, element_list, id_ = args
        self.set_cell_cycle_model(cycle)
        self.node_list = node_list
        self.id = id_
        self.ancestor_id = id_

        self.cell_data_array = [CellArea(), CellPerimeter(), CellCentre(), TargetPerimeter(),
                                TargetArea()]

        self.add_cell_data(self.cell_data_array)

        self.node_list = node_list
        self.num_nodes = len(node_list)

        if element_list is None:
            # This case happens when a new simulation is created (or if a cell springs out of
            # nowhere)

            for ii in range(len(node_list)):
                e = Element(node_list[ii], node_list[(ii + 1) % len(node_list)],
                            id=0)
                            # f"{node_list[ii].id}_{node_list[(ii + 1) % len(node_list)].id}")
                self.element_list.append(e)
                node_list[ii].cell_list.append(self)
                e.cell_list.append(self)
        else:
            # This case is mainly used for dividing cells, but may also be used to make new cells
            # at the beginning of a simulation

            self.element_list = element_list

            # Should throw in a verification step to check that nodes are in anticlockwise order
            # and match with elements

            for ii in range(len(node_list)):
                self.node_list[ii].replace_cell_list(self)

            for ii in range(len(element_list)):
                self.element_list[ii].replace_cell_list(self)

        self.split_node_function: AbstractSplitNode = RandomNode()

    def divide(self):
        """
        Divide cell when simulation is made of free cells that are not constrained to be adjacent
        to others. To divide, split the cell in half along a specified axis and add in sufficient
        nodes and elements to maintain a constant number of nodes and elements

        This process needs to be done carefully to update all the new links between node,
        element and cell

          o---------o
          |         |
          |         |
          |     1   |
          |         |
          |         |
          o---------o

        With an even number of elements becomes

          x  o------o
          |\  \     |
          | \  \    |
          |  x  x 1 |
          | 2 \  \  |
          |    \  \ |
          o-----x   o

        With an odd number of elements, it's harder to draw, but need to choose an element to
        split or give uneven spread of new elements

        Find the split points

        Give -ve ids because id is a feature of the simulation and can't be assigned here. This
        is handled in AbstractCellSimulation

        Splitnode identifies the point where division starts. A line is drawn across to the
        halfway point when travelling around the perimeter of the cell (this won't necessarily
        cut the cell directly in half, but it should be close). The halfway point will be a
        node if the cell has an even number of nodes, or the mid point of an element if it's odd

        In both cases the splitnode will be replaced by two nodes. Even will mean the opposite
        node will be split in two, (one node is added). Odd will mean the opposite element gets
        split in two (two nodes are added)
        :return:
        """
        raise TodoException

    def is_point_inside_cell(self, point):
        """

        :param point:
        :return:
        """
        raise TodoException

    def get_next_node(self, n, direction):
        """
        Use the elements to find the next node around the perimeter in direction
        :param n:
        :param direction: 1 -> anticlockwise, -1 -> clockwise
        :return:
        """
        raise TodoException

    def get_next_element(self, e, direction):
        """
        Use the nodes to find the next element around the perimeter in direction
        :param n:
        :param direction: 1 -> anticlockwise, -1 -> clockwise
        :return:
        """
        raise TodoException

    def get_nodes_between(self, start_node, end_node, direction):
        """
        This returns the nodes between start and end in direction EXCLUDING the end points
        :param start_node:
        :param end_node:
        :param direction:
        :return:
        """
        raise TodoException

    def get_elements_between(self, start_element, end_element, direction):
        """
        This returns the elements between start and end in direction EXCLUDING the end points
        :param start_element:
        :param end_element:
        :param direction:
        :return:
        """
        raise TodoException

    def get_elements_between_inclusive(self, start_element, end_element, direction):
        """
        This returns the elements between start and end in direction INCLUDING the end points
        :param start_element:
        :param end_element:
        :param direction:
        :return:
        """
        raise TodoException

    def get_split_node(self):
        """
        Using something that defines the division axis, chose a node to start the split from.
        The other side will be on the opposite side of the cell and could be an element or a
        node, depending on even or oddness of the node count
        :return:
        """
        raise TodoException

    def get_opposite_node(self, ii):
        """
        Gets the node opposite node given by index i in nodeList.
        :param ii:
        :return:
        """
        raise TodoException

    def get_opposite_element(self, ii):
        """
        Gets the node opposite element from node in index i in nodeList. This is only used for
        odd numbers of nodes.
        :param ii:
        :return:
        """
        raise TodoException

    def make_intermediate_nodes(self, s, i):
        raise TodoException

    def get_split_vector(self, s, i):
        """
        Makes a vector from the split node to the opposite side (whether that is an edge or a
        node), so the intermediate ndoes can be built. Uses the index i of the chosen node
        :param s:
        :param i:
        :return:
        """
        raise TodoException
