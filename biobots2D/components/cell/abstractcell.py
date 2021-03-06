from abc import ABC, abstractmethod
from typing import List

from biobots2D.components.cell.cellcycle.abstractcellcyclemodel import AbstractCellCycleModel
from biobots2D.components.cell.element import Element
from biobots2D.components.node.node import Node
from utils import TodoException


class AbstractCell(ABC):
    def __init__(self):
        self.id = None

        self.age = 0

        self.node_list: List[Node] = []
        self.element_list: List[Element] = []

        self.new_cell_target_area = 0.5
        self.grown_cell_target_area = 1

        self.cell_cycle_model: AbstractCellCycleModel = None

        self.deformation_energy_parameter = 10
        self.surface_energy_parameter = 1

        # Determines if we are using a free or joined cell model
        self.free_cell = False
        self.new_free_cell_separation = 0.1

        # When a cell divides in two, this will store the sister cell after division
        self.sister_cell = None

        # Stores the id of the cell that was in the initial configuration. Only can store the id
        # because the cell can be deleted from the simulation
        self.ancestor_id = None

        # A collection objects for calculating data about the cell stored in a dict
        self.cell_data = {}

        # By default, the type is 0, matching a general epithelial cell
        self.cell_type = 0
        self.inhibitory = 0

    @abstractmethod
    def divide(self):
        pass

    @abstractmethod
    def is_point_inside_cell(self, point):
        pass

    def delete(self):
        raise TodoException

    def set_cell_cycle_model(self, v: AbstractCellCycleModel):
        self.cell_cycle_model = v
        v.containing_cell = self

    def get_cell_area(self):
        """
         This and the following 3 functions could be replaced by accessing the cellData but
         they're kept here for backwards compatibility, and because these types of data are
         fundamental enough to designate a function

        :return:
        """
        return self.cell_data['cell_area'].get_data(self)

    def get_cell_target_area(self):
        """
        This is so the target area can be a function of cell age
        :return:
        """
        return self.cell_data['target_area'].get_data(self)

    def get_cell_perimeter(self):
        return self.cell_data['cell_perimeter'].get_data(self)

    def get_cell_centre(self):
        raise TodoException

    def get_cell_target_perimeter(self):
        """
        This is so the target area can be a function of cell age
        :return:
        """
        return self.cell_data['target_perimeter'].get_data(self)

    def is_ready_to_divide(self):
        return self.cell_cycle_model.is_ready_to_divide()

    def add_cell_data(self, d):
        """

        :param d:
        :return:
        """
        for d_ in d:
            self.cell_data[d_.name] = d_

    def age_cell(self, dt):
        # This will be done at the end of the time step
        self.age += dt

        # todo: this should not be reached in GPU mode
        self.cell_cycle_model.age_cell_cycle(dt)

    def get_age(self):
        return self.cell_cycle_model.get_age()

    def get_colour(self):
        return self.cell_cycle_model.get_colour()

    def do_elements_cross(self, e1, e2):
        raise TodoException

    def is_cell_self_intersecting(self):
        """
        This is the slowest possible algorithm to check self intersection so it should only be
        used WHEN INITIALISING A CELL. It should NOT be used regularly to detect intersections
        because it is EXTREMELY SLOW
        :return:
        """
        raise TodoException

    def is_node_inside_cell(self, n):
        raise TodoException

    def draw_cell(self):
        """
        Mainly for debugging
        :return:
        """
        raise TodoException

    def draw_cell_previous(self):
        raise TodoException
