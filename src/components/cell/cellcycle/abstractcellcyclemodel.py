from abc import ABC, abstractmethod

from src.components.cell.colourset import ColourSet
from src.utils.errors import TodoException


class AbstractCellCycleModel(ABC):
    def __init__(self):
        """
        An abstract class that gets the basics of a cell cycle model
        """
        self.age = None
        self.colour_set = ColourSet()
        self.colour = 1
        self.containing_cell = []

    @abstractmethod
    def is_ready_to_divide(self):
        """
        Returns true if the cell meets the conditions for dividing
        :return:
        """
        pass

    @abstractmethod
    def get_growth_phase_fraction(self):
        """
        If a cell grows, then need to know the point in this growth. This should vary from 0 (
        equal the new cell size) to 1 (fully grown), but there is no reason it can't go above 1
        if max cell size is variable.
        :return:
        """
        pass

    @abstractmethod
    def duplicate(self):
        """
        When a cell divides, duplicate the ccm for the new cell
        :return:
        """

    def get_age(self):
        """

        :return:
        """
        raise TodoException

    def set_age(self, age):
        """

        :param age:
        :return:
        """
        self.age = age

    def age_cell_cycle(self, dt):
        """

        :param dt:
        :return:
        """
        raise TodoException

    def get_colour(self):
        """

        :return:
        """
        raise TodoException
