import random
from abc import ABC, abstractmethod


class AbstractCellCycleModel(ABC):
    def __init__(self):
        """
        An abstract class that gets the basics of a cell cycle model
        """
        self.age = None
        self.colour_set = ColourSet()
        self.colour = 1
        self.containing_cell = None

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
        raise NotImplementedError

    def set_age(self, age):
        """

        :param age:
        :return:
        """
        raise NotImplementedError

    def age_cell_cycle(self, dt):
        """

        :param dt:
        :return:
        """
        raise NotImplementedError

    def get_colour(self):
        """

        :return:
        """
        raise NotImplementedError


class GrowthContactInhibition(AbstractCellCycleModel):
    def __init__(self, p, g, f, dt):
        """

        :param p:
        :param g:
        :param f:
        :param dt:
        """

        # properties
        self.mean_pause_phase_duration = None
        self.pause_phase_duration = None

        self.mean_growth_phase_duration = None
        self.growth_phase_duration = None

        self.minimum_pause_phase_duration = 0
        self.minimum_growth_phase_duration = 4

        self.pause_phase_rng = random.random()
        self.growth_phase_rng = random.random()

        self.pause_colour = None
        self.growth_colour = None
        self.inhibited_colour = None

        # constructor
        self.pause_phase_rng = lambda: random.uniform(-2, 2)
        self.growth_phase_rng = lambda: random.uniform(-2, 2)

        self.set_pause_phase_duration(p)
        self.set_growth_phase_duration(g)

        self.growth_trigger_fraction = f

        self.dt = dt

        # By default cell will start off in the pause phase
        self.set_age(round(random.uniform(0, self.pause_phase_duration()), 1))

        self.pause_colour = self.colour_set.get_number('PAUSE')
        self.growth_colour = self.colour_set.get_number('GROW')
        self.inhibited_colour = self.colour_set.get_number('STOPPED')

    def age_cell_cycle(self, dt):
        """
        Redefine the age_cell_cycle method to update the phase colour.
        :param dt:
        :return:
        """
        raise NotImplementedError

    def duplicate(self):
        """

        :return:
        """
        raise NotImplementedError

    def is_ready_to_divide(self):
        """

        :return:
        """
        raise NotImplementedError

    def get_growth_phase_fraction(self):
        """

        :return:
        """
        raise NotImplementedError

    def set_pause_phase_duration(self, p):
        """

        :param p:
        :return:
        """
        raise NotImplementedError

    def set_growth_phase_duration(self, g):
        """

        :param g:
        :return:
        """
        raise NotImplementedError

    def set_pause_phase_rng(self, func):
        """

        :param func:
        :return:
        """
        raise NotImplementedError

    def set_growth_phase_rng(self, func):
        """

        :param func:
        :return:
        """
        raise NotImplementedError
