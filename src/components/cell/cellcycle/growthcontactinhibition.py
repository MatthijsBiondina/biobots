# import random

from src.components.cell.cellcycle.abstractcellcyclemodel import AbstractCellCycleModel
from src.utils.errors import TodoException
from src.utils.tools import prng


class GrowthContactInhibition(AbstractCellCycleModel):
    def __init__(self, p, g, f, dt):
        """

        :param p:
        :param g:
        :param f:
        :param dt:
        """
        super().__init__()

        # properties
        self.mean_pause_phase_duration = None
        self.pause_phase_duration = None

        self.mean_growth_phase_duration = None
        self.growth_phase_duration = None

        self.minimum_pause_phase_duration = 0
        self.minimum_growth_phase_duration = 4

        # todo: We use our own psuedo-rng for now for interchangability with MATLAB version
        # self.pause_phase_rng = random.random()
        # self.growth_phase_rng = random.random()
        self.pause_phase_duration = prng()
        self.growth_phase_duration = prng()

        self.pause_colour = None
        self.growth_colour = None
        self.inhibited_colour = None

        # todo: We use our own psuedo-rng for now for interchangability with MATLAB version
        # constructor
        # self.pause_phase_rng = lambda: random.uniform(-2, 2)
        # self.growth_phase_rng = lambda: random.uniform(-2, 2)
        self.pause_phase_rng = lambda: (prng() * 4) - 2
        self.growth_phase_rng = lambda: (prng() * 4) - 2

        self.set_pause_phase_duration(p)
        self.set_growth_phase_duration(g)

        self.growth_trigger_fraction = f

        self.dt = dt

        # By default cell will start off in the pause phase
        # todo: We use our own psuedo-rng for now for interchangability with MATLAB version
        # self.set_age(round(random.uniform(0, self.pause_phase_duration), 1))
        self.set_age(round(prng() * self.pause_phase_duration, 1))

        self.pause_colour = self.colour_set.get_number('PAUSE')
        self.growth_colour = self.colour_set.get_number('GROW')
        self.inhibited_colour = self.colour_set.get_number('STOPPED')

    def age_cell_cycle(self, dt):
        """
        Redefine the age_cell_cycle method to update the phase colour.
        :param dt:
        :return:
        """
        self.age += dt

        if self.age < self.pause_phase_duration:
            self.colour = self.pause_colour
        else:
            raise TodoException

    def duplicate(self):
        """

        :return:
        """
        raise TodoException

    def is_ready_to_divide(self):
        """

        :return:
        """
        return self.pause_phase_duration + self.growth_phase_duration < self.get_age()

    def get_growth_phase_fraction(self):
        """

        :return:
        """
        if self.age < self.pause_phase_duration:
            return 0
        else:
            raise TodoException

    def set_pause_phase_duration(self, p):
        """

        :param p:
        :return:
        """
        self.mean_pause_phase_duration = p

        p = p + self.pause_phase_rng()

        if p < self.minimum_pause_phase_duration:
            p = self.minimum_pause_phase_duration

        self.pause_phase_duration = p

    def set_growth_phase_duration(self, g):
        """

        :param g:
        :return:
        """
        self.mean_growth_phase_duration = g

        g = g + self.growth_phase_rng()

        if g < self.minimum_growth_phase_duration:
            g = self.minimum_growth_phase_duration

        self.growth_phase_duration = g

    def set_pause_phase_rng(self, func):
        """

        :param func:
        :return:
        """
        raise TodoException

    def set_growth_phase_rng(self, func):
        """

        :param func:
        :return:
        """
        raise TodoException
