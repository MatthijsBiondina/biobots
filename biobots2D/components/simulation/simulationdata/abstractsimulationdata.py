from abc import ABC, abstractmethod

from biobots2D.components.simulation.abstractcellsimulation import AbstractCellSimulation


class AbstractSimulationData(ABC):
    def __init__(self):
        """
        This class sets out the required functions for working out various types of data that
        can be extracted from the simulation

        This will usually be a statistic that summarises the simulation state, like the wiggle
        ratio. Things as simple as the number of cells could be implemented here, but they are
        fundamental enough that they can be implemented in the lowest level i.e.
        AbstractCellSimulation. Anything that has a limited scope of simulations where it is
        useful/defined should be implemented here, as well as anything that might be stored

        This will often be closely linked to a DataStore object

        The intention is that the data will only be calculated on demand so, for instance,
        if we only calculate to store it, then it will only be calculated at the sampling
        multiple (obviously this is not suitable for data that depends on previous time steps).
        If we put the calculate operation in the get operation, then we might end up calculating
        multiple times per time step if the data is used in multiple places. To avoid this,
        we add in a time stamp for when the data was last calculated
        """
        self.time_stamp = -1

    @property
    @abstractmethod
    def name(self):
        """
        A unique identifier so the data can be accessed in a way that has a meaningful
        interpretation instead of an index
        :return:
        """
        pass

    @property
    @abstractmethod
    def data(self):
        """
        A structure that holds the data within a timestep
        :return:
        """
        pass

    @abstractmethod
    def calculate_data(self, t):
        """
        This method must return data
        :param t:
        :return:
        """
        pass

    def get_data(self, t: AbstractCellSimulation):
        """

        :param t:
        :return:
        """

        if self.time_stamp == t.t:
            return self.data
        else:
            self.time_stamp = t.t
            self.calculate_data(t)
            return self.data

