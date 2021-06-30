from abc import ABC, abstractmethod


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

    def get_data(self, t):
        """

        :param t:
        :return:
        """
        raise NotImplementedError


class SpatialState(AbstractSimulationData):

    def __init__(self):
        """
        Calculates the wiggle ratio
        """
        self._name = 'spatial_state'
        self._data = {}

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, value):
        self._name = value

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, value):
        self._data = value

    def calculate_data(self, t):
        """
        In this case, the data is a structure containing all the node positions, and a list of
        cells containing the nodes that make it up

        At some point I want to add in elements as well, primarily for when membrane is modelled,
        but I'll have to have a separate way of handling a membrane object

        This will only work when every cell has the same number of nodes so it won't work with
        the stromal situation.
        :param t:
        :return:
        """
        raise NotImplementedError
