from abc import ABC, abstractmethod

from pybrown.utils.errors import TodoException


class AbstractCellData(ABC):
    def __init__(self):
        """
        This class sets out the required functions for working out various types of data that can
        be extracted from the a cell

        This will usually be a statistic that summarises the cell state like the area. It
        shouldn't be used for things that don't really need calculating

        The intention is that the data will only be calculated on demand so, for instance,
        if we only calculate to store it, then it will only be calculated at the sampling
        multiple (obviously this is not suitable for data that depends on previous time steps).
        If we put the calculate operation in the get operation, then we might end up calculating
        multiple times per time step if the data is used in multiple places. To avoid this,
        we add in an age stamp for when the data was last calculated

        This class inherits from matlab.mixin.Copyable so the cellData can be copied and
        transfered to a daughter cell upon division
        """
        self.age_stamp = -1

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

    def get_data(self, c):
        if self.age_stamp == c.get_age():
            return self.data
        else:
            self.age_stamp = c.get_age()
            self.calculate_data(c)
            return self.data


class ElementNeighbours(AbstractCellData):
    def __init__(self):
        """
        Calculates the wiggle ratio
        """
        self._name = 'element_neighbours'
        self._data = []
        self.radius = 0.1

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

    def element_neighbours(self):
        """
        Can give a radius or just use the default
        :return:
        """
        raise TodoException

    def calculate_data(self, t):
        """
        Node list must be in order around the cell

        :param t:
        :return:
        """
        raise TodoException


if __name__ == '__main__':
    print("nodedata.py")
