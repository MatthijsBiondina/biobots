from abc import ABC, abstractmethod

from biobots2D.components.cell.abstractcell import AbstractCell


class AbstractCellData(ABC):
    def __init__(self):
        """
        This class sets out the required functions for working out various types of data that can
        be extracted from a cell

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

        # The last time point when the data was calculated. Saves calculating repeatedly in a
        # single time step
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
        A structure that holds the data within a timestamp
        :return:
        """
        pass

    @abstractmethod
    def calculate_data(self, c):
        """
        This method must return data
        :param c:
        :return:
        """
        pass

    def get_data(self, c: AbstractCell):
        if self.age_stamp == c.get_age():
            return self.data
        else:
            self.age_stamp = c.get_age()
            self.calculate_data(c)
            return self.data