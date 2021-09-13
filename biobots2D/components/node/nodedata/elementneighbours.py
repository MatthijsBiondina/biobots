from biobots2D.components.node.nodedata.abstractnodedata import AbstractNodeData
from utils import TodoException


class ElementNeighbours(AbstractNodeData):
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
