from biobots2D.components.simulation.simulationdata.abstractsimulationdata import AbstractSimulationData
from biobots2D.utils.errors import TodoException


class CentreLineLoop(AbstractSimulationData):
    def __init__(self):
        super(CentreLineLoop, self).__init__()
        self._name = "centre_line"
        self._data = []

    @property
    def name(self):
        return self._name

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, value):
        self._data = value

    def calculate_data(self, t):
        """
        Makes a sequence of points that defines the centre line of the cells
        :param t:
        :return:
        """
        raise TodoException
