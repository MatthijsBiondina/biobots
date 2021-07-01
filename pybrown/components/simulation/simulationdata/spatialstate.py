from pybrown.components.simulation.simulationdata.abstractsimulationdata import \
    AbstractSimulationData


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