import torch
from torch import tensor

from biobots2D.components.simulation.abstractcellsimulation import AbstractCellSimulation
from biobots2D.components.simulation.simulationdata.abstractsimulationdata import \
    AbstractSimulationData


class SpatialState(AbstractSimulationData):

    def __init__(self):
        """
        Calculates the wiggle ratio
        """
        super().__init__()
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

    def calculate_data(self, t: AbstractCellSimulation):
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

        node_data = torch.stack(([tensor([n.id, n.x, n.y]) for n in t.node_list]), dim=0)

        element_data = []
        for e in t.element_list:
            element_data.append([e.node_1.id, e.node_2.id])
        element_data = tensor(element_data)

        cell_data = []
        for c in t.cell_list:
            nL = c.node_list
            l = len(nL)
            cell_data.append([l, *[n.id for n in nL], c.cell_cycle_model.colour])

        self.data = {'node_data': node_data, 'element_data': element_data, 'cell_data': cell_data}
