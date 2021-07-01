import torch
from numpy import pi

from pybrown.components.node.node import Node
from pybrown.components.simulation.abstractcellsimulation import AbstractCellSimulation
from pybrown.utils.errors import TodoException
from pybrown.utils.polyshapes import nsidedpoly


class FreeCellSimulation(AbstractCellSimulation):

    def __init__(self):
        """
        This uses free cells, i.e. cells that never share elements or nodes with other cells
        """
        self._dt = 0.005
        self._t = 0
        self._step = 0

    @property
    def dt(self):
        return self._dt

    @dt.setter
    def dt(self, value):
        self._dt = value

    @property
    def t(self):
        return self._t

    @t.setter
    def t(self, value):
        self._t = value

    @property
    def step(self):
        return self._step

    @step.setter
    def step(self, value):
        self._step = value

    def make_cell_at_centre(self, N, x, y, ccm):
        """

        :param N:
        :param x:
        :param y:
        :param ccm:
        :return:
        """

        v = nsidedpoly(N, 'radius', 0.5).vertices

        # todo: reorder to MATLAB implementation for ease of debugging (rm)
        while (torch.atan2(v[0, 1], v[0, 0]) % (2 * pi) < 1.5 * pi).item():
            v = torch.cat((v[1:, :], v[0, :].unsqueeze(0)), dim=0)
        # todo: (rm)

        nodes = []
        for ii in range(N):
            nodes.append(Node(v[ii, 0] + x, v[ii, 1] + y, self._get_next_node_id()))

        c = CellFree(ccm, nodes, self._get_next_cell_id())

        print()
        print()
        raise TodoException