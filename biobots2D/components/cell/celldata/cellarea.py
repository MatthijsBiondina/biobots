import torch

from biobots2D.components.cell.abstractcell import AbstractCell
from biobots2D.components.cell.celldata.abstractcelldata import AbstractCellData
from utils.polyshapes import polyarea


class CellArea(AbstractCellData):
    def __init__(self):
        super().__init__()
        self._name = 'cell_area'
        self._data = []

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

    def calculate_data(self, c: AbstractCell):
        """
        Node list must be in order around the cell
        :param c:
        :return:
        """
        x = torch.stack([node.x for node in c.node_list])
        y = torch.stack([node.y for node in c.node_list])
        self.data = polyarea(x, y)
