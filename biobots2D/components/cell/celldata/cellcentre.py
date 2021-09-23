from biobots2D.components.cell.celldata.abstractcelldata import AbstractCellData
from utils.errors import TodoException


class CellCentre(AbstractCellData):
    def __init__(self):
        super().__init__()
        self._name = 'cell_centre'
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

    def calculate_data(self, c):
        """
        Node list must be in order around the cell
        :param c:
        :return:
        """
        raise TodoException