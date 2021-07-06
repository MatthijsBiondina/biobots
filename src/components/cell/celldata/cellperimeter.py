from src.components.cell.abstractcell import AbstractCell
from src.components.cell.celldata.abstractcelldata import AbstractCellData
from src.utils.errors import TodoException


class CellPerimeter(AbstractCellData):


    def __init__(self):
        super().__init__()
        self._name = 'cell_perimeter'
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
        self.data = sum(e.get_length() for e in c.element_list)
