import numpy as np
import torch
from numpy import pi

from biobots2D.components.cell.abstractcell import AbstractCell
from biobots2D.components.cell.celldata.abstractcelldata import AbstractCellData
from biobots2D.utils.errors import TodoException


class TargetPerimeter(AbstractCellData):
    def __init__(self):
        super().__init__()
        self._name = 'target_perimeter'
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
        See equation 31
        :param c:
        :return:
        """
        # Node list must be in order around the cell

        target_area = c.cell_data['target_area'].get_data(c)

        #  Assume the cell wants to be a regular polygon
        #  Use the number of elements to decide what type
        #
        #  From https://en.wikipedia.org/wiki/Regular_polygon#Area the perimeter
        #  of a regular polygon for a given area is
        #  p * a = 2A
        #  Where A is the area, and a is the apothem (the line from the centre to
        #  the mid point of an edge)
        #  The formula for a = p / 2ntan(pi/n)
        #  2A = p^2/2ntan(pi/n)
        #  p = sqrt(4Antan(pi/n))

        n = len(c.element_list)

        self.data = (4 * target_area * n * np.tan(pi / n)) ** .5
