from biobots3D.bots.abstractbot3d import AbstractBot3D
from biobots3D.cells.epithelialcell3d import EpithelialCell3D
from utils.errors import TodoException
from utils.tools import pyout


class FreeCellBot(AbstractBot3D):

    def __init__(self):
        super(FreeCellBot, self).__init__()

    def construct(self, **kwargs):
        self.add_cell(EpithelialCell3D(), (0, 0, 0))
        self.add_cell(EpithelialCell3D(), (2, 0, 0))

    def connect_cells(self, **kwargs):
        pass
