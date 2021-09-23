import sys

from tqdm import tqdm

from biobots3D.bots.abstractbot3d import AbstractBot3D
from biobots3D.cells.epithelialcell3d import EpithelialCell3D
from utils.errors import TodoException
from utils.tools import pyout


class FreeCellBot(AbstractBot3D):

    def __init__(self):
        super(FreeCellBot, self).__init__()

    def construct(self, **kwargs):
        nr_of_cells = 1
        n = 0

        # pbar = tqdm(total=nr_of_cells, leave=False)
        try:
            for y in range(sys.maxsize):
                if y % 2 == 0:
                    for x in [-2, 0, 2]:
                        for z in [-2, 0, 2]:
                            self.add_cell(EpithelialCell3D(pos=(x, y, z)))
                            n += 1
                            # pbar.update(1)
                            if n >= nr_of_cells:
                                raise Exception

                else:
                    for x in [-1, 1]:
                        for z in [-1, 1]:
                            self.add_cell(EpithelialCell3D(pos=(x, y, z)))
                            n += 1
                            # pbar.update(1)
                            if n >= nr_of_cells:
                                raise Exception
        except Exception:
            pass

        # self.add_cell(EpithelialCell3D(pos=(0, 0, 0)))
        # self.add_cell(EpithelialCell3D(pos=(2, 0, 0)))
        # self.add_cell(EpithelialCell3D(pos=(0, 0, 2)))
        # self.add_cell(EpithelialCell3D(pos=(2, 0, 2)))
        # self.add_cell(EpithelialCell3D(pos=(1, 1, 1)))
        # self.add_cell(EpithelialCell3D(pos=(1, -1, 1)))

    def connect_cells(self, **kwargs):
        pass
