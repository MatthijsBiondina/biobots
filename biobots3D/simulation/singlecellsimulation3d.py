import numpy as np
from numpy import sqrt

from biobots3D.bots.freecell3d import FreeCellBot
from biobots3D.simulation.abstractbiobotsimulation3d import AbstractBiobotSimulation3D
from biobots3D.simulation.abstractsimulation3d import AbstractSimulation3D
from utils.errors import TodoException


class SingleCellSimulation3D(AbstractBiobotSimulation3D):

    def __init__(self, seed: int = 49):
        super(SingleCellSimulation3D, self).__init__(seed)

    def load_simulation_objects(self, *args):
        self.add_biobot(FreeCellBot())
        self.add_biobot(FreeCellBot(), pos=np.array([5., 0., 0.]))
