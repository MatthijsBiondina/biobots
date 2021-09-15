from biobots3D.bots.freecell3d import FreeCellBot
from biobots3D.simulation.abstractbiobotsimulation3d import AbstractBiobotSimulation3D
from biobots3D.simulation.abstractsimulation3d import AbstractSimulation3D
from utils.errors import TodoException


class SingleCellSimulation3D(AbstractBiobotSimulation3D):

    def __init__(self, seed: int = 49):
        super(SingleCellSimulation3D, self).__init__()
        self.set_seed(seed)
        self.add_biobot(FreeCellBot(), R=None)
