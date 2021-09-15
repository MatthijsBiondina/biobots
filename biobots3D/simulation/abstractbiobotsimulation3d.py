from abc import ABC, abstractmethod

from biobots3D.simulation.abstractsimulation3d import AbstractSimulation3D
from utils.errors import TodoException


class AbstractBiobotSimulation3D(AbstractSimulation3D, ABC):
    def __init__(self):
        super(AbstractBiobotSimulation3D, self).__init__()
