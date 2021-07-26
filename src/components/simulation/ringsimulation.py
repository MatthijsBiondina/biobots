from abc import ABC

from src.components.simulation.abstractcellsimulation import AbstractCellSimulation
from src.components.simulation.simulationdata.centrelineloop import CentreLineLoop


class RingSimulation(AbstractCellSimulation, ABC):

    def __init__(self):
        super(RingSimulation, self).__init__()
        self._step = 0
        self.add_simulation_data(CentreLineLoop())

    @property
    def step(self):
        return self._step

    @step.setter
    def step(self, value):
        self._step = value


