from biobots3D.memory.gpusimdata import SimulationDataGPU
from biobots3D.physics.cellbasedforces import target_volume_forces
from biobots3D.simulation.parameters import DT
from utils.errors import TodoException
from utils.tools import pyout


class PhysicsEngine3D:
    def __init__(self, gpu: SimulationDataGPU):
        self.gpu = gpu

    def compute_next_timestep(self):
        self.gpu.reset_dynamic_variables()

        self.__compute_forces()

        self.__update_nodes()

        # pyout()

    def __compute_forces(self):
        target_volume_forces(self.gpu)

    def __update_nodes(self):
        self.gpu.N_pos += DT * self.gpu.N_for

        # pyout()