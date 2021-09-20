from biobots3D.memory.gpusimdata import SimulationDataGPU
from biobots3D.physics.cellbasedforces import target_volume_forces
from utils.errors import TodoException
from utils.tools import pyout


class PhysicsEngine3D:
    def __init__(self, gpu: SimulationDataGPU):
        self.gpu = gpu

    def compute_next_timestep(self):
        self.gpu.N_for += target_volume_forces(self.gpu)






