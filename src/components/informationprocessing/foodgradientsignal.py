from numpy import pi

from src.components.informationprocessing.abstractsignal import AbstractSignal
from src.components.simulation.cuda_memory import CudaMemory
from src.utils.tools import pyout
import cupy as cp


def sigmoid(x: cp.ndarray):
    return 1 / (1 + cp.exp(-x))

class FoodGradientSignal(AbstractSignal):
    def __init__(self):
        super(FoodGradientSignal, self).__init__()

    def add_signal(self, gpu: CudaMemory):
        F_pos = gpu.C_pos[gpu.Ctype_3]
        S_pos = gpu.C_pos[gpu.Ctype_4]

        dmatrix = cp.sum((S_pos[:, None, :] - F_pos[None, :, :]) ** 2, axis=2) ** .5

        # melange = 1 / (pi * (dmatrix + 1) ** 2 - pi * dmatrix ** 2)
        spice_production = dmatrix * gpu.C_inhibitory[gpu.Ctype_4][:,None]
        gpu.spice = sigmoid(cp.sum(spice_production))
