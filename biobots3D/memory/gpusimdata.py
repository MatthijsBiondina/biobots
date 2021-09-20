from typing import List

import cupy as cp
from numba import vectorize


class SimulationDataGPU:
    def __init__(self):
        self.B_ids: cp.ndarray = cp.empty(0)  # (#bbots,)
        self.B_2_C: List[cp.ndarray] = []  # [(#bbots, #cells), ...]

        self.C_ids: cp.ndarray = cp.empty(0)  # (#cells,)
        self.C_2_N: List[cp.ndarray] = []
        self.C_2_S: List[cp.ndarray] = []

        self.S_ids: cp.ndarray = cp.empty(0)
        self.S_n_0: cp.ndarray = cp.empty(0)
        self.S_n_1: cp.ndarray = cp.empty(0)
        self.S_n_2: cp.ndarray = cp.empty(0)

        self.E_ids: cp.ndarray = cp.empty(0)
        self.E_n_0: cp.ndarray = cp.empty(0)
        self.E_n_1: cp.ndarray = cp.empty(0)

        self.N_ids: cp.ndarray = cp.empty(0)
        self.N_pos: cp.ndarray = cp.empty(0)

        # dynamic variables
        self.N_for: cp.ndarray = cp.empty(0)
        self.S_are: cp.ndarray = cp.empty(0)  # Surface area
        self.S_crp: cp.ndarray = cp.empty(0)  # Surface Cross Product

    def reset_dynamic_variables(self):
        self.N_for = cp.zeros_like(self.N_pos)
        self.S_crp: cp.ndarray = cp.zeros((self.S_ids.size, 3))
