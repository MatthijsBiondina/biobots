from typing import List

import cupy as cp


class SimulationDataGPU:
    def __init__(self):
        self.B_ids: cp.ndarray = cp.empty(0)  # (#bbots,)
        self.B_2_C: List[cp.ndarray] = []  # [(#bbots, #cells), ...]

        self.C_ids: cp.ndarray = cp.empty(0)  # (#cells,)
        self.C_2_N: List[cp.ndarray] = []

        self.S_ids: cp.ndarray = cp.empty(0)
        self.S_n_0: cp.ndarray = cp.empty(0)
        self.S_n_1: cp.ndarray = cp.empty(0)
        self.S_n_2: cp.ndarray = cp.empty(0)

        self.E_ids: cp.ndarray = cp.empty(0)
        self.E_n_0: cp.ndarray = cp.empty(0)
        self.E_n_1: cp.ndarray = cp.empty(0)

        self.N_ids: cp.ndarray = cp.empty(0)
        self.N_pos: cp.ndarray = cp.empty(0)

        self.N_pos = cp.empty(0)
