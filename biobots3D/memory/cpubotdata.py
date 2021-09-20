from typing import List

import numpy as np


class BBotDataCPU:
    def __init__(self):
        self.cell_ids = np.empty(0)  # (#cells,)
        self.cell_2_N: List[object] = []  # (#cells, #nodes)
        self.cell_2_S: List[object] = []  # (#cells, #surfs)

        self.node_ids = np.empty(0)  # (#nodes,)
        self.node_pos = np.empty(0)  # (#nodes, 3)

        self.edge_ids = np.empty(0)  # (#edges,)
        self.edge_n_0 = np.empty(0)  # (#edges,)
        self.edge_n_1 = np.empty(0)  # (#edges,)

        self.surf_ids = np.empty(0)  # (#surfs,)
        self.surf_n_0 = np.empty(0)  # (#surfs,)
        self.surf_n_1 = np.empty(0)  # (#surfs,)
        self.surf_n_2 = np.empty(0)  # (#surfs,)

    def increment_cell_ids(self, val):
        self.cell_ids += val

        return val + self.cell_ids.size

    def increment_surf_ids(self, val):
        self.surf_ids += val

        return val + self.surf_ids.size

    def increment_edge_ids(self, val):
        self.edge_ids += val

        return val + self.edge_ids.size

    def increment_node_ids(self, val):
        self.node_ids += val
        self.edge_n_0 += val
        self.edge_n_1 += val
        self.surf_n_0 += val
        self.surf_n_1 += val
        self.surf_n_2 += val
        for c2n in self.cell_2_N:
            c2n += val

        return self.node_ids.size
