from abc import ABC

from utils.errors import TodoException


class AbstractSimulation3D(ABC):
    def __init__(self):
        super(AbstractSimulation3D, self).__init__()

        self.next_cell_id = 0
        self.next_surf_id = 0
        self.next_edge_id = 0
        self.next_node_id = 0
