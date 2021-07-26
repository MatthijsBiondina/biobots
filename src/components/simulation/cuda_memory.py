from typing import List

import numpy as np

from src.components.cell.abstractcell import AbstractCell
from src.components.cell.element import Element
from src.components.node.node import Node
from src.utils.tools import pyout


class CudaMemory:
    def __init__(self, cell_list: List[AbstractCell], element_list: List[Element], node_list: List[Node]):
        self.N_pos = np.array([n.position.numpy() for n in node_list])
        self.N_for = np.array([n.force.numpy() for n in node_list])

        pyout()


