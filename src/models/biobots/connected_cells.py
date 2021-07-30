import torch

from src.components.cell.celldb.epithelialcell import EpithelialCell
from src.components.forces.cellbasedforce.freecellperimeternormalisingforce import \
    FreeCellPerimeterNormalisingForce
from src.components.forces.cellbasedforce.polygoncellgrowthforce import PolygonCellGrowthForce
from src.components.forces.neighbourhoodbasedforce.cellcellinteractionforce import \
    CellCellInteractionForce
from src.components.node.node import Node
from src.components.simulation.cuda_memory import CudaMemory
from src.components.simulation.freecellsimulation import FreeCellSimulation
from src.utils.polyshapes import nsidedpoly
from src.utils.tools import pyout


class ConnectedCells(FreeCellSimulation):
    def __init__(self, t0: float = 10, seed: int = 49):
        super().__init__()
        self.set_rng_seed(seed)
        self.N = 4

        # Make first cell
        c1 = self.new_cell(0, 0)
        c2 = self.new_cell(1, 0)
        c3 = self.new_cell(0, 1)

        self.connect_cells(c1, c2)
        self.connect_cells(c1, c3)

        for c in self.cell_list:
            self.node_list += c.node_list
            self.element_list += c.element_list
        self.node_list = sorted(list(set(self.node_list)), key=lambda x: x.id)

        """ADD THE FORCES"""

        # Cell growth force
        self.add_cell_based_force(PolygonCellGrowthForce(area_P=20, perimeter_P=10, tension_P=1))

        # Node-Element interaction force - requires a SpacePartition
        self.add_neighbourhood_based_force(CellCellInteractionForce(sra=10, srr=10, da=-0.1,
                                                                    ds=0.1, dl=0.2, dt=self.dt,
                                                                    using_polys=True))

        # Tries to make the edges the same length
        self.add_cell_based_force(FreeCellPerimeterNormalisingForce(spring_rate=5))

        self.gpu = CudaMemory(self.cell_list, self.element_list, self.node_list)

    def new_cell(self, dx, dy):
        v = nsidedpoly(self.N, 'radius', 0.5).vertices
        nodes = []
        for ii in range(self.N):
            nodes.append(Node(v[ii, 0] + dx, v[ii, 1] + dy, self._get_next_node_id()))
        element_idxs = [self._get_next_element_id() for _ in range(self.N)]
        c = EpithelialCell(nodes, element_idxs, self._get_next_cell_id())

        self.cell_list.append(c)

        return c

    def connect_cells(self, c1, c2):
        E1, E2 = None, None
        e1_idx, e2_idx = 0, 0
        mindist = float('inf')
        for ii, e1 in enumerate(c1.element_list):
            for jj, e2 in enumerate(c2.element_list):
                dist1 = torch.sum((e1.node_1.position - e2.node_2.position) ** 2)
                dist2 = torch.sum((e1.node_2.position - e2.node_1.position) ** 2)
                dist = (dist1 + dist2) ** .5
                if dist < mindist:
                    E1 = e1
                    E2 = e2
                    e1_idx = ii
                    e2_idx = jj

                    mindist = dist

        E1.internal = True
        E2.internal = True

        c2.element_list[(e2_idx-1) % self.N].node_2 = E1.node_2
        E2.node_1 = E1.node_2
        E2.node_2 = E1.node_1
        c2.element_list[(e2_idx+1) % self.N].node_1 = E1.node_1

        c2.node_list = []
        for e in c2.element_list:
            c2.node_list.append(e.node_1)
