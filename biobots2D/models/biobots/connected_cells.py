from random import random

import torch

from biobots2D.components.cell.celldb.epithelialcell import EpithelialCell
from biobots2D.components.cell.celldb.heartcell import HeartCell
from biobots2D.components.central_memory.cpu_memory import CPUMemory
from biobots2D.components.forces.cellbasedforce.freecellperimeternormalisingforce import \
    FreeCellPerimeterNormalisingForce
from biobots2D.components.forces.cellbasedforce.polygoncellgrowthforce import PolygonCellGrowthForce
from biobots2D.components.forces.neighbourhoodbasedforce.cellcellinteractionforce import \
    CellCellInteractionForce
from biobots2D.components.node.node import Node
from biobots2D.components.simulation.cuda_memory import CudaMemory
from biobots2D.components.simulation.freecellsimulation import FreeCellSimulation
from biobots2D.utils.polyshapes import nsidedpoly
from biobots2D.utils.tools import pyout


class ConnectedCells(FreeCellSimulation):
    def __init__(self, seed: int = 49):
        super().__init__()
        self.set_rng_seed(seed)
        self.N = 12

        cells = self.build_grid_of_cells(8, 8)
        self.densely_connect_cells(cells)

        for c in self.cell_list:
            self.node_list += c.node_list
            self.element_list += c.element_list
        self.node_list = sorted(list(set(self.node_list)), key=lambda x: x.id)

        """ADD THE FORCES"""

        self.set_rng_seed(seed)

        # Cell growth force
        self.add_cell_based_force(PolygonCellGrowthForce(area_P=20, perimeter_P=10, tension_P=10))

        # Node-Element interaction force - requires a SpacePartition
        self.add_neighbourhood_based_force(CellCellInteractionForce(sra=10, srr=10, da=-0.1,
                                                                    ds=0.1, dl=0.2, dt=self.dt,
                                                                    using_polys=True))

        # Tries to make the edges the same length
        self.add_cell_based_force(FreeCellPerimeterNormalisingForce(spring_rate=10))

        self.gpu = CudaMemory(self.cell_list, self.element_list, self.node_list, 0.2)

    def build_grid_of_cells(self, nr_of_rows, nr_of_columns):
        cells = []

        for row in range(nr_of_rows):
            cells.append([])
            for col in range(nr_of_columns):
                if row % 2 == 0:
                    cells[-1].append(self.new_cell(col, row, 'epithelial'))
                else:
                    cells[-1].append(self.new_cell(col + .5, row, 'epithelial'))
        return cells

    def densely_connect_cells(self, cells):
        for row in range(len(cells)):
            for col in range(len(cells[row])):
                if not row == len(cells) - 1:  # if not on last row
                    self.connect_cells(cells[row][col], cells[row + 1][col])
                    if row % 2 == 0 and not col == 0:  # diagonal left
                        self.connect_cells(cells[row][col], cells[row + 1][col - 1])
                    if not row % 2 == 0 and not col == len(cells[row]) - 1:  # diagonal right
                        self.connect_cells(cells[row][col], cells[row + 1][col + 1])
                if not col == len(cells[col]) - 1:  # if not on last column
                    self.connect_cells(cells[row][col], cells[row][col + 1])

        pyout()

    def new_cell(self, dx, dy, ctype='epithelial'):
        v = nsidedpoly(self.N, 'radius', 0.5).vertices
        nodes = []
        for ii in range(self.N):
            nodes.append(Node(v[ii, 0] + dx, v[ii, 1] + dy, self._get_next_node_id()))
        element_idxs = [self._get_next_element_id() for _ in range(self.N)]

        ctype = 'epithelial' if random() < 0.4 else 'heartcell'

        if ctype == 'epithelial':
            c = EpithelialCell(nodes, element_idxs, self._get_next_cell_id())
        if ctype == 'heartcell':
            c = HeartCell(nodes, element_idxs, self._get_next_cell_id())

        self.cell_list.append(c)

        return c

    def connect_cells(self, c2, c1):
        E1, E2 = None, None
        e1_idx, e2_idx = 0, 0
        e_lst = []

        for _ in range(1):
            mindist = float('inf')
            for ii, e1 in enumerate(c1.element_list):

                for jj, e2 in enumerate(c2.element_list):
                    if e1 in e_lst or e2 in e_lst:
                        continue
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
            e_lst.append(E1)
            e_lst.append(E2)

            c2.element_list[(e2_idx - 1) % self.N].node_2 = E1.node_2
            E2.node_1 = E1.node_2
            E2.node_2 = E1.node_1
            c2.element_list[(e2_idx + 1) % self.N].node_1 = E1.node_1

            c2.node_list = []
            for e in c2.element_list:
                c2.node_list.append(e.node_1)
