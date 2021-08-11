import numpy as np
import torch
from numpy import pi

from src.components.cell.celldb.ciliacell import CiliaCell
from src.components.cell.celldb.epithelialcell import EpithelialCell
from src.components.cell.celldb.foodcell import FoodCell
from src.components.cell.celldb.heartcell import HeartCell
from src.components.cell.celldb.sensorcell import SensorCell
from src.components.central_memory.cpu_memory import CPUMemory
from src.components.forces.cellbasedforce.ciliapropagationforce import CiliaPropagationForce
from src.components.forces.cellbasedforce.freecellperimeternormalisingforce import \
    FreeCellPerimeterNormalisingForce
from src.components.forces.cellbasedforce.polygoncellgrowthforce import PolygonCellGrowthForce
from src.components.forces.neighbourhoodbasedforce.cellcellinteractionforce import \
    CellCellInteractionForce
from src.components.informationprocessing.foodgradientsignal import FoodGradientSignal
from src.components.node.node import Node
from src.components.simulation.cuda_memory import CudaMemory
from src.components.simulation.freecellsimulation import FreeCellSimulation
from src.utils.polyshapes import nsidedpoly


class Gradient(FreeCellSimulation):
    def __init__(self, t0=10, seed: int = 49):
        super(Gradient, self).__init__()
        self.set_rng_seed(seed)
        self.N = 12

        # e_cent = self.new_cell(0.0, 0.5)

        c_left = self.new_cell(-1., -.5, 'cilia', inh=True, ang=0.25 * pi)
        s_left = self.new_cell(-1., 0.5, 'sensor', inh=False)

        bl = self.new_cell(0, -.5)
        tl = self.new_cell(0, .5)

        c_right = self.new_cell(1., -.5, 'cilia', inh=False, ang=-0.25 * pi)
        s_right = self.new_cell(1., 0.5, 'sensor', inh=True)

        self.connect_cells(c_left, s_left)
        self.connect_cells(bl, tl)
        self.connect_cells(c_right, s_right)

        self.connect_cells(c_left, bl)
        self.connect_cells(bl, c_right)
        self.connect_cells(s_left, tl)
        self.connect_cells(tl, s_right)
        # self.connect_cells(c_right, c_left)
        # self.connect_cells(s_left, s_right)

        self.new_cell(5 * 1.5, 5 * 1.5)
        self.new_cell(-5 * 1.5, 5 * 1.5)
        self.new_cell(-5 * 1.5, -5 * 1.5)
        self.new_cell(5 * 1.5, -5 * 1.5, ctype='food')

        for c in self.cell_list:
            self.node_list += c.node_list
            self.element_list += c.element_list
        self.node_list = sorted(list(set(self.node_list)), key=lambda x: x.id)

        """ADD THE FORCES"""

        self.add_cell_based_force(PolygonCellGrowthForce(area_P=50, perimeter_P=10, tension_P=10))
        self.add_cell_based_force(FreeCellPerimeterNormalisingForce(spring_rate=15))
        self.add_cell_based_force(CiliaPropagationForce(propagation_magnitude=10))

        self.add_neighbourhood_based_force(CellCellInteractionForce(sra=10, srr=10, da=-0.1,
                                                                    ds=0.1, dl=0.2, dt=self.dt,
                                                                    using_polys=True))

        self.add_information_processing_signal(FoodGradientSignal())

        """init memory"""
        self.gpu = CudaMemory(self.cell_list, self.element_list, self.node_list, 0.2)
        # self.gpu = CPUMemory(self.cell_list, self.element_list, self.node_list, 0.2, self.gpu)

    def new_cell(self, dx, dy, ctype='epithelial', ang=0, inh=None):

        v = nsidedpoly(self.N, 'radius', 0.5).vertices.numpy()

        ang = np.array([[np.cos(2 * pi - ang), -np.sin(2 * pi - ang)],
                        [np.sin(2 * pi - ang), np.cos(2 * pi - ang)]])
        v = v @ ang

        nodes = []
        for ii in range(self.N):
            nodes.append(Node(v[ii, 0] + dx, v[ii, 1] + dy, self._get_next_node_id()))
        element_idxs = [self._get_next_element_id() for _ in range(self.N)]
        if ctype == 'epithelial':
            c = EpithelialCell(nodes, element_idxs, self._get_next_cell_id())
        elif ctype == 'heartcell':
            c = HeartCell(nodes, element_idxs, self._get_next_cell_id())
        elif ctype == 'cilia':
            c = CiliaCell(nodes, element_idxs, self._get_next_cell_id(), inh=inh)
        elif ctype == 'food':
            c = FoodCell(nodes, element_idxs, self._get_next_cell_id())
        elif ctype == 'sensor':
            c = SensorCell(nodes, element_idxs, self._get_next_cell_id(), inh=inh)
        else:
            raise ValueError(f"{ctype} is not a valid cell type.")

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
            E1.pointing_forward = None
            E2.pointing_forward = None
            e_lst.append(E1)
            e_lst.append(E2)

            c2.element_list[(e2_idx - 1) % self.N].node_2 = E1.node_2
            E2.node_1 = E1.node_2
            E2.node_2 = E1.node_1
            c2.element_list[(e2_idx + 1) % self.N].node_1 = E1.node_1

            c2.node_list = []
            for e in c2.element_list:
                c2.node_list.append(e.node_1)
