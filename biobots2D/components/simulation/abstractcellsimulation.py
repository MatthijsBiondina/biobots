from abc import abstractmethod, ABC
import random
from typing import List, Union

import numpy
import cupy as cp
import torch
from torch import tensor
from tqdm import tqdm

from biobots2D.components.cell.abstractcell import AbstractCell
from biobots2D.components.cell.celldeath.abstractcellkiller import AbstractCellKiller
from biobots2D.components.cell.celldeath.abstracttissuelevelcellkiller import \
    AbstractTissueLevelCellKiller
from biobots2D.components.cell.element import Element
from biobots2D.components.forces.cellbasedforce.abstractcellbasedforce import AbstractCellBasedForce
from biobots2D.components.forces.elementbasedforce.abstractelementbasedforce import \
    AbstractElementBasedForce
from biobots2D.components.forces.neighbourhoodbasedforce.abstractneighbourhoodbasedforce import \
    AbstractNeighbourhoodBasedForce
from biobots2D.components.forces.tissuebasedforce.abstracttissuebasedforce import \
    AbstractTissueBasedForce
from biobots2D.components.informationprocessing.abstractsignal import AbstractSignal
from biobots2D.components.node.node import Node
from biobots2D.components.simulation.cuda_memory import CudaMemory
from biobots2D.components.simulation.datastore.abstractdatastore import AbstractDataStore
from biobots2D.components.simulation.datawriter.abstractdatawriter import AbstractDataWriter
from biobots2D.components.simulation.modifiers.abstractsimulationmodifier import \
    AbstractSimulationModifier
from biobots2D.components.simulation.stopping.abstractstoppingcondition import \
    AbstractStoppingCondition
from biobots2D.components.spacepartition import SpacePartition
from utils.errors import TodoException
from utils.plotting import Renderer

from utils.tools import prng


class AbstractCellSimulation(ABC):

    def __init__(self):
        """
        A parent class that contains all the functions for running a simulation. The child/concrete
        class will only need a constructor that assembles the cells
        """
        super().__init__()

        self.seed = None
        self.node_list: List[Node] = []
        self.next_node_id = 0
        self.element_list = []
        self.next_element_id = 0
        self.cell_list: List[AbstractCell] = []
        self.next_cell_id = 0
        self.gpu: Union[None, CudaMemory] = None

        self.stochastic_jiggle = True  # Brownian noise?
        self.epsilon = 0.0001  # Size of the jiggle force

        self.cell_based_forces: List[AbstractCellBasedForce] = []
        self.element_based_forces: List[AbstractElementBasedForce] = []
        self.neighbourhood_based_forces: List[AbstractNeighbourhoodBasedForce] = []
        self.tissue_based_forces: List[AbstractTissueBasedForce] = []
        self.information_processing_signals: List[AbstractSignal] = []

        self.stopping_conditions: List[AbstractStoppingCondition] = []

        self.stopped = False

        self.tissue_level_killers: List[AbstractTissueLevelCellKiller] = []
        self.cell_killers: List[AbstractCellKiller] = []

        self.simulation_modifiers: List[AbstractSimulationModifier] = []

        # A collection of objects that store data over multiple time steps with also the
        # potential to write to file
        self.data_stores: List[AbstractDataStore] = []
        self.data_writers: List[AbstractDataWriter] = []

        # A collection of objects for calculating data about the simulation stored in a map
        # container so each type of data can be given a meaningful name
        self.sim_data = {}

        self.boxes: SpacePartition = None

        self.using_boxes = True

        self.write_to_file = True

        # placeholders
        self.epsilon_cuda = cp.float32(self.epsilon)
        self.zero_point_five = cp.float32(0.5)
        self.dt_cuda = cp.float32(self.dt)

    @property
    @abstractmethod
    def dt(self):
        pass

    @property
    @abstractmethod
    def t(self):
        pass

    @property
    @abstractmethod
    def step(self):
        pass

    def set_rng_seed(self, seed):
        """

        :param seed:
        :return:
        """
        self.seed = seed

        # todo: No need for this as long as we use our custom pseudo-random generator
        random.seed(seed)
        numpy.random.seed(seed)
        torch.manual_seed(seed)

    def next_time_step(self):
        """
        Updates all the forces and applies the movements
        :return:
        """
        self.generate_information_processing_signals()



        # self.generate_tissue_based_forces()
        self.generate_cell_based_forces()
        self.generate_element_based_forces()

        if self.using_boxes:
            self.generate_neighbourhood_based_forces()

        self.make_nodes_move()

        # Division must occur after movement
        # self.make_cells_divide()

        # self.kill_cells()

        self.modify_simulation_state()

        self.make_cells_age()

        self.step += 1
        self.t = self.step * self.dt

        # self.store_data()
        #
        # if self.write_to_file:
        #     self.write_data()

        if self.is_stopping_condition_met():
            self.stopped = True

        self.ccd(self.gpu)

        self.gpu.clear_dynamic_memory(self.t)

    def n_time_steps(self, n):
        """
        Advances a set number of time steps
        :param n:
        :return:
        """

        for ii in range(n):
            # Do all the calculations
            # t0 = time.time()
            self.next_time_step()
            # pyout(time.time() - t0)

            # if self.step % 1000 == 0:
            #     print(f"Time = {self.t:.3f} hours")

            if self.stopped:
                print(f"Stopping condition met at t={self.t:.3f}")
                break

    def run_to_time(self, t):
        """
        Given a time, run the simulation until we reach said time
        :param t:
        :return:
        """
        raise TodoException

    def generate_tissue_based_forces(self):
        """

        :return:
        """
        for force in self.tissue_based_forces:
            force.add_tissue_based_forces(self)

    def generate_cell_based_forces(self):
        """

        :return:
        """
        for force in self.cell_based_forces:
            force.add_cell_based_forces(self.cell_list, self.gpu)

    def generate_element_based_forces(self):
        """

        :return:
        """
        for force in self.element_based_forces:
            force.add_element_based_forces(self.element_list)

    def generate_neighbourhood_based_forces(self):
        """

        :return:
        """
        if self.boxes is None:
            ValueError("Space partition required for NeighbourhoodForces, but none set")

        for force in self.neighbourhood_based_forces:
            force.add_neighbourhood_based_forces(self.node_list, self.boxes, self.gpu)

    def generate_information_processing_signals(self):
        for signal in self.information_processing_signals:
            signal.add_signal(self.gpu)

    def make_nodes_move(self):
        """

        :return:
        """
        if self.gpu.EXEC_CPU:
            for n in self.node_list:
                eta = n.eta
                force = n.force

                if self.stochastic_jiggle:
                    # Add in a tiny amount of stochasticity to the force calculation to nudge it
                    # out
                    # of unstable equilibria

                    # Make a random direction vector
                    v = tensor([prng() - 0.5, prng() - 0.5])
                    v = v / v.norm()

                    # Add the random vector, and make sure that it is orders of magnitude smaller
                    # than the actual force
                    force += v * self.epsilon

                new_position = n.position + self.dt / eta * force

                n.move_node(new_position)

                if self.using_boxes:
                    self.boxes.update_box_for_node(n)

        self.make_nodes_move_cuda()

    def make_nodes_move_cuda(self):
        if self.stochastic_jiggle:
            # Add in a tiny amount of stochasticity to the force calculation to nudge it out
            # of unstable equilibria

            # Make a random direction vector
            v = cp.random.random(self.gpu.N_pos.shape) - self.zero_point_five
            v /= cp.linalg.norm(v, axis=1)[:, None]

            # Add the random vector, and make sure that it is orders of magnitude smaller
            # than the actual force
            self.gpu.N_for += v * self.epsilon_cuda

        self.gpu.N_pos_previous = cp.copy(self.gpu.N_pos)
        self.gpu.N_for_previous = cp.copy(self.gpu.N_for)

        self.gpu.N_pos += self.dt_cuda / self.gpu.N_eta[:, None] * self.gpu.N_for
        self.gpu.N_for = cp.zeros_like(self.gpu.N_for_previous)

    def adjust_node_position(self, n, new_pos):
        """
        Only used by modifiers. Do not use to progress the simulation

        This will move a node to a given position regardless of forces, but after all force
        movement has happened

        Previous position and previous force are not modified
        :param n:
        :param new_pos:
        :return:
        """
        raise TodoException

    def make_cells_divide(self):
        """
        Call the divide process, and update the lists
        :return:
        """
        new_cells: List[AbstractCell] = []
        new_elements: List[Element] = []
        new_nodes: List[Node] = []
        for c in self.cell_list:
            if c.is_ready_to_divide():
                raise TodoException

        self.add_new_cells(new_cells, new_elements, new_nodes)

    def add_new_cells(self, new_cells, new_elements, new_nodes):
        """
        When a cell divides, need to make sure the new cell object as well as the new elements
        and nodes are correctly added to their respective lists and boxes if relevant
        :param new_cells:
        :param new_elements:
        :param new_nodes:
        :return:
        """
        for n in new_nodes:
            raise TodoException
            n.id = self._get_next_node_id()
            if self.using_boxes:
                self.boxes.put_node_in_box(n)

        for e in new_elements:
            raise TodoException
            # Debug: element id points to nodes rather than own id
            # e.id = self._get_next_element_id()
            if self.using_boxes and not e.internal:
                self.boxes.put_element_in_boxes(e)

        for nc in new_cells:
            raise TodoException

        self.cell_list.extend(new_cells)
        self.element_list.extend(new_elements)
        self.node_list.extend(new_nodes)

    def make_cells_age(self):
        """

        :return:
        """
        if self.gpu.EXEC_CPU:
            for c in self.cell_list:
                c.age_cell(self.dt)

        self.gpu.C_age += self.dt_cuda

    def add_cell_based_force(self, f: AbstractCellBasedForce):
        """

        :param f:
        :return:
        """
        self.cell_based_forces.append(f)

    def add_element_based_force(self, f):
        """

        :param f:
        :return:
        """
        raise TodoException

    def add_neighbourhood_based_force(self, f: AbstractNeighbourhoodBasedForce):
        """

        :param f:
        :return:
        """
        self.neighbourhood_based_forces.append(f)

    def add_tissue_based_force(self, f):
        """

        :param f:
        :return:
        """
        raise TodoException

    def add_information_processing_signal(self, s: AbstractSignal):
        self.information_processing_signals.append(s)

    def add_tissue_level_killer(self, k):
        """

        :param k:
        :return:
        """
        raise TodoException

    def add_cell_killer(self, k):
        """

        :param k:
        :return:
        """
        raise TodoException

    def add_stopping_condition(self, s):
        """

        :param s:
        :return:
        """
        raise TodoException

    def add_simulation_modifier(self, m):
        """

        :param m:
        :return:
        """
        raise TodoException

    def add_data_store(self, d):
        """

        :param d:
        :return:
        """
        raise TodoException

    def add_data_writer(self, w):
        """

        :param w:
        :return:
        """
        self.data_writers.append(w)

    def add_simulation_data(self, d):
        """

        :param d:
        :return:
        """
        self.sim_data[d.name] = d

    def store_data(self):
        """

        :return:
        """
        for data_store in self.data_stores:
            data_store.store_data(self)

    def write_data(self):
        """

        :return:
        """
        for data_writer in self.data_writers:
            data_writer.write_data(self)

    def modify_simulation_state(self):
        """

        :return:
        """
        for modifier in self.simulation_modifiers:
            modifier.modify_simulation(self)

    def is_stopping_condition_met(self):
        """

        :return:
        """
        for stopping_condition in self.stopping_conditions:
            if stopping_condition.check_stopping_condition():
                return True
        return False

    def get_num_elements(self):
        return len(self.element_list)

    def animate(self, n, sm):
        """
        Since we aren't storing data at this point, the only way to animate is to calculate then
        plot
        :param n:
        :param sm:
        :return:
        """

        R = Renderer(self.gpu)
        R.render()

        for _ in tqdm(range(0, n, sm)):
            self.n_time_steps(sm)
            R.render()

    def _get_next_node_id(self):
        """

        :return:
        """
        ou = self.next_node_id
        self.next_node_id += 1
        return ou

    def _get_next_element_id(self):
        """

        :return:
        """
        ou = self.next_element_id
        self.next_element_id += 1
        return ou

    def _get_next_cell_id(self):
        """

        :return:
        """
        ii = self.next_cell_id
        self.next_cell_id += 1
        return ii

    def ccd(self, gpu: CudaMemory):
        candidates = gpu.candidates
        idxs = cp.argwhere(candidates)
        if len(idxs) == 0:
            return

        N_idxs, E_idxs = idxs[:, 0], idxs[:, 1]

        N_pos = gpu.N_pos[N_idxs]
        N_prp = gpu.N_pos_previous[N_idxs]
        E_nd1 = gpu.N_pos[gpu.E_node_1[E_idxs]]
        E_nd2 = gpu.N_pos[gpu.E_node_2[E_idxs]]

        A = E_nd1
        B = E_nd2
        C = N_prp
        D = N_pos

        ax, ay = A[:, 0], A[:, 1]
        bx, by = B[:, 0], B[:, 1]
        cx, cy = C[:, 0], C[:, 1]
        dx, dy = D[:, 0], D[:, 1]

        q_nom = ax * (by - cy) + bx * (cy - ay) + cx * (ay - by)
        q_den = (dx - cx) * (by - ay) + (cy - dy) * (bx - ax)
        q = q_nom / q_den
        p = (cx - ax + q * (dx - cx)) / (bx - ax)

        shorten = (0. <= p) & (p < 1.) & (0. <= q) & (q <= 1.)

        if any(shorten):
            gpu.N_pos[N_idxs][shorten] = (A + 0.9 * p[:, None] * (B - A))[shorten]
