# import random
from abc import abstractmethod, ABC
from typing import List

import numpy
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch import tensor

from src.components.cell.abstractcell import AbstractCell
from src.components.cell.celldeath.abstractcellkiller import AbstractCellKiller
from src.components.cell.celldeath.abstracttissuelevelcellkiller import \
    AbstractTissueLevelCellKiller
from src.components.cell.element import Element
from src.components.forces.cellbasedforce.abstractcellbasedforce import AbstractCellBasedForce
from src.components.forces.elementbasedforce.abstractelementbasedforce import \
    AbstractElementBasedForce
from src.components.forces.neighbourhoodbasedforce.abstractneighbourhoodbasedforce import \
    AbstractNeighbourhoodBasedForce
from src.components.forces.tissuebasedforce.abstracttissuebasedforce import \
    AbstractTissueBasedForce
from src.components.node.node import Node
from src.components.simulation.datastore.abstractdatastore import AbstractDataStore
from src.components.simulation.datawriter.abstractdatawriter import AbstractDataWriter
from src.components.simulation.modifiers.abstractsimulationmodifier import \
    AbstractSimulationModifier
from src.components.simulation.stopping.abstractstoppingcondition import \
    AbstractStoppingCondition
from src.components.spacepartition import SpacePartition
from src.utils.errors import TodoException
from src.utils.tools import pyout, prng


class AbstractCellSimulation(ABC):

    def __init__(self):
        """
        A parent class that contains all the functions for running a simulation. The child/concrete
        class will only need a constructor that assembles the cells
        """
        self.seed = None
        self.node_list: List[Node] = []
        self.next_node_id = 0
        self.element_list = []
        self.next_element_id = 0
        self.cell_list: List[AbstractCell] = []
        self.next_cell_id = 0

        self.stochastic_jiggle = True  # Brownian noise?
        self.epsilon = 0.0001  # Size of the jiggle force

        self.cell_based_forces: List[AbstractCellBasedForce] = []
        self.element_based_forces: List[AbstractElementBasedForce] = []
        self.neighbourhood_based_forces: List[AbstractNeighbourhoodBasedForce] = []
        self.tissue_based_forces: List[AbstractTissueBasedForce] = []

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

        super().__init__()

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
        # random.seed(seed)
        numpy.random.seed(seed)
        torch.manual_seed(seed)

    def next_time_step(self):
        """
        Updates all the forces and applies the movements
        :return:
        """

        self.generate_tissue_based_forces()
        self.generate_cell_based_forces()
        self.generate_element_based_forces()

        if self.using_boxes:
            self.generate_neighbourhood_based_forces()

        self.make_nodes_move()

        # Division must occur after movement
        self.make_cells_divide()

        self.kill_cells()

        self.modify_simulation_state()

        self.make_cells_age()

        self.step += 1
        self.t = self.step * self.dt

        self.store_data()

        if self.write_to_file:
            self.write_data()

        if self.is_stopping_condition_met():
            self.stopped = True

    def n_time_steps(self, n):
        """
        Advances a set number of time steps
        :param n:
        :return:
        """

        for ii in range(n):
            # Do all the calculations
            self.next_time_step()

            if self.step % 1000 == 0:
                print(f"Time = {self.t:.3f} hours")

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
            force.add_cell_based_forces(self.cell_list)

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
            force.add_neighbourhood_based_forces(self.node_list, self.boxes)

    def make_nodes_move(self):
        """

        :return:
        """
        for n in self.node_list:
            eta = n.eta
            force = n.force

            if self.stochastic_jiggle:
                # Add in a tiny amount of stochasticity to the force calculation to nudge it out
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
        for c in self.cell_list:
            c.age_cell(self.dt)

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

    def kill_cells(self):
        """
        Loop through the cell killers

        Currently the two killer types work differently. This is for backward compatibility with
        a hack job that I still need to work right now. Note to self, after all your work with
        DynamicLayer is done take some time to fix this up for SquareCellJoined
        :return:
        """

        for killer in self.tissue_level_killers:
            killer.kill_cells(self)

        kill_list: List[AbstractCell] = []
        for killer in self.cell_killers:
            kill_list.extend(killer.make_kill_list(self.cell_list))

        self.process_cells_to_remove(kill_list)

    def process_cells_to_remove(self, kill_list):
        """

        :param kill_list:
        :return:
        """
        # Loop from the end
        for c in reversed(kill_list):
            raise TodoException

    def is_stopping_condition_met(self):
        """

        :return:
        """
        for stopping_condition in self.stopping_conditions:
            if stopping_condition.check_stopping_condition():
                return True
        return False



    def get_num_cells(self):
        """

        :return:
        """
        raise TodoException

    def get_num_elements(self):
        """

        :return:
        """
        raise TodoException

    def get_num_nodes(self):
        """

        :return:
        """
        raise TodoException

    def visualise(self, varargin):
        """

        :param varargin:
        :return:
        """
        raise TodoException

    def visualise_area(self):
        """

        :return:
        """
        raise TodoException

    def visualise_rods(self, r):
        """

        :param r:
        :return:
        """
        raise TodoException

    def visualise_nodes(self, r):
        """

        :param r: cell radius
        :return:
        """
        raise TodoException

    def visualise_nodes_and_edges(self, r):
        """

        :param r: cell radius
        :return:
        """
        raise TodoException

    def visualise_wire_frame(self, varargin):
        """
        plot a line for each element
        :param varargin:
        :return:
        """
        raise TodoException

    def visualise_wire_frame_previous(self, varargin):
        """
        plot a line for each element
        :param varargin:
        :return:
        """

    def animate(self, n, sm):
        """
        Since we aren't storing data at this point, the only way to animate is to calculate then
        plot
        :param n:
        :param sm:
        :return:
        """

        totalSteps = 0
        while totalSteps < n:
            self.n_time_steps(sm)
            totalSteps += sm

            # rendering

            pyout()
        pyout()

        raise TodoException

    def animate_wire_frame(self, n, sm):
        """
        Since we aren't storing data at this point, the only way to animate is to calculate then
        plot
        :param n:
        :param sm:
        :return:
        """
        raise TodoException

    def animate_rods(self, n, sm, r):
        """

        :param n:
        :param sm:
        :param r:
        :return:
        """
        raise TodoException

    def animate_nodes(self, n, sm, r):
        """

        :param n:
        :param sm:
        :param r:
        :return:
        """
        raise TodoException

    def animate_nodes_and_edges(self, n, sm, r):
        """

        :param n:
        :param sm:
        :param r:
        :return:
        """
        raise TodoException

    def draw_pill(self, a, b, r):
        """
        Draws a pill shape where the centre of the circles are at a and b
        :param a: centre of circle
        :param b: centre of other circle
        :param r: radius
        :return:
        """
        raise TodoException

    def _get_next_node_id(self):
        """

        :return:
        """
        ou = self.next_node_id
        self.next_node_id += 1
        return ou

        # raise TodoException

    def _get_next_element_id(self):
        """

        :return:
        """
        raise TodoException

    def _get_next_cell_id(self):
        """

        :return:
        """
        ii = self.next_cell_id
        self.next_cell_id += 1
        return ii

    def _add_nodes_to_list(self, list_of_nodes):
        """

        :param list_of_nodes:
        :return:
        """
        raise TodoException

    def _add_elements_to_list(self, list_of_elements):
        """

        :param list_of_elements:
        :return:
        """
        raise TodoException
