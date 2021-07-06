# import random
from abc import abstractmethod, ABC
from typing import List

import numpy
import numpy as np
import torch
import matplotlib.pyplot as plt

from src.components.cell.celldeath.abstractcellkiller import AbstractCellKiller
from src.components.cell.celldeath.abstracttissuelevelcellkiller import \
    AbstractTissueLevelCellKiller
from src.components.forces.cellbasedforce.abstractcellbasedforce import AbstractCellBasedForce
from src.components.forces.elementbasedforce.abstractelementbasedforce import \
    AbstractElementBasedForce
from src.components.forces.neighbourhoodbasedforce.abstractneighbourhoodbasedforce import \
    AbstractNeighbourhoodBasedForce
from src.components.forces.tissuebasedforce.abstracttissuebasedforce import \
    AbstractTissueBasedForce
from src.components.simulation.datastore.abstractdatastore import AbstractDataStore
from src.components.simulation.datawriter.abstractdatawriter import AbstractDataWriter
from src.components.simulation.modifiers.abstractsimulationmodifier import \
    AbstractSimulationModifier
from src.components.simulation.stopping.abstractstoppingcondition import \
    AbstractStoppingCondition
from src.utils.errors import TodoException
from src.utils.tools import pyout


class AbstractCellSimulation(ABC):

    def __init__(self):
        """
        A parent class that contains all the functions for running a simulation. The child/concrete
        class will only need a constructor that assembles the cells
        """
        self.seed = None
        self.node_list = []
        self.next_node_id = 0
        self.element_list = []
        self.next_element_id = 0
        self.cell_list = []
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

        self.boxes = None

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

        pyout()
        pyout()

        raise TodoException

    def n_time_steps(self, n):
        """
        Advances a set number of time steps
        :param n:
        :return:
        """

        for ii in range(n):
            # Do all the calculations
            self.next_time_step()

            pyout()
            pyout()

        raise TodoException

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

        raise TodoException

    def generate_element_based_forces(self):
        """

        :return:
        """
        raise TodoException

    def generate_neighbourhood_based_forces(self):
        """

        :return:
        """
        raise TodoException

    def make_nodes_move(self):
        """

        :return:
        """
        raise TodoException

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
        raise TodoException

    def add_new_cells(self, new_cells, new_elements, new_nodes):
        """
        When a cell divides, need to make sure the new cell object as well as the new elements
        and nodes are correctly added to their respective lists and boxes if relevant
        :param new_cells:
        :param new_elements:
        :param new_nodes:
        :return:
        """
        raise TodoException

    def make_cells_age(self):
        """

        :return:
        """
        raise TodoException

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
        raise TodoException

    def write_data(self):
        """

        :return:
        """
        raise TodoException

    def modify_simulation_state(self):
        """

        :return:
        """
        raise TodoException

    def kill_cells(self):
        """
        Loop through the cell killers

        Currently the two killer types work differently. This is for backward compatibility with
        a hack job that I still need to work right now. Note to self, after all your work with
        DynamicLayer is done take some time to fix this up for SquareCellJoined
        :return:
        """
        raise TodoException

    def process_cells_to_remove(self, kill_list):
        """

        :param kill_list:
        :return:
        """
        raise TodoException

    def is_stopping_condition_met(self):
        """

        :return:
        """
        raise TodoException

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
        yield self.next_node_id
        self.next_node_id += 1
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
        ii = self.next_node_id
        self.next_node_id += 1
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
