from abc import ABC

from pybrown.components.cell.celldeath import AbstractTissueLevelCellKiller, AbstractCellKiller
from pybrown.components.forces.cellbasedforce import AbstractCellBasedForce
from pybrown.components.forces.elementbasedforce import AbstractElementBasedForce
from pybrown.components.forces.neighbourhoodbasedforce import AbstractNeighbourhoodBasedForce
from pybrown.components.forces.tissuebasedforce import AbstractTissueBasedForce
from pybrown.components.simulation.datastore import AbstractDataStore
from pybrown.components.simulation.datawriter import AbstractDataWriter
from pybrown.components.simulation.modifiers import AbstractSimulationModifier
from pybrown.components.simulation.stopping import AbstractStoppingCondition


class AbstractCellSimulation(ABC):
    """
    A parent class that contains all the functions for running a simulation. The child/concrete
    class will only need a constructor that assembles the cells
    """

    def __init__(self):
        self.seed = None
        self.node_list = []
        self.next_node_id = 0
        self.element_list = []
        self.next_element_id = 0
        self.cell_list = []
        self.next_cell_id = 0

        self.stochastic_jiggle = True  # Brownian noise?
        self.epsilon = 0.0001  # Size of the jiggle force

        # todo: I think these class declarations should be iterables
        self.cell_based_forces: AbstractCellBasedForce = None
        self.element_based_forces: AbstractElementBasedForce = None
        self.neighbourhood_based_forces: AbstractNeighbourhoodBasedForce = None
        self.tissue_based_forces: AbstractTissueBasedForce = None

        self.stopping_conditions: AbstractStoppingCondition = None

        self.stopped = False

        self.tissue_level_killers: AbstractTissueLevelCellKiller = None
        self.cell_killers: AbstractCellKiller = None

        self.simulation_modifiers: AbstractSimulationModifier = None

        # A collection of objects that store data over multiple time steps with also the
        # potential to write to file
        self.data_stores: AbstractDataStore = None
        self.data_writers: AbstractDataWriter = None

        # A collection of objects for calculating data about the simulation stored in a map
        # container so each type of data can be given a meaningful name
        self.sim_data = {}

        self.boxes = None




class FreeCellSimulation(AbstractCellSimulation):
    """
    This uses free cells, i.e. cells that never share elements or nodes with other cells
    """
    def __init__(self):
        """
        """
        self.dt = 0.005
        self.t = 0
        self.step = 0
