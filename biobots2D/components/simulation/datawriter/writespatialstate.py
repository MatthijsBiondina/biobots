from biobots2D.components.simulation.abstractcellsimulation import AbstractCellSimulation
from biobots2D.components.simulation.datawriter.abstractdatawriter import AbstractDataWriter
from biobots2D.utils.errors import TodoException


class WriteSpatialState(AbstractDataWriter):

    def __init__(self, sm, sub_dir):
        """
        Stores the wiggle ratio
        :param sm:
        :param sub_dir:
        """
        super().__init__()
        self._file_names = {'nodes', 'elements', 'cells'}
        self._subdirectory_structure = [sub_dir, 'SpatialState/']
        self.sampling_multiple = sm
        self.multiple_files = False
        self.data = {}

    @property
    def file_names(self):
        return self._file_names

    @file_names.setter
    def file_names(self, value):
        self._file_names = value

    @property
    def subdirectory_stucture(self):
        return self._subdirectory_structure

    @subdirectory_stucture.setter
    def subdirectory_structure(self, value):
        self._subdirectory_structure = value

    def gather_data(self, t: AbstractCellSimulation):
        """
        The simulation t must have a simulation data object collating the complete spatial state
        :param t:
        :return:
        """
        self.data = t.sim_data['spatial_state'].get_data(t)