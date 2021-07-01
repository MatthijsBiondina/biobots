from pybrown.components.simulation.datawriter.abstractdatawriter import AbstractDataWriter
from pybrown.utils.errors import TodoException


class WriteSpatialState(AbstractDataWriter):

    def __init__(self, sm, sub_dir):
        """
        Stores the wiggle ratio
        :param sm:
        :param sub_dir:
        """
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

    def gather_data(self, t):
        """
        The simulation t must have a simulation data object collating the complete spatial state
        :param t:
        :return:
        """
        raise TodoException