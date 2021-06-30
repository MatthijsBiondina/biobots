from abc import abstractmethod, ABC


class AbstractDataWriter(ABC):
    """
    This class sets out the required functions for writing data to file

    This will usually be used for something like storing the spatial positions of the nodes etc
    for later visualisation. In other words it is designed for use when each time step will
    produce a lot of data, rather than just a handful of values.
    """

    def __init__(self):
        # A structure that holds the data for a single time step. This will be a cell array for
        # each type of data held by the concrete class. It can take matrices or cell arrays. Each
        # row of the matrix or cell array should contain all the data about a single unit that is
        # being stored for example, if storing nodes, each row should be data for a single node.
        self.data = None

        # The corresponding time point
        self.time_point = None

        # How many steps between each data point
        self.sampling_multiple = None

        # How many significant figures to store
        self.precition = 5

        # A flag to determine if each time step will be written to a separate file, or all will
        # be written in a single file
        self.multiple_files = True

        # If writing multiple files, each file needs a unique number so we need to keep track of
        # where we're up to
        self.file_number = 0

        # A path pointing to where the data will be stored
        self.root_storage_location = None

        # A flag to tell the function that the full path has been made
        self.full_path_made = False

        # The full path to the folder where the data will be written
        self.full_path = None

        # A flag to determine if a time stamp is added to the start of each line
        self.time_stamp_needed = True

    @property
    @abstractmethod
    def file_names(self):
        """
        A name for the file(s) to be written. This will be given in a list, and the file
        names will match with the data in the matching data list
        :return:
        """
        pass

    @property
    @abstractmethod
    def subdirectory_stucture(self):
        """
        The sub directory structure under the root storage location. Should be at minimum
        'simname/' and can also go deeper with parameter sets etc.
        :return:
        """
        pass

    def write_data(self, t):
        raise NotImplementedError

    def write_to_multiple_files(self):
        raise NotImplementedError

    def write_to_single_file(self):
        raise NotImplementedError

    def make_full_path(self):
        raise NotImplementedError


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
        raise NotImplementedError
