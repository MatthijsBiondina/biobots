import csv
import os
import shutil
from abc import abstractmethod, ABC

from utils import TodoException


class AbstractDataWriter(ABC):
    """
    This class sets out the required functions for writing data to file

    This will usually be used for something like storing the spatial positions of the nodes etc
    for later visualisation. In tutorials words it is designed for use when each time step will
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
        self.precision = 5

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

    @abstractmethod
    def gather_data(self, t):
        pass

    def write_data(self, t):
        """

        :param t: AbstractCellSimulation
        :return:
        """
        if t.step % self.sampling_multiple == 0:
            if not self.full_path_made:
                self.make_full_path()
                self.full_path_made = True

            self.gather_data(t)

            self.time_point = t.t

            if self.multiple_files:
                raise TodoException
            else:
                self.write_to_single_file()

    def write_to_multiple_files(self):
        raise TodoException

    def write_to_single_file(self):
        for key in self.data:
            with open(os.path.join(self.full_path, f"{key}.csv"), 'a+', newline='') as f:
                writer = csv.writer(f, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
                D = self.data[key]
                for d in D:
                    if self.time_stamp_needed:
                        writer.writerow([self.time_point] + [f"{r:.5f}" for r in d])
                    else:
                        writer.writerow([f"{r:.5f}" for r in d])

    def make_full_path(self, debug=True):
        self.root_storage_location = './res/SimulationOutput'
        self.full_path = os.path.join(self.root_storage_location,
                                      'tmp' if debug else self.subdirectory_stucture)

        if os.path.isdir(self.full_path):
            shutil.rmtree(self.full_path)
        os.makedirs(self.full_path)
