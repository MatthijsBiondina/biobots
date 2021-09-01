from abc import abstractmethod, ABC


class AbstractDataStore(ABC):
    """
    This class sets out the required functions for storing data about the simulation over time.

    This could be as simple as the number of cells, or as detailed as the precise position of all
    the elements. This will often be closely linked to a SimulationData object.
    """

    def __init__(self):
        self.data = None
        self.t_points = []  # corresponding time points
        self.sampling_multiple = None  # how many steps between each data point

    @abstractmethod
    def gather_data(self, t):
        pass

    def store_data(self, t):
        raise NotImplementedError
