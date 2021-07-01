from abc import ABC, abstractmethod


class AbstractCellBasedForce(ABC):
    """
    This class gives the details for how a force will be applied to each cell (as opposed to each
    element, or the whole population)
    """

    @abstractmethod
    def add_cell_based_forces(self, cell_list: list):
        pass