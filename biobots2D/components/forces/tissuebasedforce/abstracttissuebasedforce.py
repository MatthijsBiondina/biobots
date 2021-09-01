from abc import ABC, abstractmethod


class AbstractTissueBasedForce(ABC):
    """
    This class gives the details for how a force will be applied to each cell (as opposed to each
    element, or the whole population)
    """

    @abstractmethod
    def add_tissue_based_forces(self, tissue):
        pass