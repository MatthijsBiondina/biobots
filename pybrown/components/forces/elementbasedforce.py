from abc import ABC, abstractmethod


class AbstractElementBasedForce(ABC):
    """
    This class gives the details for how a force will be applied to each Element (as opposed to
    each cell, or the whole population)
    """
    @abstractmethod
    def add_element_based_forces(self, element_list:list):
        pass