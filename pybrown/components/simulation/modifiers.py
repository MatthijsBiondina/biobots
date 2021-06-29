from abc import ABC, abstractmethod


class AbstractSimulationModifier(ABC):
    """
    This class sets out the required functions for modifying a simulation.

    Since this si directly changing the simulation state, there may be instances when multiple
    modifiers interact. Caution needs to be observed.
    """

    @abstractmethod
    def modify_simulation(self, t):
        pass
