from abc import ABC, abstractmethod


class AbstractCellKiller(ABC):
    """
    An abstract class for killing cells instantly. This is only useful for keeping the ends
    trimmed so to speak, it will cause issues if it is used for an internal cell, since the
    simulation assumes that we have a contiguous line of cells.

    If an internal cell needs to be killed, this method leaves a gap that cannot be knitted back
    together (at least at this stage). If a cell needs to die when it is internal, a special
    apoptosis killer will need to be used
    """

    @abstractmethod
    def make_kill_list(self, cell_list):
        pass