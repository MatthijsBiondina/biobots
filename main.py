import os

from src.models.biobots.connected_cells import ConnectedCells
from src.models.biobots.gradient import Gradient
from src.models.polygoncellsimulation.spheroid import Spheroid
from src.models.ringsimulation.ringbuckling import RingBuckling
from src.utils.tools import pyout, prng

# os.environ["NUMBA_DISABLE_JIT"] = "1"

# s = ConnectedCells()
s = Gradient()


s.animate(1000000,1)
pyout()
pyout()
