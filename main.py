import os

from src.models.biobots.connected_cells import ConnectedCells
from src.models.polygoncellsimulation.spheroid import Spheroid
from src.models.ringsimulation.ringbuckling import RingBuckling
from src.utils.tools import pyout, prng

# os.environ["NUMBA_DISABLE_JIT"] = "1"

s = ConnectedCells()
# s = Spheroid(10, 10, 10, 5, 0)
# s = RingBuckling(10, 10, 10, 0)
s.animate(10000, 1)
pyout()
pyout()
