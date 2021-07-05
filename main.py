from src.models.polygoncellsimulation.spheroid import Spheroid
from src.utils.tools import pyout

s = Spheroid(10, 10, 10, 5, 0)
s.animate(1000, 1)
pyout()
pyout()
