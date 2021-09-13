from typing import List

from numpy import sin, pi, cos

from biobots2D.components.node.node import Node
from biobots2D.components.simulation.ringsimulation import RingSimulation
from utils import TodoException


class RingBuckling(RingSimulation):
    """
    This simulation makes a ring of cells, so there is no start or end cell. The intention is for the
    ring to buckle and self contact. The cell-cell interaction force allows the ring to interact with
    itself
    """

    def __init__(self, n: float, t0: float, tg: float, seed: int = 49):
        """

        :param n: number of cells in the ring, we restrict it to be >=10
        :param t0: growth start age
        :param tg: growth end age
        :param seed:
        """
        super(RingBuckling, self).__init__()
        self._dt = 0.005
        self._t = 0
        self._eta = 1
        self._time_limit = 2000

        self.set_rng_seed(seed)

        # Other parameters
        # The repulsion interaction force
        self.s = 10

        # Adhesion interaction (not used by default)
        # If it is used, it is preferable to set a = s, but not vital
        self.a = 0

        # Contact inhibition fraction
        f = 0.9

        # The asymptote, separation, and limit distances for the interaction force
        self.d_asymp = 0
        self.d_sep = 0.1
        self.d_lim = 0.2

        # The energy densities for the cell growth force
        self.area_energy = 20
        self.perimeter_energy = 10
        self.tension_energy = 1

        # --------------------------------------------------------
        # Make all the cells
        # --------------------------------------------------------
        #
        # The cells in this simulation forma  closed ring so every cell will have two neighbours. The
        # diameter of the ring is determinded by the number of cells. In order to have a sensible starting
        # configuration, we set a minimum number of 10 cells.

        if n < 10:
            raise ValueError(f"For a ring, at least 10 starting cells are needed, but {n} were given.")

        # --------------------------------------------------------
        # Make a list of top nodes and bottom nodes
        # --------------------------------------------------------
        #
        # We want to minimize the difference between the top and the bottom element lenghts. The internal
        # element lengths will be 1. Cells will be spaced evenly, covering 2*pi/n rads. We also keep the
        # cell area at 0.5 since we are starting in Pause

        # Under these conditions, the radius r of the bottom nodes is given by:
        r = 0.5 * sin(2 * pi / n) - 0.5

        top_nodes: List[Node] = []
        bottom_nodes: List[Node] = []

        for ii in range(n):
            theta = 2 * pi * ii / n
            xb = r * cos(theta)
            yb = r * sin(theta)

            xt = (r + 1) * cos(theta)
            yt = (r + 1) * sin(theta)

            bottom_nodes.append(Node(xb, yb, self._get_next_node_id()))
            top_nodes.append(Node(xt, yt, self._get_next_node_id()))

        self._add_nodes_to_list(bottom_nodes)
        self._add_nodes_to_list(top_nodes)

        raise TodoException



    @property
    def dt(self):
        return self._dt

    @dt.setter
    def dt(self, value):
        self._dt = value

    @property
    def t(self):
        return self._t

    @t.setter
    def t(self, value):
        self._t = value

    @property
    def eta(self):
        return self._eta

    @eta.setter
    def eta(self, value):
        self._eta = value

    @property
    def time_limit(self):
        return self._time_limit

    @time_limit.setter
    def time_limit(self, value):
        self._time_limit = value
