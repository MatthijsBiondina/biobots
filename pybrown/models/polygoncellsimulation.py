from pybrown.components.simulation.simulation import FreeCellSimulation


class Spheroid(FreeCellSimulation):
    """
    This uses free cells, i.e. cells that never share elements or nodes with other cells
    """

    def __init__(self, t0: float = 10, tg: float = 10, s: float = 10, sreg: float = 5,
                 seed: int = 49):
        """
        Object input parameters can be chosen as desired. These are the most useful ones for
        tuning behaviour and running tests
        
        :param t0: the pause phase duration
        :param tg: the growth phase duration
        :param s: the cell-cell interaction force law parameter used for both adhesion and repulsion
        :param sreg: the perimeter normalising force
        :param seed: seed for random number generator
        """
        self.t0 = t0
        self.tg = tg
        self.s = s
        self.sreg = sreg
        self.seed = seed

        # Set the rng seed for reproducibility
        self.set_rng_seed(seed)

        # Other parameters
        # contact inhibition fraction
        self.f = 0.9

        # The asymptote, separation, and limit distances for the interaction force
        self.dAsym = -0.1
        self.dSep = 0.1
        self.dLim = 0.2

        # The energy densities for the cell growth force
        self.area_energy = 20
        self.perimeter_energy = 10
        self.tension_energy = 1

        # Make nodes around a polygon
        N = 10
        X = [0, 1, 0, 1]
        Y = [0, 0, 1, 1]

        for i in range(len(X)):
            x = X[i]
            y = Y[i]

            ccm = GrowthContactInhibition()
