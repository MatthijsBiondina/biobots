from pybrown.components.cell.cellcycle import GrowthContactInhibition
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
        super().__init__()

        # Set the rng seed for reproducibility
        self.set_rng_seed(seed)

        # Other parameters
        # contact inhibition fraction
        f = 0.9

        # The asymptote, separation, and limit distances for the interaction force
        dAsym = -0.1
        dSep = 0.1
        dLim = 0.2

        # The energy densities for the cell growth force
        area_energy = 20
        perimeter_energy = 10
        tension_energy = 1

        # Make nodes around a polygon
        N = 10
        X = [0, 1, 0, 1]
        Y = [0, 0, 1, 1]

        for i in range(len(X)):
            x = X[i]
            y = Y[i]

            ccm = GrowthContactInhibition(t0, tg, f, self.dt)

            c = self.make_cell_at_centre(N, x + 0.5 * (y % 2), y * 3 ** .5 / 2, ccm)

            self.node_list = [self.node_list, self.node_list]
            self.element_list = [self.element_list, self.element_list]
            self.cell_list = [self.cell_list, c]



        # Cell growth force
        self.add_cell_based_force(PolygonCellGrowthForce(area_energy, perimeter_energy, tension_energy))


