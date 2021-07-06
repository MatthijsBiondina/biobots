from src.components.cell.cellcycle.growthcontactinhibition import GrowthContactInhibition
from src.components.forces.cellbasedforce.freecellperimeternormalisingforce import \
    FreeCellPerimeterNormalisingForce
from src.components.forces.cellbasedforce.polygoncellgrowthforce import PolygonCellGrowthForce
from src.components.forces.neighbourhoodbasedforce.cellcellinteractionforce import \
    CellCellInteractionForce
from src.components.simulation.datawriter.writespatialstate import WriteSpatialState
from src.components.simulation.freecellsimulation import FreeCellSimulation
from src.components.simulation.simulationdata.spatialstate import SpatialState
from src.components.spacepartition import SpacePartition
from src.utils.tools import pyout


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

            self.node_list += c.node_list
            self.element_list += c.element_list
            self.cell_list.append(c)

        """ ADD THE FORCES """

        # Cell growth force
        self.add_cell_based_force(PolygonCellGrowthForce(area_energy, perimeter_energy,
                                                         tension_energy))

        # Node-Element interaction force - requires a SpacePartition
        self.add_neighbourhood_based_force(CellCellInteractionForce(s, s, dAsym, dSep, dLim,
                                                                    self.dt, True))

        # Tries to make the edges the same length
        self.add_cell_based_force(FreeCellPerimeterNormalisingForce(sreg))

        """ ADD SPACE PARTITION """
        self.boxes = SpacePartition(0.3, 0.3, self)

        """ ADD THE DATA WRITERS """
        path_name = f"Spheroid/t0{t0}gtg{tg}gs{s}gsreg{sreg}gf{f}gda{dAsym}gds{dSep}gdl{dLim}" \
                    f"ga{area_energy}gb{perimeter_energy}gt{tension_energy}g_seed{seed}g/"
        self.add_simulation_data(SpatialState())
        self.add_data_writer(WriteSpatialState(20, path_name))
