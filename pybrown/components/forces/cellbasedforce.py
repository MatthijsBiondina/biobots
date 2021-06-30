from abc import ABC, abstractmethod


class AbstractCellBasedForce(ABC):
    """
    This class gives the details for how a force will be applied to each cell (as opposed to each
    element, or the whole population)
    """

    @abstractmethod
    def add_cell_based_forces(self, cell_list: list):
        pass


class PolygonCellGrowthForce(AbstractCellBasedForce):
    def __init__(self, area_P, perimeter_P, tension_P):
        """
        Applies energy based methods to drive the cell to a target area and perimeter based on
        the energy density parameters
        """
        self.area_energy_parameter = area_P
        self.perimeter_energy_parameter = perimeter_P
        self.surface_tension_energy_parameter = tension_P

    def add_cell_based_forces(self, cell_list: list):
        """
        For each cell in the list, calculate the forces and add them to the nodes
        :param cell_list:
        :return:
        """
        raise NotImplementedError

    def add_target_area_forces(self, c):
        """
        This force comes from "A dynamic cell model for the formation of epithelial tissues",
        Nagai, Honda 2001. It comes from section 2.2 "Resistance force against cell deformation".
        This force will push to cell to a target area. If left unchecked the cell will end up at
        it's target area. The equation governing this energy is
        U = \rho h_0^2 (A - A_0)^2
        Here \rho is a area energy parameter, h_0 is the equilibrium height (in 3D) and A_0 is
        the equilibrium area. In this force, \rho h_0^2 is replaced by \alpha referred to
        areaEnergyParameter

        This energy allows the cell to compress, but penalises the compression quadratically. The
        cyctoplasm of a cell is mostly water, so can be assumed incompressible, but the bilipid
        membrane can have all sorts of molecules on its surface that may exhibit some
        compressibility.

        The resulting force comes from taking the -ve divergence of the energy, and using a
        cross-product method of finding the area of a given polygon. This results in:
        -\sum_{i} \rho h_0^2 (A - A_0)^2 * [r_{acw} - r_{cw}] x k
        Where r_{acw} and r_{cw} are vectors to the nodes anticlockwise and clockwise
        respectively of the node i, and k is a unit normal vector perpendicular to the plane of
        the nodes, and oriented by the right hand rule where anticlockwise is cw -> i -> acw. The
        cross product produces a vector in the plane perpendicular to r_{acw} - r_{cw},
        and pointing out of the cell at node i

        Practically, for each node in a cell, we take the cw and acw nodes, find the vector
        cw -> acw, find it's perpendicular vector, and apply a force along this vector according
        to the area energy parameter and the dA from equilibrium
        :param c:
        :return:
        """
        raise NotImplementedError

    def add_target_perimeter_forces(self, c):
        """
        This calculates the force applied to the cell boundary due to a difference between
        current perimeter and target perimeter. The energy held in the boundary is given by
        U = \beta (p - P_0)^2 where p is the current perimeter and P_0 is the equilibrium
        perimeter. The current perimeter is found by summing the magnitudes of the vectors
        representing the edges around the perimeter. To convert the energy to a force,
        the -ve divergence is taken with respect to the variables identifying a given node's
        coordinates. Evidently the only vectors that contribute are the one that contain the node.
        :param c:
        :return:
        """
        raise NotImplementedError

    def add_surface_tension_forces(self, c):
        """
        This force comes from "A dynamic cell model for the formation of epithelial tissues",
        Nagai, Honda 2001. It comes from section 2.1 "Tension of the cell boundary" and is a
        force resulting from the tendency to minimise the energy held in the boundary of the
        cell. If left unchecked, this will drive the boundary of the cell to zero. The equation
        governing the energy is:
        U = \sum_{j} \sigma_{\alpha,\beta} |r_{i} - r_{j}|
        i is a given node around a cell boundary, j are the nodes that share an edge with i
        \sigma_{\alpha,\beta} is the energy per unit length of the edge between cells \alpha and
        \beta.

        The resulting force comes by taking the -ve gradient of the energy, giving
        -\sigma_{\alpha,\beta} (r_{i} - r_{j}) / |r_{i} - r_{j}|

        Since the parameter \sigma is specified for any cell pair in the paper, it can be used as
        a way for any two cells to minimise their shared boundary. If used this way it should
        also account for the tendency for adhesion between two cells, meaning that the parameter
        could go negative.

        Here it will be used as a tendency for a cell to want to minimise it boundary in general.
        Adhesion forces will be dealt separately.

        Practically, the force means for any given element, it's nodes will be pulled together.
        In terms of the node, we will find a force pointing towards any other node that it shares
        an edge with.

        Technically this can be separated out as an AbstractElementBasedForce, but it is kept
        here since it is part of the "Nagai-Honda" model of a cell.
        :param c:
        :return:
        """
        raise NotImplementedError


class FreeCellPerimeterNormalisingForce(AbstractCellBasedForce):
    def __init__(self, spring_rate):
        """
        A normalising force to keep the edges around a free cell roughly the same size. It will
        push each edge to have length P/N, where P is the current perimeter and N the number of
        edges
        """
        self.spring_rate = spring_rate

    def add_cell_based_forces(self, cell_list: list):
        """
        For each cell in the list, calculate the forces and add them to the nodes
        :param cell_list:
        :return:
        """
        raise NotImplementedError

    def apply_spring_force(self, c):
        """

        :param c:
        :return:
        """
        raise NotImplementedError
