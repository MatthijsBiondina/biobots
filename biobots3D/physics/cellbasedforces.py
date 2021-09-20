import sys
from timeit import timeit

from biobots3D.memory.gpusimdata import SimulationDataGPU
from utils.tools import pyout
import cupy as cp


# @cp.fuse
def cross_product(n0: cp.ndarray, n1: cp.ndarray, n2: cp.ndarray, ou: cp.ndarray):
    a = n1 - n0
    b = n2 - n0

    ou = cp.cross(a, b)
    # ou[:, 0] = a[:, 1] * b[:, 2] - a[:, 2] * b[:, 1]
    # ou[:, 1] = a[:, 2] * b[:, 0] - a[:, 0] * b[:, 2]
    # ou[:, 2] = a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]


def target_volume_forces(gpu: SimulationDataGPU):
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
    -\sum_{i} \rho h_0^2 (A - A_0)^2 * [r_{acw} - r_{cw}] x kA dynamic cell model for the
    formation of epithelial tissues
    Where r_{acw} and r_{cw} are vectors to the nodes anticlockwise and clockwise
    respectively of the node i, and k is a unit normal vector perpendicular to the plane of
    the nodes, and oriented by the right hand rule where anticlockwise is cw -> i -> acw. The
    cross product produces a vector in the plane perpendicular to r_{acw} - r_{cw},
    and pointing out of the cell at node i

    Practically, for each node in a cell, we take the cw and acw nodes, find the vector
    cw -> acw, find it's perpendicular vector, and apply a force along this vector according
    to the area energy parameter and the dA from equilibrium

    See Equations 29 & 30
    :param c:
    :return:
    :return:
    """

    # determine current volume
    N_0 = gpu.N_pos[gpu.S_n_0]
    N_1 = gpu.N_pos[gpu.S_n_1]
    N_2 = gpu.N_pos[gpu.S_n_2]
    crp = gpu.S_crp

    N = 10
    pyout(timeit(lambda: cross_product(N_0, N_1, N_2, crp), number=N) / N)

    sys.exit(0)
    # E_01 = N_1 - N_0
    # E_02 = N_2 - N_0
    #
    # A = cp.linalg.norm(cp.cross(E_01, E_02), axis=1) / 2

    pyout()
