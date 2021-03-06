from typing import List

import cupy as cp
import torch
from numba import float32, guvectorize

from biobots2D.components.cell.abstractcell import AbstractCell
from biobots2D.components.forces.cellbasedforce.abstractcellbasedforce import AbstractCellBasedForce
from biobots2D.components.simulation.cuda_memory import CudaMemory


@guvectorize([(float32[:], float32[:], float32, float32[:])], '(n),(n),()->(n)', target="cuda")
def compute_magnitude(current_area, target_area, area_energy, res):
    for i in range(current_area.shape[0]):
        res[i] = (current_area[i] - target_area[i]) * area_energy


class PolygonCellGrowthForce(AbstractCellBasedForce):
    def __init__(self, area_P, perimeter_P, tension_P):
        """
        Applies energy based methods to drive the cell to a target area and perimeter based on
        the energy density parameters
        """

        # CPU
        self.area_energy_parameter = area_P
        self.perimeter_energy_parameter = perimeter_P
        self.surface_tension_energy_parameter = tension_P

        # CUDA placeholders
        self.area_energy_parameter_gpu = cp.float32(area_P)
        self.perimeter_energy_parameter_gpu_x2 = cp.float32(perimeter_P * 2)
        self.surface_tension_energy_parameter_gpu = cp.float32(tension_P)
        self.orthogonal_inwards = cp.array([[0., -1.], [1., 0.]])

    def add_cell_based_forces(self, cell_list: List[AbstractCell], gpu: CudaMemory):
        """
        For each cell in the list, calculate the forces and add them to the nodes
        :param cell_list:
        :return:
        """
        if gpu.EXEC_CPU:
            for c in cell_list:
                if c.cell_type != 5:  # As long as it is not stromal type
                    self.add_target_area_forces(c)
                    self.add_target_perimeter_forces(c)
                    self.add_surface_tension_forces(c)

        self.add_target_area_forces_cuda(gpu)
        self.add_target_perimeter_forces_cuda(gpu)
        self.add_surface_tension_forces_cuda(gpu)

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

        See Equations 29 & 30
        :param c:
        :return:
        """

        current_area = c.get_cell_area()
        target_area = c.get_cell_target_area()

        magnitude = self.area_energy_parameter * (current_area - target_area)

        N = len(c.node_list)
        for ii in range(N):
            n = c.node_list[ii]
            ncw = c.node_list[(ii - 1) % N]
            nacw = c.node_list[(ii + 1) % N]

            u = nacw.position - ncw.position
            v = u @ torch.tensor([[0., -1.], [1., 0.]])

            n.add_force_contribution(-v * magnitude)

    def add_target_area_forces_cuda(self, gpu: CudaMemory):
        current_area = gpu.C_area
        tarrget_area = gpu.C_target_area

        magnitude = self.area_energy_parameter_gpu * (gpu.C_area - gpu.C_target_area)

        n = gpu.N_pos[gpu.C_node_idxs]

        ncw = cp.roll(n, 1, axis=1)
        nacw = cp.roll(n, -1, axis=1)
        u = nacw - ncw
        v = u @ self.orthogonal_inwards
        F = -v * magnitude[:, None, None]

        # broadcast from node_idxs in cell_lists to node idxs in N_for
        F = F[:, None, :, :]
        F = cp.einsum('ijk...,ikl...->ijl...', F, gpu.cell2node[:, :, :, None]).squeeze(1)
        F = cp.sum(F, axis=0)

        gpu.N_for += F

    def add_target_perimeter_forces(self, c: AbstractCell):
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
        current_perimeter = c.get_cell_perimeter()
        target_perimeter = c.get_cell_target_perimeter()

        magnitude = 2 * self.perimeter_energy_parameter * (current_perimeter - target_perimeter)

        for e in c.element_list:
            r = e.get_vector_1_to_2()
            f = magnitude * r

            e.node_1.add_force_contribution(f)
            e.node_2.add_force_contribution(-f)

    def add_target_perimeter_forces_cuda(self, gpu: CudaMemory):
        current_perimeter = gpu.C_perimeter
        target_perimeter = gpu.C_target_perimeter
        magnitude = self.perimeter_energy_parameter_gpu_x2 * (current_perimeter - target_perimeter)
        r = gpu.vector_1_to_2

        # broadcast to elements per cell
        F = r * (magnitude @ gpu.cell2element)[:, None]

        # broadcast to nodes
        F = cp.sum(F[:, None, :, None] * gpu.element2node[:, :, None, :], axis=0)

        F_node_1 = F[:, :, 0]
        F_node_2 = -F[:, :, 1]

        gpu.N_for += F_node_1 + F_node_2

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
        In terms of the node, we will find a force pointing towards any tutorials node that it shares
        an edge with.

        Technically this can be separated out as an AbstractElementBasedForce, but it is kept
        here since it is part of the "Nagai-Honda" model of a cell.
        :param c:
        :return:
        """

        for e in c.element_list:
            r = e.get_vector_1_to_2()
            f = self.surface_tension_energy_parameter * r

            e.node_1.add_force_contribution(f)
            e.node_2.add_force_contribution(-f)

    def add_surface_tension_forces_cuda(self, gpu: CudaMemory):
        r = gpu.vector_1_to_2
        surface_tension_energy_parameter = self.surface_tension_energy_parameter
        F = self.surface_tension_energy_parameter * r

        A = F[:, None, :, None] * gpu.element2node[:, :, None, :]

        F = cp.sum(F[:, None, :, None] * gpu.element2node[:, :, None, :], axis=0)
        F_node_1 = F[:, :, 0]
        F_node_2 = -F[:, :, 1]
        F = F_node_1 + F_node_2

        gpu.N_for += F
