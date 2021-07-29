import sys
import time
from typing import List, Union

import cupy as cp
import numpy as np
# import numpy as cp
import torch
from torch import tensor

from src.components.forces.neighbourhoodbasedforce.abstractnodeelementforce import \
    AbstractNodeElementForce
from src.components.node.node import Node
from src.components.simulation.cuda_memory import CudaMemory, synchronize
from src.components.spacepartition import SpacePartition
from src.utils.errors import TodoException
from src.utils.tools import pyout


class CellCellInteractionForce(AbstractNodeElementForce):
    def __init__(self, sra, srr, da, ds, dl, dt, using_polys):
        """
        This force controls the interaction between cells. It encompasses node-node and node-edge
        interactions. t applies to both orientable edges (i.e. polygon cells where the inside
        defines and orientation) and non-orientable cells represented by a single edge i.e.
        rod/capsule cells like yeast etc. The force law is designed to asymptote to infinite
        repulsion at certain distance or overlap between a node and edge. It has a preferred
        separation, and an interaction limit. The specific equations are:

        if dAsym < x < dSep
            srr * log(  ( dSep - dAsym ) / ( x - dAsym)  )
        elif dSep < x < dLim
            sra * (  ( dSep - x ) / ( dSep - dAsym )  ) * exp(c*(dSep - x) )
        else (if dLim < x)
            0

        The force law is parameterised so that:
        - Attraction and repulsion strength can be modulated separately
        - The asymptote point can be set to the inside of the cell, or exactly at the boundary to
          prevent overlap
        - The preferred separation distance and interaction limit are specified individually
          (i.e. the limit does not need to be a multiple of the separation)
        - The force is zero at x = dSep, and if srr = sra, then the derivative at x = dSep is
          smooth

        Depending on the application, the force can be applied in slightly different ways. If the
        cells are capsule cells represented by edges/rods, then we are only interested in the
        absolute distance between a node and edge. If the cells are polygons, then the
        orientation of the edge is important, so we consider a signed distance based on the
        inside of the cell. A flag usingRodsOrPolygons specifies the case

        The interactions between nodes can be turned on or off via the useNodeNodeInteractions
        boolean variable, which is disabled by default. This must be set to true for rod cells
        represented by a single edge, otherwise unphysical overlap and hence numerical issues are
        much more likely

        If a polygon cell is being used, then there is the risk that an edge will invert,
        causing a non-simple polygon (i.e. a figure-8). This is mitigated to a large extent by
        permitting nodes and edges of the same cell to have  a repulsion interaction. This will
        cause the cell to expand by making the node and edge push apart. This is controlled by
        useInternalRepulsion and is enabled by default

        In order to function, this force calculator need to access a space partition that
        efficiently stores the neighbouring nodes and edges. All that needs to be accessed from
        the space partition is GetNeighbouringElements and GetNeighbouringNodes

        Optional controls:
        useNodeNodeInteractions: default false, automatically set to true for rods,
                                 can be manually set to true for polygons but is NOT RECOMMENDED
                                 as it needs smaller time step to be stable, and motion is much
                                 less smooth
        useInternalRepulsion: default false, automatically set to true for polygons, can't be set
                              true for rods

        :param sra: spring rate attraction, positive
        :param srr: spring rate repulstion, positive
        :param da: d-asymptote, typical values [-0.1, 0]
        :param ds: d-separation, typical value 0.1
        :param dl: d-limit, typical value 0.2
        :param dt: timestep size of the simulation
        :param using_polys: rods = false or polygons = true
        """
        self.spring_rate_attraction = sra
        self.spring_rate_repulsion = srr
        self.d_asymptote = da
        self.d_separation = ds
        self.d_limit = dl
        self.dt = dt

        # The shape parameter of the attraction force law, set here so it can be modified,
        # although it is not intended to be.
        self.c = 5
        self.using_polys = using_polys
        if using_polys:
            self.use_internal_repulsion = True
            self.use_node_node_interactions = False
        else:
            self.use_internal_repulsion = False
            self.use_node_node_interactions = True

            if da < 0:
                raise ValueError("CCIF:overlap (The force asymptote position allows overlap, "
                                 "which is not supported for rod cells)")

        self.spring_rate_attraction_cuda = cp.float32(self.spring_rate_attraction)
        self.spring_rate_repulsion_cuda = cp.float32(self.spring_rate_repulsion)
        self.d_asymptote_cuda = cp.float32(self.d_asymptote)
        self.d_separation_cuda = cp.float32(self.d_separation)
        self.d_limit_cuda = cp.float32(self.d_limit)
        self.dt_cuda = cp.float32(self.dt)
        self.c_cuda = cp.float32(self.c)

        self.repulsion_range_cuda = self.d_separation_cuda - self.d_asymptote_cuda
        self.attraction_range_cuda = self.d_limit_cuda - self.d_separation_cuda

        # cuda placeholders
        self.Fa = None

    def add_neighbourhood_based_forces(self, node_list: List[Node],
                                       p: Union[SpacePartition, None] = None,
                                       gpu: CudaMemory = None):
        """
        Here we calculate the forces between nodes and edges, and nodes and nodes.

        The force in ForceLaw assumes a positive scalar force is a repulsion interaction.

        For a node-edge interaction, we have to use the semi-rigid body approach in
        ApplyForcesToNodeAndElement which is supplied by the parent object. To use the function,
        the force Fa is assumed to have positive sense towards the edge. This means that Fa will
        be positive for repulsion applied to an edge, or negative for repulsion applied to the
        node.

        For a node-node interaction, the forces must be added in this method
        :param node_list: a vector of nodes in the simulation p is the space partition of nodes
                          and edges in the simulation
        :param p:
        :return:
        """
        if gpu.EXEC_CPU:
            for n in node_list:
                if self.use_node_node_interactions:
                    raise TodoException
                else:
                    e_list = p.get_neighbouring_elements(n, self.d_limit)
                    n_list: List[Node] = []

                for e in e_list:
                    if e.id == 18:
                        pyout()
                    # A unit vector tangent to the edge
                    u = e.get_vector_1_to_2()

                    # We arbitrarily choose an end point on the edge to make a vector going from edge
                    # to node, then project it onto the tangent vector to find the point of action
                    n1ton = n.position - e.node_1.position
                    n1toA = u * torch.dot(n1ton, u)

                    if self.using_polys:
                        # We use the outward pointing normal to orient the edge
                        v = e.get_outward_normal()
                        # ... and project the arbitrary vector onto the outward normal to find the
                        # signed distance between edge and node
                        x = torch.dot(n1ton, v)

                        # Need to check if node-edge interaction pair is between a node and edge of
                        # the same cell
                        internal = any(item in e.cell_list for item in n.cell_list)

                        # The negative sign is necessary because v points away from the edge and we
                        # need to point towards the edge
                        Fa = -self.force_law(x, internal) * v
                    else:
                        raise TodoException

                    self.apply_forces_to_node_and_element(n, e, Fa, n1toA)

                if self.use_node_node_interactions:
                    raise TodoException

        self.add_neighbourhood_based_forces_cuda(gpu)

    def add_neighbourhood_based_forces_cuda(self, gpu: CudaMemory):
        N_idxs, E_idxs = self.get_neighbouring_elements_cuda(gpu)

        # A unit vector tangent to the edge
        u = gpu.vector_1_to_2[E_idxs]

        # We arbitrarily choose an end point on the edge to make a vector going from edge to node,
        # then project it onto the tangent vector to find the point of action
        n1ton = gpu.N_pos[N_idxs] - gpu.N_pos[gpu.E_node_1[E_idxs]]
        n1toA = u * cp.sum(n1ton * u, axis=1)[:, None]

        # We use the outward pointing normal to orient the edge
        v = gpu.outward_normal[E_idxs]
        # ... and project the arbitrary vector onto the outward normal to find the signed distance
        # between edge and node
        x = cp.sum(n1ton * v, axis=1)

        Fa = -self.force_law_cuda(gpu, x, N_idxs, E_idxs)[:, None] * v

        self.apply_forces_to_node_and_element_cuda(gpu, N_idxs, E_idxs, Fa, n1toA)

    def get_neighbouring_elements_cuda(self, gpu: CudaMemory):
        candidates = cp.ones((gpu.N_pos.shape[0], gpu.E_node_1.shape[0]), dtype=cp.bool)

        # find nodes in element interaction regions
        pnt = gpu.N_pos
        start = gpu.N_pos[gpu.E_node_1]
        end = gpu.N_pos[gpu.E_node_2]
        line_vec = end - start
        pnt_vec = pnt[:, None] - start[None, :]
        line_len = cp.linalg.norm(line_vec, axis=1)
        line_unitvec = line_vec / line_len[:, None]
        pnt_vec_scaled = pnt_vec / line_len[None, :, None]
        t = cp.sum(line_unitvec[None, :, :] * pnt_vec_scaled, axis=2)
        nearest = line_vec[None, :, :] * t[:, :, None]
        dist = cp.linalg.norm(nearest - pnt_vec, axis=2)
        candidates = candidates & (0. <= t) & (t <= 1.) & (dist < self.d_limit_cuda)

        # same element
        # same_cell = cp.sum(gpu.cell2node.T[None, :] * gpu.cell2node.T[:, None], axis=2) > 0.5

        candidates = candidates & ~ gpu.node2element_mask
        idxs = cp.argwhere(candidates)
        N_idxs, E_idxs = idxs[:, 0], idxs[:, 1]
        return N_idxs, E_idxs

    def force_law(self, x, internal):
        """
        This calculates the scalar force for the given separation x and the controlling
        parameters as outline in the preamble
        :param x:
        :param internal:
        :return:
        """

        if not internal:
            # The interaction is between separate cells
            if self.d_asymptote < x and x < self.d_separation:
                # raise TodoException
                Fa = self.spring_rate_repulsion \
                     * torch.log((self.d_separation - self.d_asymptote) / (x - self.d_asymptote))
                return Fa
            elif self.d_separation <= x and x < self.d_limit:
                Fa = self.spring_rate_attraction \
                     * ((self.d_separation - x) / (self.d_separation - self.d_asymptote)) \
                     * torch.exp(self.c * (self.d_separation - x) / self.d_separation)
                return Fa
            else:
                tensor(0.)
        else:
            raise TodoException

    def force_law_cuda(self, gpu: CudaMemory, x, N_idxs, E_idxs):
        # Need to check if node-edge interaction pair is between a node and edge of the same cell
        internal_mask = cp.any(gpu.cell2node_mask.T[N_idxs] & gpu.cell2element_mask.T[E_idxs],
                               axis=1)
        repulsion_mask = (self.d_asymptote_cuda < x) & (x < self.d_separation_cuda)
        attraction_mask = (self.d_separation_cuda < x) & (x < self.d_limit_cuda)
        repulsion_idxs = cp.where(repulsion_mask & ~ internal_mask)
        attraction_idxs = cp.where(attraction_mask)
        internal_idxs = cp.where(internal_mask & repulsion_mask)

        Fa = cp.zeros((internal_mask.shape[0],), dtype=cp.float32)
        Fa_rep = self.spring_rate_repulsion_cuda \
                 * cp.log(self.repulsion_range_cuda / (x[repulsion_idxs] - self.d_asymptote_cuda))
        Fa_att = self.spring_rate_attraction_cuda \
                 * ((self.d_separation_cuda - x[attraction_idxs]) / self.attraction_range_cuda) \
                 * cp.exp(self.c_cuda * (self.d_separation_cuda - x[attraction_idxs]) \
                          / self.d_separation_cuda)
        Fa_int = - self.spring_rate_repulsion_cuda \
                 * cp.log(self.repulsion_range_cuda / (x[internal_idxs] - self.d_asymptote_cuda))

        Fa[repulsion_idxs] = Fa_rep
        Fa[attraction_idxs] = Fa_att
        Fa[internal_idxs] = Fa_int

        return Fa

    def apply_forces_to_node_and_element_cuda(self, gpu: CudaMemory, N_idxs: cp.ndarray,
                                              E_idxs: cp.ndarray, Fa: cp.ndarray,
                                              n1toA: cp.ndarray):
        eta1 = gpu.N_eta[gpu.E_node_1[E_idxs]]
        eta2 = gpu.N_eta[gpu.E_node_2[E_idxs]]
        etaA = gpu.N_eta[N_idxs]

        r1 = gpu.N_pos[gpu.E_node_1[E_idxs]]
        r2 = gpu.N_pos[gpu.E_node_2[E_idxs]]
        rA = r1 + n1toA

        # First, find the anlge. To do this, we need the force from the node, in the element's
        # body system of coordinates
        u = gpu.vector_1_to_2[E_idxs]
        v = gpu.outward_normal[E_idxs]
        Fab = cp.stack((cp.sum(Fa * v, axis=1), cp.sum(Fa * u, axis=1)), axis=1)

        # Next, we determine the equivalent drag of the centre and the position of the centre of
        # drag
        etaD = eta1 + eta2
        rD = (eta1[:, None] * r1 + eta2[:, None] * r2) / etaD[:, None]

        # We then need the vector from the centre of drag to both nodes (note, these are relative
        # the fixed system of coordinates, not the body system of coordinates)
        rDto1 = r1 - rD
        rDto2 = r2 - rD
        rDtoA = rA - rD

        # These give us the moment of drag about the centre of drag
        ID = eta1 * cp.sum(rDto1 ** 2, axis=1) + eta2 * cp.sum(rDto2 ** 2, axis=1)

        # The moment created by the node is then force times perpendicular distance.  We must use
        # the body system of coordinates in order to get the correct direction. (We could
        # probably get away without the transform, since we only need the length, but we'd have
        # to be careful about choosing the sign correctly)
        rDtoAb = cp.stack((cp.sum(rDtoA * v, axis=1), cp.sum(rDtoA * u, axis=1)), axis=1)

        # The moment is technically rDtoAby * Fabx - rDtoAbx * Faby but by definition, the y-axis
        # aligns with the element, so all x components are 0
        # todo: the above statement changes when friction between cells comes into play
        M = -rDtoAb[:, 1] * Fab[:, 0]

        # Now we can find the change in angle in the given time step
        a = self.dt_cuda * M / ID

        # This angle can now be used in a rotation matrix to determine the new position of the
        # nodes. We can apply it directly to rDto1 and rDto2 since the angle is in the plance (a
        # consequence of 2D)
        Rot = cp.stack((cp.stack((cp.cos(a), -cp.sin(a)), axis=1),
                        cp.stack((cp.sin(a), cp.cos(a)), axis=1)), axis=1)

        # Need to transpose the vector so we can apply the rotation
        rDto1_new = (Rot @ rDto1[:, :, None]).squeeze(axis=2)
        rDto2_new = (Rot @ rDto2[:, :, None]).squeeze(axis=2)

        # Now shift from element system of coordinates to global
        r1f = rD + rDto1_new
        r2f = rD + rDto2_new

        # Translate the nodes with the linear motion
        r1f += (self.dt_cuda * Fa) / etaD[:, None]
        r2f += (self.dt_cuda * Fa) / etaD[:, None]

        # We now have the final position of the element after rotation
        #
        # The simulation represents elements as two nodes, so all of the forces have to be
        # realised there this means that any movement has to be in terms of a force on a node

        # The change in position

        dr1 = r1f - r1
        dr2 = r2f - r2

        # The force that would be applied to make a node move to its new position
        Fequiv1 = eta1[:, None] * dr1 / self.dt_cuda
        Fequiv2 = eta2[:, None] * dr2 / self.dt_cuda

        gpu.N_for[gpu.E_node_1[E_idxs]] += Fequiv1
        gpu.N_for[gpu.E_node_2[E_idxs]] += Fequiv2
        gpu.N_for[N_idxs] += -Fa
