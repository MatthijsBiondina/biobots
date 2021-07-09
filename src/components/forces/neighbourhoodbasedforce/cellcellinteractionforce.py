from typing import List, Union

import torch
from torch import tensor

from src.components.forces.neighbourhoodbasedforce.abstractnodeelementforce import \
    AbstractNodeElementForce
from src.components.node.node import Node
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

    def add_neighbourhood_based_forces(self, node_list: List[Node],
                                       p: Union[SpacePartition, None] = None):
        """
        Here we calculate the forces between nodes and edges, and nodes and nodes.

        The force in ForceLaw assumes a positive scalar force is a repulsion interaction.

        For a node-edge interaction, we have to use the semi-rigid body approach in
        ApplyForcesToNodeAndElement which is supplied by the parent object. To use the function,
        the force Fa is assumed to have positive sense towards the edge. This means that Fa will
        be positive for repulsion applied to an edge, or negative for repulsion applied to the node.

        For a node-node interaction, the forces must be added in this method
        :param node_list: a vector of nodes in the simulation p is the space partition of nodes
                          and edges in the simulation
        :param p:
        :return:
        """
        for n in node_list:
            if self.use_node_node_interactions:
                raise TodoException
            else:
                e_list = p.get_neighbouring_elements(n, self.d_limit)
                n_list: List[Node] = []

            for e in e_list:
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
