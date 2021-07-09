from abc import ABC

import torch
from torch import Tensor

from src.components.cell.element import Element
from src.components.forces.neighbourhoodbasedforce.abstractneighbourhoodbasedforce import \
    AbstractNeighbourhoodBasedForce
from src.components.node.node import Node
from src.utils.errors import TodoException
from src.utils.tools import pyout


class AbstractNodeElementForce(AbstractNeighbourhoodBasedForce, ABC):
    def __init__(self):
        """
        Applies a force to push the angle of a cell corner towards its prefered value
        """
        self.r = None
        self.dt = None

    def apply_forces_to_node_and_element(self, n: Node, e: Element, Fa: Tensor, n1toA: Tensor):
        """
        This takes a node, an element, a force and a point and uses them to work out the forces
        applied to the node and the element

        Fa is the force pointing to the element. It uses the drag dominated equations for a
        rigid body

        To solve the motion, we need to account for linear movement and rotational movement. To
        do this, we solve the angular velocity of the element in its body system of coordinates.
        This requires a "moment of drag" for the element, based on its length and the drag
        coefficients of its nodes. We produce an angle that the element rotates through during
        the time step. In addition to the rotation, we solve the linear motion of the element at
        its "centre of drag", again, determined by its length and the drag coefficients of its
        nodes. This produces a vector that the centre of drag moves along in the given time
        interval. Once we have both the angle and vector, the rotation is aplied first,
        moving the nodes to their rotated position assuming no linear movement, then the linear
        movement is applied to each node.

        This process produces a final position after rotation and translation, however, in order
        to apply this to the simulation we need to convert this back into an equivalent force in
        purely linear motion.
        :param n:
        :param e:
        :param Fa: the force pointing to the element
        :param nltoA:
        :return:
        """
        eta1 = e.node_1.eta
        eta2 = e.node_2.eta
        etaA = n.eta

        r1 = e.node_1.position
        r2 = e.node_2.position
        rA = r1 + n1toA

        # First, find the angle. To do this, we need the force from the node, in the element's
        # body system of coordinates

        u = e.get_vector_1_to_2()
        v = e.get_outward_normal()

        Fab = torch.stack((torch.dot(Fa, v), torch.dot(Fa, u)), dim=0)

        # Next, we determine the equivalent drag of the centre and the position of the centre of
        # drag
        etaD = eta1 + eta2
        rD = (eta1 * r1 + eta2 * r2) / etaD

        # We then need the vector from the centre of drag to both nodes (note, these are relative
        # the fixed system of coordinates, not the body system of coordinates)
        rDto1 = r1 - rD
        rDto2 = r2 - rD
        rDtoA = rA - rD

        # These give us the moment of drag about the centre of drag
        ID = eta1 * torch.sum(rDto1 ** 2) + eta2 * torch.sum(rDto2 ** 2)

        # The moment created by the node is then force times perpendicular distance.  We must use
        # the body system of coordinates in order to get the correct direction. (We could
        # probably get away without the transform, since we only need the length, but we'd have
        # to be careful about choosing the sign correctly)
        rDtoAb = torch.stack((torch.dot(rDtoA, v), torch.dot(rDtoA, u)), dim=0)

        # The moment is technically rDtoAby * Fabx - rDtoAbx * Faby but by definition, the y-axis
        # aligns with the element, so all x components are 0
        # todo: the above statement changes when friction between cells comes into play
        M = -rDtoAb[1] * Fab[0]

        # Now we can find the change in angle in the given time step
        a = self.dt * M / ID

        # This angle can now be used in a rotation matrix to determine the new position of the
        # nodes. We can apply it directly to rDto1 and rDto2 since the angle is in the plance (a
        # consequence of 2D)
        Rot = torch.tensor([[torch.cos(a), -torch.sin(a)], [torch.sin(a), torch.cos(a)]])

        # Need to transpose the vector so we can apply the rotation
        rDto1_new = Rot @ rDto1.T
        rDto2_new = Rot @ rDto2.T

        # Now shift from element system of coordinates to global
        r1f = rD + rDto1_new
        r2f = rD + rDto2_new

        # Translate the nodes with the linear motion
        r1f += (self.dt * Fa) / etaD
        r2f += (self.dt * Fa) / etaD

        # We now have the final position of the element after rotation
        #
        # The simulation represents elements as two nodes, so all of the forces have to be
        # realised there this means that any movement has to be in terms of a force on a node

        # The change in position
        dr1 = r1f - r1
        dr2 = r2f - r2

        # The force that would be applied to make a node move to its new position
        Fequiv1 = eta1 * dr1 / self.dt
        Fequiv2 = eta2 * dr2 / self.dt

        e.node_1.add_force_contribution(Fequiv1)
        e.node_2.add_force_contribution(Fequiv2)
        n.add_force_contribution(-Fa)
