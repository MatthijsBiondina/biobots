from abc import ABC

from pybrown.components.forces.neighbourhoodbasedforce.abstractneighbourhoodbasedforce import \
    AbstractNeighbourhoodBasedForce
from pybrown.utils.errors import TodoException


class AbstractNodeElementForce(AbstractNeighbourhoodBasedForce, ABC):
    def __init__(self):
        """
        Applies a force to push the angle of a cell corner towards its prefered value
        """
        self.r = None
        self.dt = None

    def apply_forces_to_node_and_element(self, n, e, Fa, nltoA):
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
        raise TodoException
