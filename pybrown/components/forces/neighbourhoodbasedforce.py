from abc import ABC, abstractmethod


class AbstractNeighbourhoodBasedForce(ABC):
    """
    This class gives the details for how a force will be applied to each cell (as opposed to each
    element, or the whole population)
    """

    @abstractmethod
    def add_neighbourhood_based_forces(self, node_list):
        pass

    def apply_forces_to_node_and_element(self, n, e, Fa, nltoA):
        """
        This takes a node, an element, a force, and a point and uses them to work out the forces
        applied to the node and the element.

        It uses the drag dominated equations of motion for a rigid body.

        To solve the motion, we need to account for linear movement and rotational movement. To
        do this, we solve the angular velocity of the element in its body system of coordinates.
        This requires a "moment of drag" for the element, based on its length and the drag
        coefficients of its nodes. We produce an angle that the element rotates through during
        the time step. In addition to the rotation, we solve the linear motion of the element at
        its "centre of drag", again, determined by its length and the drag coefficients of its
        nodes. This produces a vector that the centre of drag moves aling in the given time
        interval. Once we have both the angle and vector, the rotation is applied first,
        moving the nodes to their rotated position assuming no linear movement, then the linear
        movement is applied to each node.

        This process produces a final position after rotation and translation, however, in order
        to apply this to the simulation we need to convert this back into equivalent force in
        purely linear motion.

        :param n:
        :param e:
        :param Fa: the force pointing to the element
        :param nltoA:
        :return:
        """
        raise NotImplementedError
