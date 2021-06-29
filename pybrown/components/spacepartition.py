class SpacePartition:
    """
    This class holds a space partition for all the nodes in a simulation. It distributes each
    node into a box based on its spatial position. This will help to minimize the space searching
    effort needed to find collisions and resolve them.

    The partition is regular, so that the width and height of each box is the same, called dx and
    dy. A node can instantly work out which box it is n by taking x//dx and y//dy. However,
    we also need a place to store all the nodes that are in the same box, so they can be compared for interactions.

    There are two main operations to perform here. The first is move nodes between boxes,
    and the second is to query neighbours. Depending on where the node is precisely found,
    the neighbours will either be found in the same box, or an adjacent box and nowhere else.

    This will also need to store which boxes an element passes through, in order to work out the
    node-edge interactions. Setting and moving the edge boxes will take a bit more work.

    We use matlab arrays to store the box contents, and we use a bit of trickery to allow -ve
    indices

    All of the box handling process will be done in the cell simulation this just implements the
    processing the simulation will call
    """
    def __init__(self):
        # Each quadrant is part of the cartesian plane
        # 1: (+,+)
        # 2: (+,-)
        # 3: (-,-)
        # 4: (-,+)

        # nodesQ and elementsQ are lists
        #