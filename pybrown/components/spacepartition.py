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

    def __init__(self, dx, dy, t):
        # Each quadrant is part of the cartesian plane
        # 1: (+,+)
        # 2: (+,-)
        # 3: (-,-)
        # 4: (-,+)

        # nodesQ and elementsQ are lists containing the 4 quadrants. The index matches to the
        # quadrant number. Each quadrant is a cell matrix that matches the actual box location.
        # Each box is a cell vector of nodes or elements. So, all in all they are cell vectors of
        # cell arrays of vectors
        self.nodes_Q = [[], [], [], []]
        self.elements_Q = [[], [], [], []]

        # The syze of the cell arrays is dynamic, so will be regularly reallocated as the
        # simulation progresses. This could cause issues as the simulation gets large.

        # The lengths of the box edges
        self.dx, self.dy = dx, dy
        self.simulation = t

        # A flag stating if the partition will search only the boxes that a node is in or close
        # to, rather than all 8 surrounding boxes. For small dx,dy in comparision to the average
        # element length, taking all 8 will probably be more efficient than specifying the
        # precise boxes. If the boxes are quite large or the elements and nodes are quite dense,
        # this should be set to true, as it will reduce the number of comparison operations.

        self.only_boxes_in_proximity = True

        for i in range(len(t.node_list)):
            self.put_node_in_boxes(t.node_list[i])

        for i in range(len(t.element_list)):
            # If the element is an internal element, skip it because they don't interact with nodes
            e = t.element_list[i]
            if not e.is_element_internal():
                self.put_element_in_boxes(e)

    def get_neighbouring_elements(self, n, r):
        """
        A function that efficiently finds the set of elements that are within a distance r of the
        node n.

        There are two stages:
        1. Get the candidate elements. This includes grabbing elements form adjacent boxes if the
           node is close to a box boundary.
        2. Calculate the distances to each candidate element. This involves making sure the node
           is within the range of the element.

        The elements are assebled into a vector

        :param n: anchor node
        :param r: radius
        :return:
        """
        raise NotImplementedError

    def get_neighbouring_nodes(self, nl, r):
        """
        Given the node n and the radius r find all the nodes that are neighbours.
        :param nl:
        :param r:
        :return:
        """
        raise NotImplementedError

    def get_neighbouring_nodes_and_elements(self, n, r):
        """
        Finds the neighbouring nodes and elements at the same time, taking account of the obtuse
        angled element pairs necessitating node-node interactions.

        The nodes it finds are only part of edges in proximity. This does not work if NodeCells
        are in the simulation, as it does not find isolated nodes.
        :param n:
        :param r:
        :return:
        """
        raise NotImplementedError

    def get_all_adjacent_element_boxes(self, n):
        """
        Grab all potential elements form the 8 adjacent boxes
        :param n:
        :return:
        """
        raise NotImplementedError

    def get_all_adjacent_node_boxes(self, n):
        """
        Grab all potential Nodes from the 8 adjacent boxes
        :param n:
        :return:
        """
        raise NotImplementedError

    def assemble_candidate_elements(self, n, r):
        """

        :param n:
        :param r:
        :return:
        """
        raise NotImplementedError

    def assemble_candidate_nodes(self, n, r):
        """

        :param n:
        :param r:
        :return:
        """
        raise NotImplementedError

    def quick_unique(self, b):
        """

        :param b:
        :return:
        """
        raise NotImplementedError

    def put_node_in_box(self, n):
        """

        :param n:
        :return:
        """
        raise NotImplementedError

    def put_element_in_boxes(self, e):
        """
        Given the list of elements that a given node is part of, distribute the elements to the
        element boxes. This will require putting elements in intermediate boxes too.
        :param e:
        :return:
        """
        raise NotImplementedError

    def get_node_box_from_node(self, n):
        """
        Returns the same box that n is in
        :param n:
        :return:
        """
        raise NotImplementedError

    def get_element_box_from_node(self, n):
        """
        Returns the same box that n is in
        :param n:
        :return:
        """
        raise NotImplementedError

    def get_adjacent_indices_from_node(self, n, direction):
        """

        :param n:
        :param direction: [a, b] where a,b = 1 or -1
                          1 indicates an increase in the global index and -1 a decrease
                          a is applied to I and b is applied to J
        :return:
        """
        raise NotImplementedError

    def get_adjacent_node_box_from_node(self, n, direction):
        """
        Returns the node box adjacent to the one indicated specifying the directionection
        :param n:
        :param direction:
        :return:
        """
        raise NotImplementedError

    def get_adjacent_element_box_from_node(self, n, direction):
        """
        Returns the node box adjacent to the one indicated specifying the directionection
        :param n:
        :param direction:
        :return:
        """
        raise NotImplementedError

    def get_box_indices_between_points(self, pos1, pos2):
        """
        Given two points, we want all the indices between them in order to determine which boxes
        an element needs to go in
        :param pos1:
        :param pos2:
        :return:
        """
        raise NotImplementedError

    def get_box_indices_between_nodes(self, n1, n2):
        """
        Given two nodes, we want all the indices between them
        :param n1:
        :param n2:
        :return:
        """
        raise NotImplementedError

    def get_box_indices_between_nodes_previous(self, n1, n2):
        """
        Redundant

        Given two nodes, we want all the indices between their previous positions
        :param n1:
        :param n2:
        :return:
        """
        raise NotImplementedError

    def get_box_indices_between_nodes_previous_current(self, n1, n2):
        """
        Given two nodes, we want all the indices between n1s previous and n2s current positions
        :param n1:
        :param n2:
        :return:
        """
        raise NotImplementedError

    def make_element_box_list(self, q1, i1, j1, q2, i2, j2):
        """
        This method for finding the boxes that we should put the elements in is not exact. An
        exact method will get exactly the right boxes and no more but as a consequence, will need
        to be checked at every time step, which can slow things down. An exact method might be
        better when the box size is quite small in relation to the max element length. An exact
        method transverses the vector beteen the ttwo nodes and calculates the position where it
        crosses the box boundaries. It uses this to know which box to add the element to.

        A non exact method will look at all the possible boxes the element could pass through,
        given that we only know which boxes its end points are in. This will only need to be
        updated when a node moves to a new box.

        The non exact method used here is probably the greediest method and the least efficient
        in a small box case, but is quick, and arrives at the same answer when the boxes are the
        large, hence it is kept for now.

        To find the boxes that the element could pass through it is much simpler to convert to
        global indices, then back to quadrants.
        :param q1:
        :param i1:
        :param j1:
        :param q2:
        :param i2:
        :param j2:
        :return:
        """
        raise NotImplementedError

    def update_box_for_node(self, n):
        """

        :param n:
        :return:
        """
        raise NotImplementedError

    def update_box_for_node_adjusted(self, n):
        """
        Used when manually moving a node to a new position a special method is needed since
        previousPosition is not changed in this propcess
        :param n:
        :return:
        """
        raise NotImplementedError

    def update_boxes_for_elements_using_node(self, n1):
        """
        This function will be used as each node is moved As such, we know the node n1 has _just_
        moved therefore we need to look at the current position and the previous position to see
        which boxes need changing. We know nothing about the other nodes of the elements so at
        this point we just assume they are in their final position. This will cause doubling up
        of effort if both nodes end up moving to a new box, but this should be fairly rare
        occurrance.

        Logic of processing:
        If the node n1 is the first one from an element to move boxes, then we use the current
        position for n2. If node n1 is the second to move, then the current position for n2 will
        still be the correct to use.
        :param n1:
        :return:
        """
        raise NotImplementedError

    def update_boxes_for_elements_using_node_adjusted(self, n1):
        """
        This function does the same as UpdateBoxesForElementsUsingNode when the given node is
        moved by adjusting it's position manually
        :param n1:
        :return:
        """
        raise NotImplementedError

    def update_boxes_for_element(self, e):
        """
        This function will be run in the simulation. It will be done after all the nodes have moved
        :param e:
        :return:
        """
        raise NotImplementedError

    def move_element_to_new_boxes(self, old, new, e):
        """

        :param old: the set of boxes the element used to be in
        :param new: the set that it should be in now
        :param e:
        :return:
        """
        raise NotImplementedError

    def insert_node(self, q, i, j, n):
        """
        This is the sensible way to do this, but it doesn't always work properly
        if i > size(obj.nodesQ{q},1) || j > size(obj.nodesQ{q},2)
            obj.nodesQ{q}{i,j} = [n];
        else
            obj.nodesQ{q}{i,j}(end + 1) = n;
        end
        :param q:
        :param i:
        :param j:
        :param n:
        :return:
        """
        raise NotImplementedError

    def insert_element(self, q, i, j, e):
        """

        :param q:
        :param i:
        :param j:
        :param e:
        :return:
        """
        raise NotImplementedError

    def remove_element_from_box(self, q, i, j, e):
        """
        If it gets to this point, the elemet should be in the given box. If it's not, could be a
        sign of other problems but the simulation can continue
        :param q:
        :param i:
        :param j:
        :param e:
        :return:
        """
        raise NotImplementedError

    def remove_node_from_partition(self, n):
        """
        Used when cells die
        :param n:
        :return:
        """
        raise NotImplementedError

    def remove_element_from_partition(self, e):
        """
        Used when cells die. Assumes that the element is completely up to date
        :param e:
        :return:
        """
        raise NotImplementedError

    def repair_modified_element(self, e):
        """
        One or more of the nodes has been modified, so we need to fix the boxes
        :param e:
        :return:
        """
        raise NotImplementedError

    def get_node_box(self, x, y):
        """
        Given a pair of coordinates, access the matching box
        :param x:
        :param y:
        :return:
        """
        raise NotImplementedError

    def is_node_in_new_box(self, n):
        """
        Redundant now, but leaving for testing
        :param n:
        :return:
        """
        raise NotImplementedError

    def get_element_box(self, x, y):
        """
        given a pair of coordinates, access the matching box
        :param x:
        :param y:
        :return:
        """
        raise NotImplementedError

    def get_quadrant_and_indices(self, x, y):
        """

        :param x:
        :param y:
        :return:
        """
        raise NotImplementedError

    def get_global_indices(self, x, y):
        """
        The indices we would have if matlab could handle negative indices

        redundant? (python handles negative indices)
        :param x:
        :param y:
        :return:
        """
        raise NotImplementedError

    def convert_to_global(self, q, i, j):
        """

        :param q:
        :param i:
        :param j:
        :return:
        """
        raise NotImplementedError

    def convert_to_quadrant(self, I, J):
        """

        :param I:
        :param J:
        :return:
        """
        raise NotImplementedError

    def get_indices(self, x, y):
        """
        Determine the indices.
        :param x:
        :param y:
        :return:
        """
        raise NotImplementedError

    def get_quadrant(self, x, y):
        """
        Determine the correct quadrant
        1: (+,+)
        2: (+,-)
        3: (-,-)
        4: (-,+)
        :param x:
        :param y:
        :return:
        """
        raise NotImplementedError
