from math import floor
from typing import List

import torch
from torch import Tensor, tensor

from biobots2D.components.cell.element import Element
from biobots2D.components.node.node import Node
from utils import TodoException
from utils.polyshapes import inpolygon
from utils.tools import pyout


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
            self.put_node_in_box(t.node_list[i])

        for i in range(len(t.element_list)):
            # If the element is an internal element, skip it because they don't interact with nodes
            e = t.element_list[i]
            if not e.is_element_internal():
                self.put_element_in_boxes(e)

    def get_neighbouring_elements(self, n: Node, r: float):
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

        b = []

        if self.only_boxes_in_proximity:
            b = self.assemble_candidate_elements(n, r)
        else:
            raise TodoException

        neighbours: List[Element] = []

        for e in b:
            u = e.get_vector_1_to_2()
            v = e.get_outward_normal()

            # Make box around element; determine if node is in that box
            n1 = e.node_1
            n2 = e.node_2

            p1 = n1.position + v * r
            p2 = n1.position - v * r
            p3 = n2.position - v * r
            p4 = n2.position + v * r

            # x, y = torch.chunk(torch.stack((p1, p2, p3, p4), dim=0), 2, dim=1)
            # x, y = x.squeeze(1), y.squeeze(1)

            poly = torch.stack((p1, p2, p3, p4), dim=0)

            inside = inpolygon(n.x, n.y, poly)

            if inside:
                neighbours.append(e)

        return neighbours

    def get_neighbouring_nodes(self, nl, r):
        """
        Given the node n and the radius r find all the nodes that are neighbours.
        :param nl:
        :param r:
        :return:
        """
        raise TodoException

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
        raise TodoException

    def get_all_adjacent_element_boxes(self, n):
        """
        Grab all potential elements form the 8 adjacent boxes
        :param n:
        :return:
        """
        raise TodoException

    def get_all_adjacent_node_boxes(self, n):
        """
        Grab all potential Nodes from the 8 adjacent boxes
        :param n:
        :return:
        """
        raise TodoException

    def assemble_candidate_elements(self, n: Node, r: float):
        """

        :param n:
        :param r:
        :return:
        """
        # if n.id == 2:
        #     pyout()

        b: List[Element] = [e for e in self.get_element_box_from_node(n)]

        # Then check if the node is near a boundary
        # Need to decide if the process of check is more effort than just taking the adjacent
        # boxes always, even when the node is in the middle of its box

        if floor(n.x / self.dx) != floor((n.x - r) / self.dx):  # left
            b.extend(self.get_adjacent_element_box_from_node(n, [-1, 0]))
        if floor(n.x / self.dx) != floor((n.x + r) / self.dx):  # right
            b.extend(self.get_adjacent_element_box_from_node(n, [1, 0]))
        if floor(n.y / self.dy) != floor((n.y - r) / self.dy):  # bottom
            b.extend(self.get_adjacent_element_box_from_node(n, [0, -1]))
        if floor(n.y / self.dy) != floor((n.y + r) / self.dy):  # top
            b.extend(self.get_adjacent_element_box_from_node(n, [0, 1]))
        if (floor(n.x / self.dx) != floor((n.x - r) / self.dx)
                and floor(n.y / self.dy) != floor((n.y - r) / self.dy)):  # left bottom
            b.extend(self.get_adjacent_element_box_from_node(n, [-1, -1]))
        if (floor(n.x / self.dx) != floor((n.x + r) / self.dx)
                and floor(n.y / self.dy) != floor((n.y - r) / self.dy)):  # right bottom
            b.extend(self.get_adjacent_element_box_from_node(n, [1, -1]))
        if (floor(n.x / self.dx) != floor((n.x - r) / self.dx)
                and floor(n.y / self.dy) != floor((n.y + r) / self.dy)):  # left top
            b.extend(self.get_adjacent_element_box_from_node(n, [-1, 1]))
        if (floor(n.x / self.dx) != floor((n.x + r) / self.dx)
                and floor(n.y / self.dy) != floor((n.y + r) / self.dy)):  # right top
            b.extend(self.get_adjacent_element_box_from_node(n, [1, 1]))

        # Remove duplicates (elements can be in multiple boxes)
        b = self.quick_unique(b)

        # Remove nodes own elements
        b = sorted(list(set(b) - set(n.element_list)), key=lambda x: x.id)

        # Remove elements from the cell the node is in. It does not interact with them except
        # indirectly via a volume force

        for c in n.cell_list:
            b = sorted(list(set(b) - set(c.element_list)), key=lambda x: x.id)

        return b

    def assemble_candidate_nodes(self, n, r):
        """

        :param n:
        :param r:
        :return:
        """
        raise TodoException

    def quick_unique(self, b):
        """

        :param b:
        :return:
        """
        return sorted(list(set(b)), key=lambda x: x.id)

    def put_node_in_box(self, n: Node):
        """

        :param n:
        :return:
        """

        q, i, j = self.get_quadrant_and_indices(n.x, n.y)

        self.insert_node(q, i, j, n)

    def put_element_in_boxes(self, e: Element):
        """
        Given the list of elements that a given node is part of, distribute the elements to the
        element boxes. This will require putting elements in intermediate boxes too.
        :param e:
        :return:
        """

        n1 = e.node_1
        n2 = e.node_2

        Q, I, J = self.get_box_indices_between_nodes(n1, n2)

        for ii in range(Q.size(0)):
            q, i, j = Q[ii], I[ii], J[ii]

            self.insert_element(Q[ii], I[ii], J[ii], e)

    def get_node_box_from_node(self, n):
        """
        Returns the same box that n is in
        :param n:
        :return:
        """
        raise TodoException

    def get_element_box_from_node(self, n):
        """
        Returns the same box that n is in
        :param n:
        :return:
        """
        # Returns the same box that n is in
        q, i, j = self.get_quadrant_and_indices(n.x, n.y)

        try:

            return self.elements_Q[q][i][j]
        except KeyError:
            raise KeyError(f"Elements don't exist where expected in the partition (q{q}, ({i}, "
                           f"{j})")

    def get_adjacent_indices_from_node(self, n: Node, direction: List[int]):
        """

        :param n:
        :param direction: [a, b] where a,b = 1 or -1
                          1 indicates an increase in the global index and -1 a decrease
                          a is applied to I and b is applied to J
        :return:
        """

        a, b = direction
        q, i, j = self.get_quadrant_and_indices(n.x, n.y)
        I, J = self.convert_to_global(q, i, j)

        return self.convert_to_quadrant(I + a, J + b)

    def get_adjacent_node_box_from_node(self, n, direction):
        """
        Returns the node box adjacent to the one indicated specifying the directionection
        :param n:
        :param direction:
        :return:
        """
        raise TodoException

    def get_adjacent_element_box_from_node(self, n: Node, direction: List[int]):
        """
        Returns the node box adjacent to the one indicated specifying the direction
        :param n:
        :param direction:
        :return:
        """

        q, i, j = self.get_adjacent_indices_from_node(n, direction)

        try:
            return self.elements_Q[q][i][j]
        except IndexError:
            return []

    def get_box_indices_between_points(self, pos1: Tensor, pos2: Tensor):
        """
        Given two points, we want all the indices between them in order to determine which boxes
        an element needs to go in
        :param pos1:
        :param pos2:
        :return:
        """

        q1, i1, j1 = self.get_quadrant_and_indices(pos1[0], pos1[1])
        q2, i2, j2 = self.get_quadrant_and_indices(pos2[0], pos2[1])

        qp, ip, jp = self.make_element_box_list(q1, i1, j1, q2, i2, j2)

        return qp, ip, jp

    def get_box_indices_between_nodes(self, n1: Node, n2: Node):
        """
        Given two nodes, we want all the indices between them
        :param n1:
        :param n2:
        :return:
        """

        ql, il, jl = self.get_box_indices_between_points(n1.position, n2.position)

        return ql, il, jl

    def get_box_indices_between_nodes_previous(self, n1, n2):
        """
        Redundant

        Given two nodes, we want all the indices between their previous positions
        :param n1:
        :param n2:
        :return:
        """
        raise TodoException

    def get_box_indices_between_nodes_previous_current(self, n1: Node, n2: Node):
        """
        Given two nodes, we want all the indices between n1s previous and n2s current positions
        :param n1:
        :param n2:
        :return:
        """

        return self.get_box_indices_between_points(n1.previous_position, n2.position)

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

        :param q1:
        :param i1:
        :param j1:
        :param q2:
        :param i2:
        :param j2:
        :return:
        """

        # To find the boxes that the element could pass through it is much simpler to convert to
        # global indices, then back to quadrants.
        I1, J1 = self.convert_to_global(q1, i1, j1)
        I2, J2 = self.convert_to_global(q2, i2, j2)

        Il = torch.tensor(range(I1, I2 + 1) if I1 < I2 else range(I2, I1 + 1))
        Jl = torch.tensor(range(J1, J2 + 1) if J1 < J2 else range(J2, J1 + 1))

        # since MATLAB indexes from 1 instead of 0, I think conversion to global should take this
        # into account

        # MATLAB                                 PYTHON
        #         (-1, 2) | ( 1,  2)                      (-1,  1) | ( 0,  1)
        # (-2, 1) (-1, 1) | ( 1,  1) (2,  1)     (-2,  0) (-1,  0) | ( 0,  0) ( 1,  0)
        # ----------------+-----------------     ------------------+------------------
        # (-2,-1) (-1,-1) | ( 1, -1) (2, -1)     (-2, -1) (-1, -1) | ( 0, -1) ( 1, -1)
        #         (-1,-2) | ( 1, -2)                      (-1, -2) | ( 0, -2)

        # Note that intermediate solutions p and q do not match MATLAB implementation since
        # meshgrid and flatten go along different dimension in MATLAB
        p, q = torch.meshgrid(Il, Jl)
        # p, q = [x.T for x in torch.meshgrid(Il, Jl)]

        Il = p.reshape(-1)
        Jl = q.reshape(-1)

        ql, il, jl = self.convert_to_quadrant(Il, Jl)

        return ql, il, jl

    def update_box_for_node(self, n: Node):
        """

        :param n:
        :return:
        """
        qn, in_, jn = self.get_quadrant_and_indices(n.x, n.y)
        qo, io_, jo = self.get_quadrant_and_indices(*n.previous_position)

        if not (qn == qo and in_ == io_ and jn == jo):
            # The given node is in a different box compared to previous timestep/position,
            # so need to do some adjusting.

            self.insert_node(qn, in_, jn, n)

            self.nodes_Q[qo][io_][jo].remove(n)

            # Also need to adjust the elements
            self.update_boxes_for_elements_using_node(n)



    def update_box_for_node_adjusted(self, n):
        """
        Used when manually moving a node to a new position a special method is needed since
        previousPosition is not changed in this propcess
        :param n:
        :return:
        """
        raise TodoException

    def update_boxes_for_elements_using_node(self, n1: Node):
        """
        This function will be used as each node is moved As such, we know the node n1 has _just_
        moved therefore we need to look at the current position and the previous position to see
        which boxes need changing. We know nothing about the tutorials nodes of the elements so at
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

        for e in n1.element_list:
            if not e.is_element_internal():
                n2 = e.get_other_node(n1)

                qn, in_, jn = self.get_box_indices_between_nodes(n1, n2)
                qo, io_, jo = self.get_box_indices_between_nodes_previous_current(n1, n2)

                self.move_element_to_new_boxes(torch.stack((qo, io_, jo), dim=1),
                                               torch.stack((qn, in_, jn), dim=1), e)


    def update_boxes_for_elements_using_node_adjusted(self, n1):
        """
        This function does the same as UpdateBoxesForElementsUsingNode when the given node is
        moved by adjusting it's position manually
        :param n1:
        :return:
        """
        raise TodoException

    def update_boxes_for_element(self, e):
        """
        This function will be run in the simulation. It will be done after all the nodes have moved
        :param e:
        :return:
        """
        raise TodoException

    def move_element_to_new_boxes(self, old: Tensor, new: Tensor, e: Element):
        """

        :param old: the set of boxes the element used to be in
        :param new: the set that it should be in now
        :param e:
        :return:
        """

        ql, il, jl = [v.squeeze(-1) for v in new.chunk(3, dim=-1)]
        qp, ip, jp = [v.squeeze(-1) for v in old.chunk(3, dim=-1)]

        L = new.unsqueeze(1).expand(new.size(0), old.size(0), -1)
        P = old.unsqueeze(0).expand(new.size(0), old.size(0), -1)

        EQ = ~torch.all(torch.eq(L, P), dim=-1)

        # Get the unique new boxes
        J = torch.all(EQ, dim=1).nonzero().squeeze(0)

        for ii in J:
            self.insert_element(ql[ii], il[ii], jl[ii], e)

        # Get the old boxes and the indices to remove
        J = torch.all(EQ, dim=0).nonzero().squeeze(0)

        for ii in J:
            self.remove_element_from_box(qp[ii], ip[ii], jp[ii], e)

    def insert_node(self, q: Tensor, i: Tensor, j: Tensor, n: Node):
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
        q, i, j = [int(val.item()) for val in (q, i, j)]
        try:
            if n in self.nodes_Q[q][i][j]:
                pyout(f"Node {n} already in elements_Q[{q}][{i}][{j}]")
            else:
                self.nodes_Q[q][i][j].append(n)
        except IndexError:
            while len(self.nodes_Q[q]) <= i:
                self.nodes_Q[q].append([])
            while len(self.nodes_Q[q][i]) <= j:
                self.nodes_Q[q][i].append([])
            self.nodes_Q[q][i][j].append(n)

    def insert_element(self, q, i, j, e):
        """

        :param q:
        :param i:
        :param j:
        :param e:
        :return:
        """
        q = int(q.item())
        i = int(i.item())
        j = int(j.item())

        try:
            if e in self.elements_Q[q][i][j]:
                pyout(f"Element {e} already in elements_Q")
            else:
                self.elements_Q[q][i][j].append(e)
        except IndexError:
            while len(self.elements_Q[q]) <= i:
                self.elements_Q[q].append([])
            while len(self.elements_Q[q][i]) <= j:
                self.elements_Q[q][i].append([])
            self.elements_Q[q][i][j].append(e)

    def remove_element_from_box(self, q, i, j, e):
        """
        If it gets to this point, the elemet should be in the given box. If it's not, could be a
        sign of tutorials problems but the simulation can continue
        :param q:
        :param i:
        :param j:
        :param e:
        :return:
        """
        self.elements_Q[q][i][j].remove(e)

    def remove_node_from_partition(self, n):
        """
        Used when cells die
        :param n:
        :return:
        """
        raise TodoException

    def remove_element_from_partition(self, e):
        """
        Used when cells die. Assumes that the element is completely up to date
        :param e:
        :return:
        """
        raise TodoException

    def repair_modified_element(self, e):
        """
        One or more of the nodes has been modified, so we need to fix the boxes
        :param e:
        :return:
        """
        raise TodoException

    def get_node_box(self, x, y):
        """
        Given a pair of coordinates, access the matching box
        :param x:
        :param y:
        :return:
        """
        raise TodoException

    def is_node_in_new_box(self, n):
        """
        Redundant now, but leaving for testing
        :param n:
        :return:
        """
        raise TodoException

    def get_element_box(self, x, y):
        """
        given a pair of coordinates, access the matching box
        :param x:
        :param y:
        :return:
        """
        raise TodoException

    def get_quadrant_and_indices(self, x, y):
        """
        N.B. this returns different number than MATLAB implementation, because python is a
        sensible language that counts from 0 ¯\_(ツ)_/¯
        :param x:
        :param y:
        :return:
        """

        q = self.get_quadrant(x, y)
        i, j = self.get_indices(x, y)
        return q, i, j

    def get_global_indices(self, x, y):
        """
        The indices we would have if matlab could handle negative indices

        redundant? (python handles negative indices)
        :param x:
        :param y:
        :return:
        """
        raise TodoException

    def convert_to_global(self, q: Tensor, i: Tensor, j: Tensor):
        """

        :param q:
        :param i:
        :param j:
        :return:
        """

        # Note the difference in implementation between MATLAB and Python version, due to MATLAB
        # indexing from 0
        #
        # MATLAB                                 PYTHON
        #         (-1, 2) | ( 1,  2)                      (-1,  1) | ( 0,  1)
        # (-2, 1) (-1, 1) | ( 1,  1) (2,  1)     (-2,  0) (-1,  0) | ( 0,  0) ( 1,  0)
        # ----------------+-----------------     ------------------+------------------
        # (-2,-1) (-1,-1) | ( 1, -1) (2, -1)     (-2, -1) (-1, -1) | ( 0, -1) ( 1, -1)
        #         (-1,-2) | ( 1, -2)                      (-1, -2) | ( 0, -2)

        if q == 0:
            return i, j
        elif q == 1:
            return i, -j - 1
        elif q == 2:
            return -i - 1, -j - 1
        elif q == 3:
            return -i - 1, j
        else:
            raise ValueError(f"q must be 0, 1, 2, or 3; got {q} instead.")

    def convert_to_quadrant(self, I, J):
        """

        :param I:
        :param J:
        :return:
        """

        q = self.get_quadrant(I, J)

        # Note the additional translation for quadrants 1, 2, and 3 due to the difference between
        # chosen indexing schemes in the MATLAB and PYTHON implementations
        #
        # MATLAB                                 PYTHON
        #         (-1, 2) | ( 1,  2)                      (-1,  1) | ( 0,  1)
        # (-2, 1) (-1, 1) | ( 1,  1) (2,  1)     (-2,  0) (-1,  0) | ( 0,  0) ( 1,  0)
        # ----------------+-----------------     ------------------+------------------
        # (-2,-1) (-1,-1) | ( 1, -1) (2, -1)     (-2, -1) (-1, -1) | ( 0, -1) ( 1, -1)
        #         (-1,-2) | ( 1, -2)                      (-1, -2) | ( 0, -2)

        i = torch.abs(I) - 1 * ((q == 2) | (q == 3))
        j = torch.abs(J) - 1 * ((q == 1) | (q == 2))

        return q, i, j

    def get_indices(self, x, y):
        """
        Determine the indices.
        :param x:
        :param y:
        :return:
        """

        i = torch.floor(torch.abs(x / self.dx)).int()
        j = torch.floor(torch.abs(y / self.dy)).int()

        return i, j

    def get_quadrant(self, x, y):
        """
        Determine the correct quadrant
        0: (+,+)
        1: (+,-)
        2: (-,-)
        3: (-,+)

        N.B. this is different from MATLAB implementation, because python is a sensible language
        that counts from 0 ¯\_(ツ)_/¯
        :param x:
        :param y:
        :return:
        """

        if x.ndim:
            q = torch.zeros(x.size(), dtype=torch.int64)
            q = torch.where((x >= 0) & (y < 0), torch.full_like(q, 1), q)
            q = torch.where((x < 0) & (y < 0), torch.full_like(q, 2), q)
            q = torch.where((x < 0) & (y >= 0), torch.full_like(q, 3), q)

            return q
        else:

            if x >= 0:
                if y >= 0:
                    return tensor(0)
                else:
                    return tensor(1)
            else:
                if y < 0:
                    return tensor(2)
                else:
                    return tensor(3)

    def debug(self):
        try:
            if len(self.elements_Q[1][1][0]) >= 22:
                pyout('Help')
        except IndexError:
            pass
