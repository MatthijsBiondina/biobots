from src.components.cell.abstractcell import AbstractCell
from src.components.cell.element import Element


class HeartCell(AbstractCell):
    def __init__(self, node_list, element_ids, id_):
        super(HeartCell, self).__init__()
        self.id = id_
        self.node_list = node_list
        self.cell_type = 1

        self.element_list = []
        for ii in range(len(node_list)):
            node_1 = node_list[ii]
            node_2 = node_list[(ii + 1) % len(node_list)]
            e = Element(node_1, node_2, id=element_ids[ii])
            self.element_list.append(e)
            node_1.cell_list.append(self)
            e.cell_list.append(self)

    def divide(self):
        pass

    def is_point_inside_cell(self, point):
        pass