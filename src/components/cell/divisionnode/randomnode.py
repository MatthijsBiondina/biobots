from src.components.cell.divisionnode.abstractsplitnode import AbstractSplitNode
from src.utils.errors import TodoException


class RandomNode(AbstractSplitNode):
    def __init__(self):
        """
        This class sets out the required functions for working out the node in a free cell where
        division starts from
        """
        super().__init__()


    def get_split_node(self, c):
        raise TodoException