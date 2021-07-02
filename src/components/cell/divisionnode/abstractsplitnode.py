from abc import ABC, abstractmethod


class AbstractSplitNode(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def get_split_node(self, c):
        pass