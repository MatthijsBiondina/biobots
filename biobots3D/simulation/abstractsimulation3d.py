from abc import ABC

from utils.errors import TodoException


class AbstractSimulation3D(ABC):
    def __init__(self):
        raise TodoException