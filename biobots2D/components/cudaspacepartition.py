from utils import TodoException
import cupy as cp

class CudaSpacePartition:
    def __init__(self, cpu, N, E1, E2):
        self.dx = cp.float32(cpu.dx)
        self.dy = cp.float32(cpu.dy)

    def get_neighbouring_elements(self, n: cp.ndarray, r: cp.float32):



        raise TodoException

    def assemble_candidate_elements(self, n: cp.ndarray, r: cp.float32):
        raise TodoException

    def quick_unique(self, b):
        raise TodoException

    def put_node_in_box(self, b):
        raise TodoException

    def put_element_in_boxes(self, e):
        raise TodoException

