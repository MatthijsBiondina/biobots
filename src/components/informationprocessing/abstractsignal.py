from abc import ABC, abstractmethod

from src.components.simulation.cuda_memory import CudaMemory


class AbstractSignal:
    @abstractmethod
    def add_signal(self, gpu: CudaMemory):
        pass