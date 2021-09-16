import random
from abc import ABC, abstractmethod
from typing import List, Tuple

import cupy as cp
import numpy as np
from numpy import pi, sin, cos, int64
import quaternion

from biobots3D.bots.abstractbot3d import AbstractBot3D
from biobots3D.memory.gpusimdata import SimulationDataGPU
from utils.errors import TodoException
from utils.tools import pyout, set_seed


class AbstractSimulation3D(ABC):
    def __init__(self, seed=49):
        super(AbstractSimulation3D, self).__init__()
        set_seed(seed)

        self.bots: List[AbstractBot3D] = []

        self.next_bbot_id, self.next_cell_id, self.next_edge_id, self.next_node_id = 0, 0, 0, 0
        self.next_surf_id = 0
        self.load_simulation_objects()
        del self.next_bbot_id, self.next_cell_id, self.next_edge_id, self.next_surf_id, \
            self.next_node_id

        # GPU data
        self.gpu: SimulationDataGPU = SimulationDataGPU()

        self.load_gpu()


    @abstractmethod
    def load_simulation_objects(self, *args):
        raise NotImplementedError

    def add_biobot(self,
                   bot: AbstractBot3D,
                   pos: np.ndarray = np.array([0, 0, 0]),
                   R: np.quaternion = np.quaternion(1., 0., 0., 0.)
                   ):
        bot.id = self.next_bbot_id
        self.next_bbot_id += 1

        bbot_data = bot.cpu

        self.next_cell_id = bbot_data.increment_cell_ids(self.next_cell_id)
        self.next_surf_id = bbot_data.increment_surf_ids(self.next_surf_id)
        self.next_edge_id = bbot_data.increment_edge_ids(self.next_edge_id)
        self.next_node_id = bbot_data.increment_node_ids(self.next_node_id)

        # self.next_cell_id += bbot_data.cell_ids.size
        #
        # bbot_data.node_ids += self.next_node_id
        # bbot_data.edge_n_0 += self.next_node_id
        # bbot_data.edge_n_1 += self.next_node_id
        # bbot_data.surf_n_0 += self.next_node_id
        # bbot_data.surf_n_1 += self.next_node_id
        # bbot_data.surf_n_2 += self.next_node_id
        # self.next_node_id += bbot_data.node_ids.size
        #
        # bbot_data.edge_ids += self.next_edge_id
        # self.next_edge_id += bbot_data.edge_ids.size
        #
        # bbot_data.surf_ids += self.next_surf_id
        # self.next_surf_id += bbot_data.surf_ids.size

        # Translate and rotate
        if R is None:  # generate random quaternion
            ang = np.random.uniform(-pi, pi)
            vec = np.random.uniform(-1, 1, 3)
            vec /= np.linalg.norm(vec)
            R = np.quaternion(cos(ang), sin(ang) * vec[0], sin(ang) * vec[1], sin(ang) * vec[2])

        bbot_data.node_pos -= np.mean(bot.cpu.node_pos, axis=0)
        bbot_data.node_pos = quaternion.rotate_vectors(R, bbot_data.node_pos)
        bbot_data.node_pos += pos

        self.bots.append(bot)

    def load_gpu(self):

        nr_of_bbots = len(self.bots)
        nr_of_cells = max([np.max(b.cpu.cell_ids) for b in self.bots]) + 1
        nr_of_surfs = max([np.max(b.cpu.surf_ids) for b in self.bots]) + 1
        nr_of_edges = max([np.max(b.cpu.edge_ids) for b in self.bots]) + 1
        nr_of_nodes = max([np.max(b.cpu.node_ids) for b in self.bots]) + 1

        B_ids = np.arange(nr_of_bbots)
        B_2_C = [np.empty(0)] * nr_of_bbots
        C_ids = np.arange(nr_of_cells)
        C_2_N = [np.empty(0)] * nr_of_cells
        N_ids = np.arange(nr_of_nodes)
        N_pos = np.empty((nr_of_nodes, 3))
        E_ids = np.arange(nr_of_edges)
        E_n_0 = np.empty(nr_of_edges, dtype=int64)
        E_n_1 = np.empty(nr_of_edges, dtype=int64)
        S_ids = np.arange(nr_of_surfs)
        S_n_0 = np.empty(nr_of_surfs, dtype=int64)
        S_n_1 = np.empty(nr_of_surfs, dtype=int64)
        S_n_2 = np.empty(nr_of_surfs, dtype=int64)

        del nr_of_bbots, nr_of_cells, nr_of_surfs, nr_of_edges, nr_of_nodes

        for bii, b in enumerate(self.bots):
            b = b.cpu

            for cii, c_id in enumerate(b.cell_ids):
                gpuii = np.argwhere(C_ids == c_id).squeeze(0)
                C_2_N[gpuii.squeeze(0)] = b.cell_2_N[cii]
                try:
                    B_2_C[bii][cii] = c_id
                except IndexError:
                    B_2_C[bii] = np.full(b.cell_ids.size, c_id)
            del cii, c_id

            for sii, s_id in enumerate(b.surf_ids):
                gpuii = np.argwhere(S_ids == s_id).squeeze(0)
                S_n_0[gpuii] = b.surf_n_0[sii]
                S_n_1[gpuii] = b.surf_n_1[sii]
                S_n_2[gpuii] = b.surf_n_2[sii]
            del sii, s_id

            for eii, e_id in enumerate(b.edge_ids):
                gpuii = np.argwhere(E_ids == e_id).squeeze(0)
                E_n_0[gpuii] = b.edge_n_0[eii]
                E_n_1[gpuii] = b.edge_n_1[eii]
            del eii, e_id

            for nii, n_id in enumerate(b.node_ids):
                gpuii = np.argwhere(N_ids == n_id).squeeze(0)
                N_pos[gpuii] = b.node_pos[nii]
            del nii, n_id
        del bii, b, gpuii

        del self.bots

        self.gpu.B_2_C = cp.array(B_2_C)
        self.gpu.B_ids = cp.array(B_ids)
        self.gpu.C_2_N = cp.array(C_2_N)
        self.gpu.C_ids = cp.array(C_ids)
        self.gpu.E_ids = cp.array(E_ids)
        self.gpu.E_n_0 = cp.array(E_n_0)
        self.gpu.E_n_1 = cp.array(E_n_1)
        self.gpu.N_ids = cp.array(N_ids)
        self.gpu.N_pos = cp.array(N_pos)
        self.gpu.S_ids = cp.array(S_ids)
        self.gpu.S_n_0 = cp.array(S_n_0)
        self.gpu.S_n_1 = cp.array(S_n_1)
        self.gpu.S_n_2 = cp.array(S_n_2)


