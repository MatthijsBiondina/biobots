from abc import ABC, abstractmethod

from tqdm import tqdm

from biobots3D.graphics.graphicsengine3d import GraphicsEngine3D
from biobots3D.physics.physicsengine3d import PhysicsEngine3D
from biobots3D.simulation.abstractsimulation3d import AbstractSimulation3D
from utils.errors import TodoException
from utils.tools import pyout


class AbstractBiobotSimulation3D(AbstractSimulation3D):

    def __init__(self, seed, mode: str = "display", fname="default"):
        super(AbstractBiobotSimulation3D, self).__init__(seed)
        self.physics_engine = PhysicsEngine3D(self.gpu)
        self.graphics_engine = None if mode is None else GraphicsEngine3D(self.gpu, mode, fname)

    def animate(self, steps=3600, d_step=1):
        for step in tqdm(range(steps)):
            self.next_time_step()

            if step % d_step == 0:
                self.render()

    def next_time_step(self):
        self.physics_engine.compute_next_timestep()

    def render(self):
        if self.graphics_engine is not None:
            self.graphics_engine.render_opengl()  # pyout()
