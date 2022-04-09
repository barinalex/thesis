import os.path

import pybullet as p
import pybullet_data
from scripts.constants import Dirs
from scripts.engine.modelbased import ModelBased
from scripts.datamanagement.datamanagement import loadconfig
import time


class Visualizer:
    """pybullet visualization"""
    def __init__(self, engine: ModelBased):
        self.engine = engine
        self.pcId = p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.loadURDF("plane.urdf", physicsClientId=self.pcId)
        urdf = os.path.join(Dirs.urdf, "buggy_diff.urdf")
        self.modelid = p.loadURDF(urdf, physicsClientId=self.pcId)
        path = os.path.join(Dirs.configs, "default.yaml")
        config = loadconfig(path=path)
        self.timeinterval = config["timeinterval"]
        self.last_step = time.time()

    def step(self):
        """move robot models to its new position"""
        p.resetBasePositionAndOrientation(self.modelid,
                                          self.engine.getpos(),
                                          self.engine.getorn(),
                                          physicsClientId=self.pcId)
        time.sleep(max(0, self.timeinterval - (time.time() - self.last_step)))
        self.last_step = time.time()

    def disconnect(self):
        """disconnect from physics client"""
        p.disconnect(physicsClientId=self.pcId)
