import os.path

import numpy as np
import pybullet as p
import pybullet_data
from scripts.constants import Dirs
from scripts.datamanagement.datamanagement import loadconfig
from scripts.utils.linalg_utils import get_pybullet_quaternion
import time


class Visualizer:
    """pybullet visualization"""
    def __init__(self):
        self.pcId = p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.loadURDF("plane.urdf", physicsClientId=self.pcId)
        urdf = os.path.join(Dirs.urdf, "buggy_diff.urdf")
        self.modelid = p.loadURDF(urdf, physicsClientId=self.pcId)
        path = os.path.join(Dirs.configs, "default.yaml")
        config = loadconfig(path=path)
        self.timeinterval = config["timeinterval"]
        self.last_step = time.time()

    def step(self, pos: np.ndarray, orn: np.ndarray):
        """
        Move car model to its new position

        :param pos: new car position, shape (3,)
        :param orn: new car orientation as a quaternion (w x y z)
        """
        orn = get_pybullet_quaternion(q=orn)
        p.resetBasePositionAndOrientation(self.modelid, pos, orn, physicsClientId=self.pcId)
        time.sleep(max(0, self.timeinterval - (time.time() - self.last_step)))
        self.last_step = time.time()

    def disconnect(self):
        """disconnect from physics client"""
        p.disconnect(physicsClientId=self.pcId)
