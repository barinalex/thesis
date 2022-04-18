import os.path

import numpy as np
import pybullet as p
import pybullet_data
from scripts.constants import Dirs
from scripts.datamanagement.datamanagement import loadconfig
from scripts.utils.linalg_utils import get_pybullet_quaternion
import time
from collections import deque


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
        self.linesqueue = deque()
        self.n_wps = 10

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

    def addline(self, from_, to_, color=None):
        """
        Create debug line between trajectory points

        :param from_: line start, shape (2,)
        :param to_: line end, shape (2,)
        :param color: list shape (3,) with rgb values
        :return: id of a created line
        """
        if color is None:
            color = [0, 1, 0]
        from_ = np.hstack((from_, 0.005))
        to_ = np.hstack((to_, 0.005))
        id = p.addUserDebugLine(lineFromXYZ=from_, lineToXYZ=to_, lineWidth=5,
                                lineColorRGB=color, physicsClientId=self.pcId)
        self.linesqueue.append(id)
        self.n_wps -= 1
        if self.n_wps <= 0:
            self.popline()
            self.n_wps += 1

    def popline(self):
        """
        Remove oldest line
        """
        id = self.linesqueue.popleft()
        p.removeUserDebugItem(itemUniqueId=id, physicsClientId=self.pcId)

    def disconnect(self):
        """disconnect from physics client"""
        p.disconnect(physicsClientId=self.pcId)
