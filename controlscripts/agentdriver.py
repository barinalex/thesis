from threading import Lock
from agent import Agent
from queuebuffer import QueueBuffer
from waypointer import Waypointer
from utils import load_raw_data
from state import State
import numpy as np
import os.path


class AgentDriver:
    def __init__(self):
        policypath = os.path.join("data", "policy", "ppo_2022_04_18_12_00_09_493536.zip")
        n_wps = 10
        bufsize = 1
        pointspath = os.path.join("data", "points", "lap_r1_s3.npy")
        points = load_raw_data(path=pointspath)
        initvector = np.array([0, 0, 0, -1, 0])
        self.agent = Agent()
        self.agent.load(path=policypath)
        self.state = State(timestep=0.005)
        self.waypointer = Waypointer(n_wps=n_wps, points=points)
        self.actbuffer = QueueBuffer(size=bufsize, initvector=initvector)
        self.velbuffer = QueueBuffer(size=10, initvector=initvector)
        self.lin = np.zeros(3)
        self.ang = np.zeros(3)
        self.action = np.array([-1, 0])
        self.actlock = Lock()
        self.odomlock = Lock()

    def updatestate(self):
        """
        Given velocities from sensor update imaginary state
        """
        with self.odomlock:
            self.state.set(vel=self.lin, ang=self.ang)
        self.state.update_pos()
        self.state.update_orn()

    def make_observation(self) -> np.ndarray:
        """
        :return: agent's state observation
        """
        obs = np.hstack((self.lin[:2], self.ang[2], self.action[0], self.action[1]))
        self.actbuffer.add(element=obs)
        state = self.actbuffer.get_vector()
        wps = self.waypointer.get_waypoints_vector()
        wps = np.hstack((wps, np.zeros((wps.shape[0], 1))))
        wps = self.state.toselfframe(v=wps)
        return np.hstack((state, wps[:, :2].flatten()))

    def act(self):
        """
        :return: throttle, steering
        """
        with self.actlock:
            self.updatestate()
            self.waypointer.update(pos=self.state.getpos()[:2])
            obs = self.make_observation()
            self.action = self.agent.act(observation=obs)
        return self.action[0], self.action[1]

    def updatevelocities(self, lin: np.ndarray, ang: np.ndarray):
        """
        :param lin: linear velocity
        :param ang: angular velocity
        """
        with self.odomlock:
            self.lin = lin
            self.ang = ang


if __name__ == "__main__":
    AgentDriver()
