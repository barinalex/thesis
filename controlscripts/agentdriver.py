from agent import Agent
from queuebuffer import QueueBuffer
from waypointer import Waypointer
from utils import load_raw_data
from state import State
import numpy as np
import os.path


class AgentDriver:
    def __init__(self):
        # policypath = os.path.join("data", "policy", "ppo_mjc_2022_05_01_19_11_03_544420.zip")
        policypath = os.path.join("data", "policy", "ppo_mlp_2022_05_01_18_29_08_505558.zip")
        n_wps = 10
        bufsize = 1
        pointspath = os.path.join("data", "points", "lap_pd02_r1_s2.npy")
        points = load_raw_data(path=pointspath)
        initvector = np.array([0, 0, 0])
        self.agent = Agent()
        self.agent.load(path=policypath)
        self.state = State(timestep=0.01)
        self.waypointer = Waypointer(n_wps=n_wps, points=points)
        self.actbuffer = QueueBuffer(size=bufsize, initvector=initvector)
        self.velbuffer = QueueBuffer(size=10, initvector=initvector)
        self.lin = np.zeros(3)
        self.ang = np.zeros(3)

    def updatestate(self):
        """
        Given velocities from sensor update imaginary state
        """
        self.state.set(vel=self.lin, ang=self.ang)
        self.state.update_pos()
        self.state.update_orn()

    def make_observation(self) -> np.ndarray:
        """
        :return: agent's state observation
        """
        obs = np.hstack((self.lin[:2], self.ang[2]))
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
        obs = self.make_observation()
        action = self.agent.act(observation=obs)
        return tuple(action)

    def update(self, lin: np.ndarray, ang: np.ndarray):
        """
        :param lin: linear velocity
        :param ang: angular velocity
        """
        self.lin = lin
        self.ang = ang
        self.updatestate()
        self.waypointer.update(pos=self.state.getpos()[:2])


if __name__ == "__main__":
    AgentDriver()
