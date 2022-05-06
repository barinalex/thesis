import time

from agent import Agent
from queuebuffer import QueueBuffer
from waypointer import Waypointer
from utils import load_raw_data
from state import State
import numpy as np
import os.path
import sys
import logging
logging.basicConfig(stream=sys.stdout, level=logging.INFO)


class AgentDriver:
    def __init__(self, policy: str = "mlp", trajectory: str = "lap"):
        policies = {"mlp": "ppo_mlp_2022_05_01_18_29_08_505558.zip",
                    "mjc": "ppo_mjc_2022_05_01_19_11_03_544420.zip"}
        policypath = os.path.join("data", "policy", policies[policy])
        n_wps = 10
        bufsize = 5 if policy == "mlp" else 1
        trajectories = {"inf": "inf_pd02_r1.npy",
                        "lap": "lap_pd02_r1_s2.npy",
                        "rand": "n1_wps500_smth50_mplr10.npy"}
        pointspath = os.path.join("data", "points", trajectories[trajectory])
        points = load_raw_data(path=pointspath)
        self.agent = Agent()
        logging.info(f"Load policy")
        self.agent.load(path=policypath)
        logging.info(f"Policy ready")
        self.state = State(timestep=0.01)
        self.waypointer = Waypointer(n_wps=n_wps, points=points)
        initvector = np.array([0, 0, 0])
        self.velbuffer = QueueBuffer(size=bufsize, initvector=initvector)
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
        wps = self.waypointer.get_waypoints_vector()
        wps = np.hstack((wps, np.zeros((wps.shape[0], 1))))
        wps = self.state.toselfframe(v=wps)
        return np.hstack((obs, wps[:, :2].flatten()))

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
        :return: True if waypoints was closed
        """
        obs = np.hstack((lin[:2], ang[2]))
        self.velbuffer.add(element=obs)
        vec = self.velbuffer.get_sequential_input()
        vel = np.mean(vec, axis=1).flatten()
        self.lin = np.hstack((vel[:2], np.zeros(1)))
        self.ang = np.hstack((np.zeros(2), vel[2]))
        self.updatestate()
        return self.waypointer.update(pos=self.state.getpos()[:2])


if __name__ == "__main__":
    AgentDriver()
