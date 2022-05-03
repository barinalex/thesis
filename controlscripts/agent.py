import torch
from stable_baselines3 import PPO
import numpy as np
import torch.nn as nn
# torch.device("cpu")


class Agent:
    def __init__(self):
        self.policy = None

    def load(self, path: str):
        """
        :param path: full path to a pretrained policy
        """
        self.policy = PPO.load(path, device="cpu")

    def act(self, observation: np.ndarray) -> np.ndarray:
        """
        :param observation: numpy array
        :return: action: [throttle, turn]
        """
        action, _state = self.policy.predict(observation, deterministic=True)
        return action
