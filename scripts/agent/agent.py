from scripts.constants import DT, Dirs
from stable_baselines3 import PPO
import numpy as np
import torch.nn as nn


class Agent:
    def __init__(self):
        self.policy = None

    def load(self, path: str):
        """
        :param path: full path to a pretrained policy
        """
        self.policy = PPO.load(path)

    def act(self, observation: np.ndarray) -> np.ndarray:
        """
        :param observation: numpy array
        :return: action: [throttle, turn]
        """
        action, _state = self.policy.predict(observation)
        return action


if __name__ == "__main__":
    import os
    path = os.path.join(Dirs.policy, "ppo_2022_04_18_11_10_45_430539.zip")
    agent = Agent()
    agent.load(path=path)

    from scripts.datamanagement.datamanagement import loadconfig
    from scripts.simulation.joystickinputwrapper import JoystickInputWrapper
    from scripts.environments.environment import Environment
    from scripts.engine.mujocoengine import MujocoEngine
    config = loadconfig(os.path.join(Dirs.configs, "env.yaml"))
    env = Environment(config=config, engine=MujocoEngine(visualize=True))
    done = False
    obs = env.make_observation(action=[-1, 0])
    while not done:
        action = agent.act(observation=obs)
        obs, reward, done, _ = env.step(action=action)
        print(obs)
