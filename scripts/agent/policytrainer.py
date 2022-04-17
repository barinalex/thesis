from scripts.constants import DT, Dirs
from scripts.environments.environment import Environment
from scripts.engine.engine import Engine
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3 import PPO, A2C, SAC
import numpy as np
from typing import Callable, List


class PolicyTrainer:
    def __init__(self, engine: Engine, config: dict, multiprocessing: bool = False):
        self.multiprocessing = multiprocessing
        self.config = config
        self.engine = engine
        self.env = self.makevectorizedenv() if multiprocessing else None
        self.callbacks = []
        self.alg = self.definealgorithm()

    def definealgorithm(self):
        """
        :return: stable-baseline3 algorithm instance. PPO, TD3 ...
        """
        return PPO(policy=self.config['policy'],
                   env=self.env,
                   verbose=1,
                   learning_rate=self.config['learning_rate'],
                   ent_coef=self.config['ent_coef'],
                   gamma=self.config['gamma'],
                   clip_range=self.config['clip_range'],
                   policy_kwargs=eval(self.config['policy_kwargs']),
                   device='cpu'
                   )

    def save(self):
        """
        Store policy parameters
        """
        pass

    def train(self):
        """
        :return: stable-baseline3 trained algorithm instance. PPO, TD3 ...
        """
        self.alg.learn(total_timesteps=self.config['timesteps'],
                       callback=self.callbacks)

    def definecallback(self):
        """
        :return: stable-baseline3 callback instance
        """
        pass

    def initcallbacks(self) -> List[BaseCallback]:
        """
        :return: list of callback for policy training
        """
        pass

    def makeenv(self) -> Environment:
        """
        :return: configured environment
        """
        return Environment(self.config, self.engine)

    def makesubenv(self, rank: int, seed: int = 0) -> Callable:
        """
        Utility function for multiprocessed env.
        :param seed: the initial seed for RNG
        :param rank: index of the subprocess
        :return: function that return an env for a subprocess
        """
        def envinit() -> Environment:
            env = self.makeenv()
            env.seed(seed + rank)
            return env
        set_random_seed(seed)
        return envinit

    def makevectorizedenv(self) -> SubprocVecEnv:
        """
        :return: vectorized environment
        """
        return SubprocVecEnv([self.makesubenv(rank=i) for i in range(self.config["n_cpu"])], start_method='fork')