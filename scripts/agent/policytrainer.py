from scripts.constants import DT, Dirs
from scripts.environments.environment import Environment
from scripts.engine.engine import Engine
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.callbacks import EvalCallback
from scripts.datamanagement.datamanagement import saveconfig
from scripts.engine.mujocoengine import MujocoEngine
from stable_baselines3 import PPO, A2C, SAC
import numpy as np
import torch.nn as nn
from typing import Callable, List


class PolicyTrainer:
    def __init__(self, engine: Engine, config: dict, multiprocessing: bool = False):
        self.multiprocessing = multiprocessing
        self.config = config
        self.engine = engine
        self.env = self.makevectorizedenv() if multiprocessing else self.makeenv()
        self.callbacks = [self.define_evalcallback()]
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
                   device='cpu')

    def save(self, path: str):
        """
        Store policy parameters

        :param path: full save path
        """
        saveconfig(path=f"{path}.yaml", config=self.config)
        self.alg.save(path=path)

    def train(self):
        """
        :return: stable-baseline3 trained algorithm instance. PPO, TD3 ...
        """
        self.alg.learn(total_timesteps=self.config['timesteps'],
                       callback=self.callbacks)

    def define_evalcallback(self):
        """
        :return: stable-baseline3 callback instance
        """
        eval_freq = int(self.config['callback_freq']) # * self.config['n_cpu'])
        # path = os.path.join(Dirs.models, "mlp_2022_05_01_12_30_00_981419")
        # engine = TCNNBased(path=path, visualize=False)
        engine = MujocoEngine()
        # engine = MLPBased(path=path)
        config = self.config.copy()
        config["trajectories"] = config["evaltrajectories"]
        env = Environment(config, engine)

        return EvalCallback(env,
                            log_path=Dirs.policy,
                            eval_freq=eval_freq,
                            n_eval_episodes=10,
                            best_model_save_path=Dirs.policy,
                            deterministic=True,
                            render=False)

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


if __name__ == "__main__":
    from scripts.engine.mujocoengine import MujocoEngine
    from scripts.engine.tcnnbased import TCNNBased
    from scripts.engine.mlpbased import MLPBased
    from scripts.datamanagement.datamanagement import loadconfig
    from scripts.datamanagement.pathmanagement import gettimestamp
    import os
    path = os.path.join(Dirs.configs, "ppo.yaml")
    config = loadconfig(path=path)
    path = os.path.join(Dirs.configs, "env.yaml")
    config.update(loadconfig(path=path))
    # path = os.path.join(Dirs.models, "mlp_hist5_ppo_2022_05_05_16_37_13_929111")
    # engine = TCNNBased(path=path)
    # engine = MLPBased(path=path)
    engine = MujocoEngine()
    trainer = PolicyTrainer(engine=engine, config=config)
    trainer.train()
    timestamp = gettimestamp()
    path = os.path.join(Dirs.policy, f"mjc_free_ppo_{timestamp}")
    trainer.save(path=path)
