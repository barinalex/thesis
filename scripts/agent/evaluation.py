import numpy as np
import os
from scripts.constants import Dirs
from scripts.agent.agent import Agent
from scripts.datamanagement.datamanagement import loadconfig
from scripts.environments.environment import Environment
from scripts.engine.mujocoengine import MujocoEngine
from scripts.engine.tcnnbased import TCNNBased


def evaluationloop(env: Environment, agent: Agent, n: int) -> np.ndarray:
    """
    :param env: configured environment with gym interface
    :param agent: class with trained policy to act
        on observations from environment
    :param n: number of episodes
    :return: list of stats for each episodes [rewards sum, number of timesteps]
    """
    stats = np.zeros((n, 2))
    for i in range(n):
        done = False
        obs = env.make_observation(action=[-1, 0])
        while not done:
            action = agent.act(observation=obs)
            obs, reward, done, _ = env.step(action=action)
            print(obs)
            stats[i] += [reward, 1]
        env.reset()
    return stats


def evaluate_tcnn_based() -> np.ndarray:
    """
    :return: list of rewards for each episodes
    """
    config = loadconfig(os.path.join(Dirs.configs, "env.yaml"))
    path = os.path.join(Dirs.models, "tcnn_2022_04_22_11_27_58_275542")
    engine = TCNNBased(path=path, visualize=False)
    # config["trajectories"] = "n10_wps500_smth50_mplr10.npy"
    config["trajectories"] = "inf_r1.npy"
    env = Environment(config=config, engine=engine, random=False)
    path = os.path.join(Dirs.policy, "ppo_tcnn_2022_04_22_11_57_47_144995.zip")
    agent = Agent()
    agent.load(path=path)
    rewards = evaluationloop(env=env, agent=agent, n=2)
    return rewards


def evaluate_mujoco_based() -> np.ndarray:
    """
    :return: list of rewards for each episodes
    """
    engine = MujocoEngine(visualize=True)
    config = loadconfig(os.path.join(Dirs.configs, "env.yaml"))
    # config["trajectories"] = "n10_wps500_smth50_mplr10.npy"
    config["trajectories"] = "inf_r1.npy"
    env = Environment(config=config, engine=engine, random=False)
    path = os.path.join(Dirs.policy, "ppo_2022_04_18_12_00_09_493536.zip")
    agent = Agent()
    agent.load(path=path)
    rewards = evaluationloop(env=env, agent=agent, n=2)
    return rewards


def compare_tcnn2mujoco_based():
    tcnn_rws = evaluate_tcnn_based()
    mujoco_rws = evaluate_mujoco_based()
    print("TCNN REWARDS")
    print(tcnn_rws)
    print("MUJOCO REWARDS")
    print(mujoco_rws)


if __name__ == "__main__":
    compare_tcnn2mujoco_based()
