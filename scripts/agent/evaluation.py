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
            # print(obs)
            stats[i] += [reward, 1]
        env.reset()
    return stats


def evaluate_tcnn_based(n: int) -> np.ndarray:
    """
    :param n: number of episodes

    :return: list of rewards for each episodes
    """
    config = loadconfig(os.path.join(Dirs.configs, "env.yaml"))
    path = os.path.join(Dirs.models, "tcnn_2022_04_24_19_54_44_249977")
    engine = TCNNBased(path=path, visualize=False)
    config["trajectories"] = "n10_wps500_smth50_mplr10.npy"
    config["trajectories"] = "inf_pd01_r1.npy"
    # config["trajectories"] = "lap_pd01_r1_s2.npy"
    env = Environment(config=config, engine=engine, random=False)
    path = os.path.join(Dirs.policy, "ppo_tcnn_2022_04_25_15_02_48_445186.zip")
    agent = Agent()
    agent.load(path=path)
    rewards = evaluationloop(env=env, agent=agent, n=n)
    return rewards


def evaluate_mujoco_based(n: int) -> np.ndarray:
    """
    :param n: number of episodes

    :return: list of rewards for each episodes
    """
    engine = MujocoEngine(visualize=False)
    config = loadconfig(os.path.join(Dirs.configs, "env.yaml"))
    # config["trajectories"] = "n10_wps500_smth50_mplr10.npy"
    config["trajectories"] = "inf_pd01_r1.npy"
    # config["trajectories"] = "lap_pd01_r1_s2.npy"
    env = Environment(config=config, engine=engine, random=False)
    path = os.path.join(Dirs.policy, "ppo_mjc_2022_04_25_14_17_26_866055.zip")
    agent = Agent()
    agent.load(path=path)
    rewards = evaluationloop(env=env, agent=agent, n=n)
    return rewards


def compare_tcnn2mujoco_based(n: int):
    """
    :param n: number of episodes

    :return: list of rewards for each episodes
    """
    tcnn_rws = evaluate_tcnn_based(n=n)
    mujoco_rws = evaluate_mujoco_based(n=n)
    print("TCNN REWARDS")
    print(tcnn_rws)
    print(np.mean(tcnn_rws, axis=0))
    print(np.std(tcnn_rws, axis=0))
    print("MUJOCO REWARDS")
    print(mujoco_rws)
    print(np.mean(mujoco_rws, axis=0))
    print(np.std(mujoco_rws, axis=0))


if __name__ == "__main__":
    # mujoco_rws = evaluate_mujoco_based(n=1)
    # print(mujoco_rws)
    compare_tcnn2mujoco_based(n=1)