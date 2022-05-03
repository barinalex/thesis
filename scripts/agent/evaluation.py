import numpy as np
import os
from scripts.constants import Dirs
from scripts.agent.agent import Agent
from scripts.datamanagement.datamanagement import loadconfig
from scripts.environments.environment import Environment
from scripts.engine.mujocoengine import MujocoEngine
from scripts.engine.tcnnbased import TCNNBased
from scripts.engine.mlpbased import MLPBased


history = {"act": [],
           }


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
        obs = env.make_observation()
        while not done:
            action = agent.act(observation=obs)
            history["act"].append(action)
            obs, reward, done, _ = env.step(action=action)
            # print(obs[:3])
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
    engine = TCNNBased(path=path, visualize=True)
    config["trajectories"] = "n10_wps500_smth50_mplr10.npy"
    config["trajectories"] = "inf_pd02_r1.npy"
    # config["trajectories"] = "lap_pd02_r1_s2.npy"
    env = Environment(config=config, engine=engine, random=False)
    path = os.path.join(Dirs.policy, "ppo_tcnn_2022_04_30_14_38_23_442446.zip")
    agent = Agent()
    agent.load(path=path)
    rewards = evaluationloop(env=env, agent=agent, n=n)
    return rewards


def evaluate_mlp_based(n: int) -> np.ndarray:
    """
    :param n: number of episodes

    :return: list of rewards for each episodes
    """
    config = loadconfig(os.path.join(Dirs.configs, "env.yaml"))
    path = os.path.join(Dirs.models, "mlp_2022_05_01_12_30_00_981419")
    # engine = TCNNBased(path=path, visualize=True)
    engine = MLPBased(path=path, visualize=True)
    # config["trajectories"] = "n10_wps500_smth50_mplr10.npy"
    # config["trajectories"] = "inf_pd02_r1.npy"
    config["trajectories"] = "lap_pd02_r1_s2.npy"
    env = Environment(config=config, engine=engine, random=False)
    path = os.path.join(Dirs.policy, "ppo_mlp_2022_05_01_18_29_08_505558.zip")
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
    config["trajectories"] = "n10_wps500_smth50_mplr10.npy"
    # config["trajectories"] = "inf_pd02_r1.npy"
    # config["trajectories"] = "lap_pd02_r1_s2.npy"
    env = Environment(config=config, engine=engine, random=False)
    path = os.path.join(Dirs.policy, "ppo_mjc_2022_05_01_19_11_03_544420.zip")
    agent = Agent()
    agent.load(path=path)
    rewards = evaluationloop(env=env, agent=agent, n=n)
    return rewards


def compare_custom2mujoco_based(n: int):
    """
    :param n: number of episodes

    :return: list of rewards for each episodes
    """
    custom_rws = evaluate_mlp_based(n=n)
    mujoco_rws = evaluate_mujoco_based(n=n)
    print("CUSTOM REWARDS")
    print(custom_rws)
    print(np.mean(custom_rws, axis=0))
    print(np.std(custom_rws, axis=0))
    print("MUJOCO REWARDS")
    print(mujoco_rws)
    print(np.mean(mujoco_rws, axis=0))
    print(np.std(mujoco_rws, axis=0))


if __name__ == "__main__":
    # mujoco_rws = evaluate_mujoco_based(n=1)
    # print(mujoco_rws)
    mlp_rws = evaluate_mlp_based(n=1)
    print(mlp_rws)
    # compare_custom2mujoco_based(n=5)
    history["act"] = np.asarray(history["act"])
    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(np.arange(len(history["act"])), history["act"][:, 0])
    # plt.plot(history["pos"][autoindices, 0], history["pos"][autoindices, 1])
    plt.show()

