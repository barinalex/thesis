import numpy as np
import os
from scripts.constants import Dirs
from scripts.agent.agent import Agent
from scripts.datamanagement.utils import load_raw_data
from scripts.datamanagement.datamanagement import loadconfig
from scripts.environments.environment import Environment
from scripts.engine.mujocoengine import MujocoEngine
from scripts.engine.tcnnbased import TCNNBased
from scripts.engine.mlpbased import MLPBased
import glob


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
    config["trajectories"] = "lap_pd02_r1_s2.npy"
    env = Environment(config=config, engine=engine, random=False)
    path = os.path.join(Dirs.policy, "ppo_tcnn_2022_04_30_14_38_23_442446.zip")
    agent = Agent()
    agent.load(path=path)
    rewards = evaluationloop(env=env, agent=agent, n=n)
    return rewards


def evaluate_mlp_based(n: int, pname: str = "mlp_hist5_ppo_2022_05_05_16_37_13_929111.zip") -> np.ndarray:
    """
    :param n: number of episodes

    :return: list of rewards for each episodes
    """
    config = loadconfig(os.path.join(Dirs.configs, "env.yaml"))
    path = os.path.join(Dirs.models, "mlp_hist5_2022_05_05_11_23_43_430257")
    # engine = TCNNBased(path=path, visualize=True)
    engine = MLPBased(path=path, visualize=False)
    config["trajectories"] = "n10_wps500_smth50_mplr10.npy"
    # config["trajectories"] = "inf_pd02_r1.npy"
    # config["trajectories"] = "lap_pd02_r1_s2.npy"
    # config["trajectories"] = "n1_wps500_smth50_mplr10.npy"
    env = Environment(config=config, engine=engine, random=False)
    path = os.path.join(Dirs.policy, pname)
    agent = Agent()
    agent.load(path=path)
    rewards = evaluationloop(env=env, agent=agent, n=n)
    return rewards


def evaluate_mujoco_based(n: int, pname: str = "mjc_ppo_2022_05_05_18_07_46_972885.zip") -> np.ndarray:
    """
    :param n: number of episodes

    :return: list of rewards for each episodes
    """
    engine = MujocoEngine(visualize=True)
    config = loadconfig(os.path.join(Dirs.configs, "env.yaml"))
    config["trajectories"] = "n10_wps500_smth50_mplr10.npy"
    # config["trajectories"] = "inf_pd02_r1.npy"
    # config["trajectories"] = "lap_pd02_r1_s2.npy"
    # config["trajectories"] = "n1_wps500_smth50_mplr10.npy"
    env = Environment(config=config, engine=engine, random=False)
    path = os.path.join(Dirs.policy, pname)
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
    # print(custom_rws)
    print(np.mean(custom_rws, axis=0))
    print(np.std(custom_rws, axis=0))
    print("MUJOCO REWARDS")
    # print(mujoco_rws)
    print(np.mean(mujoco_rws, axis=0))
    print(np.std(mujoco_rws, axis=0))

    custom_rws = evaluate_mlp_based(n=n, pname="mjc_ppo_2022_05_05_18_07_46_972885.zip")
    mujoco_rws = evaluate_mujoco_based(n=n, pname="mlp_hist5_ppo_2022_05_05_16_37_13_929111.zip")
    print("MUJOCO on CUSTOM REWARDS")
    # print(custom_rws)
    print(np.mean(custom_rws, axis=0))
    print(np.std(custom_rws, axis=0))
    print("CUSTON on MUJOCO REWARDS")
    # print(mujoco_rws)
    print(np.mean(mujoco_rws, axis=0))
    print(np.std(mujoco_rws, axis=0))


def getexperimentresults(path: str, n: int = 500) -> float:
    """
    :param path: full path to a directory with experiment results
    :param n: take first n timesteps
    :return: sum of rewards
    """
    history = {
               # "pos": [],
               # "orn": [],
               # "ipos": [],
               # "iorn": [],
               # "euler": [],
               # "lin": [],
               # "ang": [],
               # "timestamp": [],
               # "updated": [],
               # "act": [],
               # "servos": [],
               "rewards": [],
               # "auto": [],
               # "acttime": []
               }
    for key in history.keys():
        history[key] = load_raw_data(path=os.path.join(path, key + ".npy"))
    return float(np.sum(history["rewards"][:n]))


def getexperiment_stats(paths: list, n: int = 500) -> (float, float):
    """
    :param paths: list of full paths to directories with experiment results
    :param n: take first n timesteps
    :return: mean sum of rewards, standard deviation
    """
    results = np.asarray([getexperimentresults(path=path, n=n) for path in paths])
    return np.mean(results), np.std(results)


def evaluate_experiments():
    experiments = ["mjc_inf", "mjc_lap", "mjc_rand", "mlp_inf", "mlp_lap", "mlp_rand"]
    for experiment in experiments:
        pathname = os.path.join(Dirs.experiments, "[3|4]", experiment + "*")
        paths = [path for path in glob.glob(pathname=pathname)]
        mean, std = getexperiment_stats(paths=paths, n=1000)
        print(f"experiment: {experiment}; mean: {mean}; std:{std}")


if __name__ == "__main__":
    mujoco_rws = evaluate_mujoco_based(pname="best_model", n=1)
    print(mujoco_rws)
    # mlp_rws = evaluate_mlp_based(n=1)
    # print(mlp_rws)
    # compare_custom2mujoco_based(n=10)
    # history["act"] = np.asarray(history["act"])
    # import matplotlib.pyplot as plt
    # plt.figure()
    # plt.plot(np.arange(len(history["act"])), history["act"][:, 0])
    # # plt.plot(history["pos"][autoindices, 0], history["pos"][autoindices, 1])
    # plt.show()
    # evaluate_experiments()
