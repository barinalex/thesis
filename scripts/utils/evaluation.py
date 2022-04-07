import time

import numpy as np
from sklearn.metrics import mean_squared_error
from scripts.utils.linalg_utils import euler_from_quaternion
import scripts.datamanagement.datamanager as dm
from scripts.engine.enginesfactory import get_env_engine
from scripts.constants import datatypes
from scripts.engine.engines.abstractengine import AbstractEngine
from scripts.simulation.inputwrapper.datainputwrapper import DataWrapper
from scripts.simulation.simulator import Simulator


def orn_mean_sq_error(orn1: np.ndarray, orn2: np.ndarray) -> float:
    """rotation around z axis. angles are from an interval [-pi,pi]"""
    diff = abs(orn1[:, 2] - orn2[:, 2])
    diff = [2 * np.pi - x if x > np.pi else x for x in diff]
    return np.array([x**2 for x in diff]).sum() / len(diff)


def evaluate(gt: dict, eng: dict) -> (float, float):
    """compare engine trajectory to a ground truth and return evaluation score"""
    mse_pos = mean_squared_error(gt[datatypes['pos']], eng[datatypes['pos']])
    gt_orn = np.asarray([euler_from_quaternion(orn) for orn in gt[datatypes['orn']]])
    eng_orn = np.asarray([euler_from_quaternion(orn) for orn in eng[datatypes['orn']]])
    mse_orn = orn_mean_sq_error(gt_orn, eng_orn)
    return mse_pos, mse_orn


def get_n_episodes(datadir: str, n_episodes: int) -> dict:
    """load evaluation dataset and return n random episodes from it"""
    evalset = dm.loadruns(datadir=datadir)
    keys = np.array(list(evalset.keys()))
    indices = np.random.randint(low=0, high=len(keys), size=n_episodes)
    return {key: evalset[key] for key in keys[indices]}


def engine_to_groundtruth(engine: AbstractEngine, datadir: str, n_episodes: int = 10) -> (float, float):
    """compare engine response to input to a real robot and
    return score that is MSE between positions and orientations"""
    evalset = get_n_episodes(datadir=datadir, n_episodes=n_episodes)
    msep = np.zeros(n_episodes)
    mseo = np.zeros(n_episodes)
    i = 0
    for key in evalset.keys():
        dw = DataWrapper(data=evalset[key])
        engine.set_initial_state(observation=dw.get_observation(index=0))
        with Simulator(inputwrapper=dw, viz_on=False, models=[engine]) as sim:
            sim.simulate()
        history = engine.gethistory()
        msep[i], mseo[i] = evaluate(gt=evalset[key], eng=history)
        # print(f"mse: pos: {msep[i]}; orn: {mseo[i]}")
        i += 1
    return np.mean(msep), np.mean(mseo)


def agent_sim_to_real():
    """compare agent episodes with same trajectories
    in simulation and real world"""
    pass


def evalagent(uid1: int):
    """
    run an agents in a simulation on different trajectories
    and return it's mean score
    Args:
        uid - unique id of an agent to evaluate
    """
    pass


if __name__ == "__main__":
    from scripts.datamanagement.pathmanagement import PathManager
    pm = PathManager()
    msep, mseo = engine_to_groundtruth(engine=get_env_engine(eng_type='pbe'),
                                       datadir=f"{pm.paths['rawdata']}/concretemerged_evaluation",
                                       n_episodes=5)
    print("mse pos", msep)
