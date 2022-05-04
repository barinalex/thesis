import os.path
import numpy as np
from scripts.simulation.datainputwrapper import DataWrapper
from scripts.simulation.simulator import Simulator
from scripts.engine.engine import Engine
from sklearn.metrics import mean_squared_error
from scripts.datamanagement.datamanagementutils import load_raw_data
import quaternion
from scripts.utils.linalg_utils import angle2x


def trajectory2xaxis(trajectory: np.ndarray) -> np.ndarray:
    """
    :param trajectory: ground truth trajectory. shape (n, 2)
    :return: rotated trajectory
    """
    rotation = angle2x(vector=trajectory[1] - trajectory[0])
    q = quaternion.from_euler_angles(-rotation/2, 0, -rotation/2)
    rm = quaternion.as_rotation_matrix(q)
    rotated = (rm @ trajectory.T).T
    return rotated[:, :2]


def evalsection(positions: np.ndarray, actions: np.ndarray, engine: Engine) -> (float, np.ndarray):
    """
    :param positions: ground truth positions. shape (n, 2)
    :param actions: actions taken. shape (n, 2)
    :param engine: engine to evaluate
    :return: (mean squared error between ground truth and simulated trajectories,
        simulated trajectory)
    """
    inputwrapper = DataWrapper(actions=actions)
    sim = Simulator(iw=inputwrapper, engine=engine)
    simpositions = sim.simulate()
    return mean_squared_error(positions, simpositions[:, :2]), simpositions[:, :2]


def evalloop(path: str, engine: Engine, m: int, length: int) -> (np.ndarray, np.ndarray):
    """
    :param path: path to a file with full episode ground truth data
    :param engine: engine to evaluate
    :param m: number of sections to sample for evaluation
    :param length: lenght of sampled sections
    :return: (mean squared errors. shape (m,),
        ground truth trajectories. shape (m, length, 2)),
        simulated trajectories. shape (m, length, 2))
    """
    positions = -load_raw_data(path=os.path.join(path, "positions.npy"))
    actions = load_raw_data(path=os.path.join(path, "actions.npy"))
    linear = load_raw_data(path=os.path.join(path, "linear.npy"))
    angular = load_raw_data(path=os.path.join(path, "angular.npy"))
    N = positions.shape[0]
    mse = np.zeros(m)
    gtpositions = np.zeros((m, length-1, 2))
    simpositions = np.zeros((m, length-1, 2))
    for i in range(m):
        start = np.random.randint(N - length)
        indices = np.arange(start, start + length)
        engine.setstate(vel=linear[start], ang=angular[start])
        gtpositions[i] = trajectory2xaxis(positions[indices][:-1] - positions[start])
        mse[i], simpositions[i] = evalsection(positions=gtpositions[i],
                                              actions=actions[indices],
                                              engine=engine)
        engine.reset()
    return mse, gtpositions, simpositions


if __name__ == "__main__":
    from scripts.constants import Dirs
    from scripts.engine.mlpbased import MLPBased
    import matplotlib.pyplot as plt
    episodes = ["2022_05_01_11_51_35_858887"]
    path = os.path.join(Dirs.realdata, episodes[0])

    epath = os.path.join(Dirs.models, "mlp_2022_05_01_12_30_00_981419")
    engine = MLPBased(path=epath)
    m = 5
    mse, gtpos, simpos = evalloop(path=path, engine=engine, m=m, length=200)
    for i in range(m):
        print(mse[i])
        plt.figure()
        plt.plot(gtpos[i, :, 0], gtpos[i, :, 1], color="b")
        plt.plot(simpos[i, :, 0], simpos[i, :, 1], color="r")
        plt.show()

    # positions = -load_raw_data(path=os.path.join(path, "positions.npy"))
    # length = 500
    # n = positions.shape[0]
    # start = np.random.randint(n - length)
    # indices = np.arange(start, start + length)
    # positions = positions[indices] - positions[start]
    # rotated = trajectory2xaxis(positions)
    # plt.figure()
    # plt.plot(positions[:, 0], positions[:, 1], color="b")
    # plt.plot(rotated[:, 0], rotated[:, 1], color="r")
    # plt.show()

