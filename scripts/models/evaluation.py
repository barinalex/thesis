import os.path
import numpy as np
from scripts.simulation.datainputwrapper import DataWrapper
from scripts.simulation.simulator import Simulator
from scripts.engine.engine import Engine
from scripts.engine.mlpbased import MLPBased
from scripts.engine.tcnnbased import TCNNBased
from sklearn.metrics import mean_squared_error
from scripts.constants import Dirs, DT
import matplotlib.pyplot as plt
from scripts.datamanagement.datamanagementutils import load_raw_data


def computeorientations(pos) -> np.ndarray:
    """
    :param pos: positions. shape (n, 2)
    :return: angles to x axis
    """
    pos = pos[1:] - pos[:-1]
    angles = np.arctan2(pos[:, 1], pos[:, 0])
    return angles


def custommse(gt, sim) -> float:
    """
    :param gt: ground truth positions. shape (n, 2)
    :param sim: simulated positions. shape (n, 2)
    """
    poserror = np.linalg.norm(gt - sim)
    gtorn = computeorientations(gt)
    simorn = computeorientations(sim)
    ornerror = (gtorn - simorn) ** 2
    error = np.multiply(poserror, ornerror)
    return error.mean()


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
    return mean_squared_error(positions[:, :2], simpositions[:, :2]), simpositions[:, :2]
    # return custommse(positions[:, :2], simpositions[:, :2]), simpositions[:, :2]


def evalloop(sections, engine: Engine, msections: int) -> (np.ndarray, np.ndarray):
    """
    :param sections: validation sections from the real episode
    :param engine: engine to evaluate
    :param msections: take at most first msections for evaluation
    :return: (mean squared errors. shape (m,),
        simulated trajectories. shape (m, length, 2))
    """
    m = min(sections[DT.pos].shape[0], msections)
    length = sections[DT.pos].shape[1]
    mse = np.zeros(m)
    simpositions = np.zeros((m, length-1, 2))
    for i in range(m):
        engine.setstate(vel=sections[DT.lin][i][0], ang=sections[DT.ang][i][0])
        obs = np.hstack((sections[DT.lin][i][0][:2],
                         sections[DT.ang][i][0][2],
                         sections[DT.act][i][0]))
        for _ in range(engine.buffer.size):
            engine.buffer.add(obs)
        mse[i], simpositions[i] = evalsection(positions=sections[DT.pos][i][: -1],
                                              actions=sections[DT.act][i],
                                              engine=engine)
        engine.reset()
    return mse, simpositions


def compareengines():
    mlppath = os.path.join(Dirs.models, "mlp_2022_05_01_12_30_00_981419")
    histpath = os.path.join(Dirs.models, "mlp_hist5_2022_05_05_11_23_43_430257")
    tcnnpath = os.path.join(Dirs.models, "tcnn_2022_05_05_13_08_24_709391")
    mlp = (MLPBased(path=mlppath), "MLP")
    hist = (MLPBased(path=histpath), "History MLP")
    tcnn = (TCNNBased(path=tcnnpath), "TCNN")
    engines = [mlp, hist, tcnn]

    m = 5
    sections = {}
    for key in DT.bagtypes:
        sections[key] = load_raw_data(path=os.path.join(Dirs.valid, key + ".npy"))

    figure, axis = plt.subplots(3, m)
    for i, engine in enumerate(engines):
        mse, simpos = evalloop(sections=sections, engine=engine[0], msections=m)
        print(f"{engine[1]} MSE:", np.mean(mse))
        axis[i][0].set_ylabel(engine[1])
        for j in range(m):
            # axis[i][j].set_xlabel("meters")
            axis[i][j].plot(sections[DT.pos][j, :, 0], sections[DT.pos][j, :, 1], color="b")
            axis[i][j].plot(simpos[j, :, 0], simpos[j, :, 1], color="r")
    plt.show()


if __name__ == "__main__":
    compareengines()
    exit()

    sections = {}
    for key in DT.bagtypes:
        sections[key] = load_raw_data(path=os.path.join(Dirs.valid, key + ".npy"))

    epath = os.path.join(Dirs.models, "mlp_2022_05_01_12_30_00_981419")
    engine = MLPBased(path=epath)

    m = 10
    mse, simpos = evalloop(sections=sections, engine=engine, msections=m)

    print(np.mean(mse))
    figure, axis = plt.subplots(1, m)
    for i in range(m):
        print(mse[i])
        axis[i].set_xlabel("meters")
        axis[0].set_ylabel("meters")
        axis[i].plot(sections[DT.pos][i, :, 0], sections[DT.pos][i, :, 1], color="b")
        axis[i].plot(simpos[i, :, 0], simpos[i, :, 1], color="r")
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

