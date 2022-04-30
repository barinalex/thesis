import numpy as np
import itertools
import matplotlib.pyplot as plt
from scripts.datamanagement.datamanagement import loadconfig, get_data
from scripts.datamanagement.datamanagementutils import load_raw_data, reshape_no_batches, readjson
from scripts.constants import Dirs, DT
import os


def plotfeatures(data: list, x: int = 2, y: int = 2):
    """
    :param data: list with features to plot. list of tuples (title, numpy array, x label, y label)
        all features are supposed to have same shape (n,)
    :param x: number of plots along x axis
    :param y: number of plots along y axis
    """
    figure, axis = plt.subplots(y, x)
    indices = list(itertools.product(np.arange(np.max((x, y))), repeat=2))
    indices = [i for i in indices if i[0] < y and i[1] < x]
    n = len(data)
    for i, t in zip(indices[:n], data):
        axis[i[0], i[1]].plot(np.arange(t[1].shape[0]), t[1])
        axis[i[0], i[1]].set_title(t[0])
        axis[i[0], i[1]].set_xlabel(t[2])
        axis[i[0], i[1]].set_ylabel(t[3])
    plt.show()


def plot2d(data: tuple):
    """
    :param data: tuple (title, numpy array shape (n,), x label, y label)
    """
    plt.figure()
    plt.plot(np.arange(data[1].shape[0]), data[1])
    plt.title(label=data[0])
    plt.xlabel(data[2])
    plt.ylabel(data[3])
    plt.show()


def makehistogram(fig, data: np.ndarray):
    """
    :param fig: figure to place histogram on
    :param data: array of floats, shape (n, )
    """
    hist, edges = np.histogram(data)
    widths = abs(edges[1:] - edges[:-1])
    fig.bar(edges[:len(hist)], hist, width=np.min(widths), color='b', alpha=1, align='edge')


def plothistograms(data: np.ndarray):
    """
    :param data: array of floats, shape (n, 2)
    """
    figure, axis = plt.subplots(1, 2)
    # makehistogram(axis[0], data[:, 0])
    # axis[0].set_title("Delta linear velocity along X axis")
    # axis[0].set_xlabel("meters per second")
    # axis[0].set_ylabel("samples")
    # makehistogram(axis[1], data[:, 1])
    # axis[1].set_title("Delta linear velocity along Y axis")
    # axis[1].set_xlabel("meters per second")
    # axis[1].set_ylabel("samples")
    # makehistogram(axis[0], data[:, 2])
    # axis[0].set_title("Delta angular velocity")
    # axis[0].set_xlabel("radians per second")
    # axis[0].set_ylabel("samples")
    makehistogram(axis[0], data[:, 3])
    axis[0].set_title("Throttle")
    axis[0].set_xlabel("value")
    axis[0].set_ylabel("samples")
    makehistogram(axis[1], data[:, 4])
    axis[1].set_title("Turn")
    axis[1].set_xlabel("value")
    axis[1].set_ylabel("samples")
    plt.show()


def plotevals():

    path = os.path.join(Dirs.models, "mlp_2022_04_24_19_56_37_855601.evals" + ".npy")
    mlpevals = load_raw_data(path=path)
    path = os.path.join(Dirs.models, "mlp_2022_04_24_20_04_23_633569.evals" + ".npy")
    hmlpevals = load_raw_data(path=path)
    path = os.path.join(Dirs.models, "tcnn_2022_04_24_19_54_44_249977.evals" + ".npy")
    cnnevals = load_raw_data(path=path)

    epochs = np.arange(mlpevals.shape[0])

    figure, axis = plt.subplots(1, 3)
    axis[0].set_title("MLP")
    axis[0].set_xlabel("epochs")
    axis[0].set_ylabel("loss")
    axis[0].plot(epochs, mlpevals[:, 0], color='b')
    axis[0].plot(epochs, mlpevals[:, 1], color='r')
    axis[0].legend(['train loss', 'test loss'])

    epochs = np.arange(cnnevals.shape[0])
    axis[1].set_title("TCNN")
    axis[1].set_xlabel("epochs")
    # axis[1].set_ylabel("loss")
    axis[1].plot(epochs, cnnevals[:, 0], color='b')
    axis[1].plot(epochs, cnnevals[:, 1], color='r')
    axis[1].legend(['train loss', 'test loss'])

    epochs = np.arange(hmlpevals.shape[0])

    axis[2].set_title("MLP with history")
    axis[2].set_xlabel("epochs")
    # axis[2].set_ylabel("loss")
    axis[2].plot(epochs, hmlpevals[:, 0], color='b')
    axis[2].plot(epochs, hmlpevals[:, 1], color='r')
    axis[2].legend(['train loss', 'test loss'])
    plt.show()


def plottrainingdata():
    # path = os.path.join(Dirs.realdata, "2022_04_30_14_34_06_977957")
    # positions = load_raw_data(path=f"{path}/positions.npy")
    # actions = load_raw_data(path=f"{path}/actions.npy")
    # linear = load_raw_data(path=f"{path}/linear.npy")
    # angular = load_raw_data(path=f"{path}/angular.npy")
    # n = positions.shape[0]
    # indices = np.arange(0, n, 2)
    # positions = positions[indices]
    # actions = actions[indices]
    # linear = linear[indices]
    # angular = angular[indices]
    #
    # from scripts.datamanagement.datafilters import applyfilter
    # from scripts.datamanagement.datamanagement import make_labels
    # from scripts.datamanagement.datamanagementutils import reshapeto2d

    path = os.path.join(Dirs.configs, "cnn.yaml")
    config = loadconfig(path=path)
    k=100000

    # data = [("Raw linear velocity along X axis (forward velocity)", linear[:k, 0], "time step", "meters per second"),
    #         ("Raw linear velocity along Y axis", linear[:k, 1], "time step", "meters per second"),
    #         ("Raw angular velocity around Z axis", angular[:k, 2], "time step", "meters per second"),
    #         ("Action: throttle", actions[:k, 0], "time step", ""),
    #         ("Action: turn", actions[:k, 1], "time step", "")]

    # config["test_size"] = 0
    train, test, ncnts = get_data(params=config)
    obs, labels = reshape_no_batches(train[DT.obs], train[DT.labels])
    print(obs.shape)

    data = [("Raw linear velocity along X axis (forward velocity)", obs[:k, 0], "time step", "meters per second"),
            ("Raw linear velocity along Y axis", obs[:k, 1], "time step", "meters per second"),
            ("Raw angular velocity around Z axis", obs[:k, 2], "time step", "meters per second"),
            ("Action: throttle", obs[:k, 3], "time step", ""),
            ("Action: turn", obs[:k, 4], "time step", "")]
    #
    # data = [("Action throttle", obs[:, 3], "time step", ""),
    #         ("Action turn", obs[:, 4], "time step", "")]
    #
    # data = [("Delta linear velocity along X axis", labels[100:k*2, 0], "time step", "meters per second"),
    #         ("Delta linear velocity along Y axis", labels[100:k*2, 1], "time step", "meters per second"),
    #         ("Delta angular velocity", labels[100:k*2, 2], "time step", "meters per second")]

    # data = [("Filtered linear velocity along X axis (forward velocity)", obs[:, 0], "time step", "meters per second"),
    #         ("Filtered linear velocity along Y axis", obs[:, 1], "time step", "meters per second"),
    #         ("Filtered angular velocity around Z axis", obs[:, 2], "time step", "meters per second")]

    figure, axis = plt.subplots(1, len(data))
    for i, d in enumerate(data):
        axis[i].set_title(d[0])
        axis[i].set_xlabel(d[2])
        axis[0].set_ylabel(d[3])
        axis[i].plot(np.arange(d[1].shape[0]), d[1])
    plt.show()
    # for d in data:
    #     plot2d(data=d)


def plotobshistogram():
    path = os.path.join(Dirs.configs, "mlp.yaml")
    config = loadconfig(path=path)
    config["test_size"] = 0
    train, test, ncnts = get_data(params=config)
    obs, labels = reshape_no_batches(train[DT.obs], train[DT.labels])
    # plt.figure()
    # makehistogram(plt, obs[:, 0])
    # plt.show()
    plothistograms(data=obs)
    plt.show()


def plot_policy_learning_curve(maxtimesteps: int = None):
    """load evaluation callback results and plot as a learning curve"""

    path = os.path.join(Dirs.policy, "ppo_tcnn_2022_04_30_14_38_23_442446" + ".npz")
    mj = load_raw_data(path=path)
    path = os.path.join(Dirs.policy, "ppo_tcnn_2022_04_30_14_38_23_442446" + ".npz")
    tcnn = load_raw_data(path=path)

    means = np.mean(mj['results'], axis=1)
    time = np.arange(maxtimesteps, step=maxtimesteps//means.shape[0]) if maxtimesteps else np.arange(means.shape[0])

    figure, axis = plt.subplots(1, 2)
    axis[0].set_title("MuJoCo")
    axis[0].set_xlabel("timesteps")
    axis[0].set_ylabel("reward")
    axis[0].plot(time[:means.shape[0]], means)

    means = np.mean(tcnn['results'], axis=1)
    time = np.arange(maxtimesteps, step=maxtimesteps//means.shape[0]) if maxtimesteps else np.arange(means.shape[0])

    axis[1].set_title("TCNN")
    axis[1].set_xlabel("timesteps")
    axis[1].set_ylabel("reward")
    axis[1].plot(time[:means.shape[0]], means)
    plt.show()


def plotexperiment():
    path = os.path.join(Dirs.experiments, "2022_04_30_12_45_43_220823")
    data = readjson(path=path)
    positions = [entry["pos"] for entry in data]
    positions = np.asarray(positions)
    plt.figure()
    plt.plot(positions[:0], positions[:1])
    plt.show()



if __name__ == "__main__":
    # plotobshistogram()
    plottrainingdata()
    # exit()
    # plotevals()
    # path = os.path.join(Dirs.policy, "ppo_tcnn_2022_04_18_17_42_46_675414.npz")
    # plot_policy_learning_curve(maxtimesteps=1000000)
    # plotexperiment()
    exit()
    # pass


    # points = load_raw_data(os.path.join(Dirs.trajectories, "n1000_wps500_smth50_mplr10.npy"))
    # points = points[:, :70, :]
    #
    # figure, axis = plt.subplots(1, 2)
    # axis[0].set_xlabel("X meters")
    # axis[0].set_ylabel("Y meters")
    # for traj in points[:5,:30]:
    #     axis[0].scatter(traj[:, 0], traj[:, 1])
    #
    # axis[1].set_xlabel("X meters")
    # axis[1].set_ylabel("Y meters")
    # for traj in points[5:]:
    #     axis[1].plot(traj[:, 0], traj[:, 1])
    # plt.show()

    lap = load_raw_data(os.path.join(Dirs.trajectories, "lap_pd01_r1_s2.npy"))
    inf = load_raw_data(os.path.join(Dirs.trajectories, "inf_pd01_r1.npy"))
    figure, axis = plt.subplots(1, 2)
    axis[0].set_xlabel("X meters")
    axis[0].set_ylabel("Y meters")
    axis[0].plot(lap[0, :, 0], lap[0, :, 1])
    axis[1].set_xlabel("X meters")
    axis[1].set_ylabel("Y meters")
    axis[1].plot(inf[0, :, 0], inf[0, :, 1])
    plt.show()
