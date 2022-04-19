import numpy as np
import itertools
import matplotlib.pyplot as plt
from scripts.datamanagement.datamanagement import loadconfig, get_data
from scripts.datamanagement.datamanagementutils import load_raw_data, reshape_no_batches
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
    path = os.path.join(Dirs.models, "mlp_2022_04_12_18_06_34_082909.evals" + ".npy")
    evals = load_raw_data(path=path)
    plt.figure()
    epochs = np.arange(evals.shape[0])
    plt.plot(epochs, evals[:, 0], color='b')
    plt.plot(epochs, evals[:, 1], color='r')
    plt.legend(['train loss', 'test loss'])
    plt.show()


def plottrainingdata():
    # path = os.path.join(Dirs.simdata, "2022_04_11_20_42_08_312119")
    # positions = load_raw_data(path=f"{path}/positions.npy")
    # actions = load_raw_data(path=f"{path}/actions.npy")
    # linear = load_raw_data(path=f"{path}/linear.npy")
    # angular = load_raw_data(path=f"{path}/angular.npy")
    #
    # plt.figure()
    # plt.plot(positions[:, 0], positions[:, 1])
    # plt.show()


    # data = [("Raw linear velocity along X axis (forward velocity)", linear[:, 0], "time step", "meters per second"),
    #         ("Raw linear velocity along Y axis", linear[:, 1], "time step", "meters per second"),
    #         ("Raw angular velocity around Z axis", angular[:, 2], "time step", "meters per second"),
    #         ("Action: throttle", actions[:, 0], "time step", ""),
    #         ("Action: turn", actions[:, 1], "time step", "")]


    path = os.path.join(Dirs.configs, "mlp.yaml")
    config = loadconfig(path=path)
    # config["test_size"] = 0
    train, test, ncnts = get_data(params=config)
    obs, labels = reshape_no_batches(train[DT.obs], train[DT.labels])
    print(obs.shape)
    #
    # data = [("Action throttle", obs[:, 3], "time step", ""),
    #         ("Action turn", obs[:, 4], "time step", "")]

    # data = [("Delta linear velocity along X axis", labels[:, 0], "time step", "meters per second"),
    #         ("Delta linear velocity along Y axis", labels[:, 1], "time step", "meters per second"),
    #         ("Delta angular velocity", labels[:, 2], "time step", "meters per second")]

    data = [("Filtered linear velocity along X axis (forward velocity)", obs[:, 0], "time step", "meters per second"),
            ("Filtered linear velocity along Y axis", obs[:, 1], "time step", "meters per second"),
            ("Filtered angular velocity around Z axis", obs[:, 2], "time step", "meters per second")]

    for d in data:
        plot2d(data=d)


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


def plot_policy_learning_curve(path: str, maxtimesteps: int = None):
    """load evaluation callback results and plot as a learning curve"""
    ev = load_raw_data(path)
    means = np.mean(ev['results'], axis=1)
    time = np.arange(maxtimesteps, step=maxtimesteps//means.shape[0]) if maxtimesteps else np.arange(means.shape[0])
    plt.figure()
    plt.xlabel("timesteps")
    plt.ylabel("reward")
    plt.plot(time[:means.shape[0]], means)
    plt.show()


if __name__ == "__main__":
    # plotobshistogram()
    # plottrainingdata()
    # plotevals()
    # path = os.path.join(Dirs.policy, "ppo_tcnn_2022_04_18_17_42_46_675414.npz")
    # plot_policy_learning_curve(path=path, maxtimesteps=500000)
    # pass
    points = load_raw_data(os.path.join(Dirs.trajectories, "n10_wps500_smth50_mplr10.npy"))
    points = points[:, :150, :]
    print(points.shape)
    plt.figure()
    plt.xlabel("X meters")
    plt.ylabel("Y meters")
    for traj in points:
        plt.plot(traj[:, 0], traj[:, 1])
    plt.show()
