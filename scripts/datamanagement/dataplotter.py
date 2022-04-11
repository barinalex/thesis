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


def plotfeature(data: tuple):
    """
    :param data: tuple (title, numpy array shape (n,), x label, y label)
    """
    plt.figure()
    plt.plot(np.arange(data[1].shape[0]), data[1])
    plt.title(label=data[0])
    plt.xlabel(data[2])
    plt.ylabel(data[3])
    plt.show()


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
    plotfeature(data=d)
