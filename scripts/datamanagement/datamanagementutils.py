import os.path
import sys
import json
import numpy as np
from scripts.constants import DT, Dirs
from scripts.datamanagement.pathmanagement import create_directories


def readjson(path: str):
    """
    Read data from json file

    :param path: path to a json file
    """
    data = None
    try:
        with open(path, 'r') as f:
            data = json.load(f)
    except:
        pass
    return data


def drop_first(data: dict, n: int) -> dict:
    """
    :return: data without first n samples
    """
    return {key: data[key][n:] for key in data.keys()}


def drop_last(data: dict, n: int) -> dict:
    """
    :return: data without last n samples
    """
    return {key: data[key][:(len(data[key])-n)] for key in data.keys()}


def chop_ends(data: dict, n: int) -> dict:
    """
    :return: data without first and last n samples
    """
    return drop_first(data=drop_last(data=data, n=n), n=n)


def load_raw_data(path: str) -> np.ndarray:
    """
    load data from a file. exit on fail

    :param path: path to a file
    :return: numpy array
    """
    try:
        return np.load(path)
    except Exception as e:
        sys.exit(f"ERROR WHILE LOADING DATA: {e}")


def save_raw_data(data: np.ndarray, path: str):
    """
    save numpy array to a file. exit on fail

    :param data: numpy array
    :param path: where to save
    """
    try:
        np.save(file=path, arr=data)
    except Exception as e:
        sys.exit(f"ERROR WHILE SAVING DATA: {e}")


def reshapeto2d(data: np.ndarray) -> np.ndarray:
    """
    shape should be (n,m) for concatenation, not (n,)

    :param data: numpy array of a shape (n,) or (n,m,...)
    :return: numpy array of a shape (n,1) or (n,m,...)
    """
    return data[:, np.newaxis] if len(data.shape) == 1 else data


def load_data_type(datadir: str, datatype: str) -> np.ndarray:
    """
    load all data of the same type from directory

    :param datadir: directory with data
    :param datatype: type of data to load
    :return: numpy array. exit on fail
    """
    try:
        path = os.path.join(datadir, f"{datatype}.npy")
        return reshapeto2d(load_raw_data(path))
    except Exception as e:
        sys.exit(f"ERROR WHILE LOADING DATA: {e}")


def correct_throttle(throttle: np.ndarray) -> np.ndarray:
    """
    clip throttle and map it to [-1, 1] from [0, 1]

    :param throttle: numpy array of a shape (n,)
    :return: numpy array (n,) with corrected throttle data
    """
    return (np.clip(throttle, 0, 1) - 0.5) * 2


def make_history_of_observations(data: np.ndarray, dim: int, n: int) -> np.ndarray:
    """
    combine several consecutive samples to create a vector
    of n_hist previous observations (history)

    :param data: numpy array with observations
    :param dim: dimensionality of observations
    :param n: number of observations to put into vector
    :return: numpy array with vectors of n observations
    """
    obs = np.zeros((len(data), dim * n))
    for i in range(n):
        obs[i] = np.hstack((np.zeros((n - i - 1) * dim), data[:i + 1].reshape((i + 1) * dim)))
    for i in range(n - 1, len(data) - 1):
        obs[i] = data[i - n + 1:i + 1].reshape(dim * n)
    return obs


def calculate_deltas(data: np.ndarray) -> np.ndarray:
    """
    difference between value at a time t+1 and t (deltas)

    :param data: numpy array
    :return: numpy array with computed deltas
    """
    deltas = np.zeros(data.shape)
    deltas[1:] = data[1:] - data[0:-1]
    return deltas


def calculate_deltas_old(data: np.ndarray, clip_val: float, scale: int) -> np.ndarray:
    """
    clip and scale difference between value at a time t+1 and t (deltas)

    :param data: numpy array
    :param clip_val: interval to clip deltas
    :param scale: multiplier of deltas
    :return: numpy array with computed deltas
    """
    deltas = np.zeros(data.shape)
    deltas[0] = np.clip(data[0] * scale, -clip_val, clip_val)
    deltas[1:] = np.clip((data[1:] - data[0:-1]) * scale, -clip_val, clip_val)
    return np.clip(deltas, -clip_val, clip_val)


def load_dataset(datadir: str, dts: list) -> dict:
    """
    load all types od data from a directory to create a complete dataset

    :param datadir: directory with data
    :param dts: types of data to load
    :return: dictionary without None values for keys
    """
    ds = {dt: load_data_type(datadir=datadir, datatype=dt) for dt in dts}
    keys = list(ds.keys())
    for key in keys:
        ds.pop(key) if ds[key] is None else None
    return ds


def save_dataset(data: dict, path: str):
    """
    save each value of a dict to separate files named as key

    :param data: dict with a dataset
    :param path: path to a directory to save the dataset
    """
    create_directories(path=path)
    for key in data.keys():
        keypath = os.path.join(path, key)
        np.save(keypath, data[key])


def load_episode(path) -> dict:
    """
    :param path: path to a directory with saved episode
    :return: dictionary with all episode observations along with trajectory
    """
    dts = DT.typeslist
    dts.append("trajectory")
    return load_dataset(datadir=path, dts=dts)


def save_trajectory(data: np.ndarray, name: str):
    """
    :param data: numpy array. trajectory points
    :param name: name of the trajectory
    """
    np.save(f"{Dirs.trajectories}/{name}", data)


def load_trajectory(name: str) -> np.ndarray:
    """
    :param name: name of the trajectory
    :return: numpy array. trajectory points
    """
    return load_raw_data(path=f"{Dirs.trajectories}/{name}")


def reshape_batch_first(obs: np.ndarray, labels: np.ndarray, batchsize: int) -> (np.ndarray, np.ndarray):
    """
    make set of batches from flat data

    :param obs: observations. numpy array
    :param labels: labels. numpy array
    :param batchsize: size of a batch
    :return: tuple (observations, labels) with batches as a first dimension of a size batchsize
    """
    n_samples = obs.shape[0] - obs.shape[0] % batchsize
    obs, labels = obs[:n_samples], labels[:n_samples]
    obs = obs.reshape((n_samples // batchsize, batchsize, *obs.shape[1:]))
    labels = labels.reshape((n_samples // batchsize, batchsize, *labels.shape[1:]))
    return obs, labels


def reshape_no_batches(obs, labels):
    """
    stack batches into one dimension

    :param obs: observations. numpy array
    :param labels: labels. numpy array
    :return: tuple (observations, labels) without batches in a first dimension
    """
    obs = obs.reshape(obs.shape[0] * obs.shape[1], *obs.shape[2:])
    labels = labels.reshape(labels.shape[0] * labels.shape[1], *labels.shape[2:])
    return obs, labels


def random_interval(limit: int, size: int) -> (int, int):
    """
    :param limit: max possible index
    :param size: size of an interval
    :return: tuple (first index, last index), where last - first = size
    """
    # first = np.random.randint(low=0, high=limit - size)
    first = limit - size - 1
    return first, first + size


def random_subset(data: dict, size: int) -> (int, dict):
    """
    :param data: dict containing a dataset
    :param size: size of a subset to extract
    :return: tuple (first index of a subset, subset), where subset is a
        random subset of data with a specified size
    """
    start = np.random.randint(low=0, high=(len(data[list(data.keys())[0]]) - size))
    return start, {key: data[key][start:start+size] for key in data.keys()}


def concatenate_dicts(d1: dict, d2: dict) -> dict:
    """
    :param d1: main data
    :param d2: data to add to d1. d2 has same keys as d1
    and values of type np.ndarray with same shape as in d1 if d1 is not empty
    :return: concatenated dicts
    """
    if not d1:
        return d2
    for key in d1.keys():
        d1[key] = np.concatenate((d1[key], d2[key]))
    return d1


if __name__ == "__main__":
    def reducefrequency():
        import glob
        for edir in glob.glob(pathname=os.path.join(Dirs.realdata, "*")):
            data = load_dataset(datadir=edir, dts=DT.bagtypes)
            n = data[list(data.keys())[0]].shape[0]
            indices = np.arange(0, n, 2)
            data = {key: item[indices] for key, item in data.items()}
            save_dataset(data=data, path=edir)

    reducefrequency()

