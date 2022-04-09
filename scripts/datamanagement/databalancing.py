import numpy as np
from scripts.constants import DT


def add_symmetrical_copy(data: np.ndarray, index: int):
    """
    simply negate data and concatenate with original

    :param data: numpy array of a shape (r, c)
    :param index: column index to copy
    :return: data with symmetrical copy
    """
    mirror = np.copy(data)
    mirror[:, index] = -mirror[:, index]
    return np.concatenate((data, mirror))


def balance_left_right_turn(data: dict) -> dict:
    """
    make symmetrical copy of all data relevant to turn of a robot
    and concatenate it with true data. i.e. make all left turns right and vice versa

    :param data: dictionary with data of types vel, ang, jact at least
    :return: dictionary with symmetrical copy of data
    """
    data[DT.lin] = add_symmetrical_copy(data=data[DT.lin], index=1)
    data[DT.ang] = add_symmetrical_copy(data=data[DT.ang], index=2)
    data[DT.act] = add_symmetrical_copy(data=data[DT.act], index=1)
    for key in data.keys():
        if key not in [DT.lin, DT.ang, DT.act]:
            data[key] = np.concatenate((data[key], np.copy(data[key])))
    return data


def dobalance(params: dict, data: dict) -> dict:
    """
    balance data according to a config

    :param params: parameters of data preprocessing
    :param data: dictionary with a full dataset
    :return: dictionary with balanced dataset
    """
    if 'turn' in params['balance']:
        data = balance_left_right_turn(data)
    return data


def get_subset_indices(data: np.ndarray, a: float, b: float) -> np.ndarray:
    """
    :param data: shape (n, )
    :param a: low condition
    :param b: high condition
    :return: indices of a subset of data where condition is satisfied
    """
    return ((data >= a) & (data < b)).nonzero()[0]


def get_symmetrical_balance_indices(obs: np.ndarray, dim: int, nbins: int = 20) -> np.ndarray:
    """
    :param obs: observations. shape (n, dims)
    :param dim: dimension to balance
    :param nbins: number of bins for a histogram according to which data are balanced
    :return: indices of observations to be added to balance data
    """
    tobalance = obs[:, dim]
    nbins = nbins if nbins % 2 == 0 else nbins - 1
    bound = np.max((np.max(tobalance), -np.min(tobalance)))
    hist, edges = np.histogram(tobalance, bins=nbins, range=(-bound, bound))
    indices = np.empty(0, dtype=int)
    for i in range(nbins // 2):
        index = i if hist[i] < hist[nbins - 1 - i] else nbins - 1 - i
        subindices = get_subset_indices(data=tobalance, a=edges[index], b=edges[index + 1])
        amount = abs(hist[i] - hist[nbins - 1 - i])
        if amount > 0 and hist[index] > 0:
            samples = np.random.choice(np.arange(len(subindices)), amount, replace=True)
            indices = np.concatenate((indices, subindices[samples]))
    return indices


def balance_data(obs: np.ndarray, labels: np.ndarray, dim: int, sobs: np.ndarray = None) -> (np.ndarray, np.ndarray):
    """
    :param obs: observations. shape (n, dims)
    :param labels: labels. shape (n, ?)
    :param dim: dimension to balance
    :param sobs: sequential observations. shape (n, sequencelength, dims)
    :return: balanced (observations. sequential if specified, labels)
    """
    indices = get_symmetrical_balance_indices(obs=obs, dim=dim)
    if sobs is not None:
        return np.concatenate((sobs, sobs[indices]), axis=0), np.concatenate((labels, labels[indices]))
    return np.concatenate((obs, obs[indices]), axis=0), np.concatenate((labels, labels[indices]))
