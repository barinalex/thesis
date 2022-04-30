import os.path

import numpy as np
import torch
import yaml
import glob
from typing import List
from scripts.constants import DT, Dirs
from scripts.datamanagement.datamanagementutils import *
from scripts.datamanagement.databalancing import *
from scripts.datamanagement.datafilters import *


def loadconfig(path: str) -> dict:
    """
    :param path: path to a configuration file
    :return: configurations as a dictionary
    """
    with open(path) as f:
        try:
            return yaml.load(stream=f, Loader=yaml.FullLoader)
        except IOError as e:
            sys.exit(f"FAILED TO LOAD CONFIG {path}: {e}")


def saveconfig(path: str, config: dict):
    """
    :param path: path to store configuration file
    :param config: configuration file
    """
    with open(path, "w") as f:
        try:
            yaml.dump(config, f)
        except IOError as e:
            sys.exit(f"FAILED TO SAVE CONFIG {path}: {e}")


def preprocess_raw_data(params: dict, data: dict) -> dict:
    """
    do filtering, balancing and everything that might help to train NN

    :param params: parameters of data preprocessing
    :param data: dictionary with a full dataset
    :return: dictionary with preprocessed dataset
    """
    data[DT.lin] = applyfilter(params=params, data=data[DT.lin])
    data[DT.ang] = applyfilter(params=params, data=data[DT.ang])
    return dobalance(params=params, data=data)


def make_labels(lin: np.ndarray, ang: np.ndarray) -> np.ndarray:
    """
    difference between velocity/angular at time t+1 and t: v(t+1) - v(t)

    :param lin: velocity data
    :param ang: angular velocity data
    :return: labels for observations
    """
    lin_delta = calculate_deltas(lin)
    ang_delta = calculate_deltas(ang)
    return np.concatenate((lin_delta, ang_delta), axis=1)


def make_observations(data: dict) -> np.ndarray:
    """
    correct raw data,
    create set of input vectors [lin x, y, angular, throttle, turn]

    :param data: dictionary with a full dataset
    :return: numpy array of observations. concatenation of input arrays
    """
    lin = reshapeto2d(data[DT.lin][:, 0:2])
    ang = reshapeto2d(data[DT.ang][:, 2])
    act = data[DT.act]
    act[:, 0] = correct_throttle(act[:, 0])
    return np.concatenate((lin, ang, act), axis=1)


def label_observations(obs: np.ndarray) -> np.ndarray:
    """
    movements in 2 dimensions
    observations: velocity x y, angular velocity z, throttle, turn
    labels: velocity deltas, angular velocity deltas

    :param obs: observations. numpy array (n, 5) [vel x, y, angular, throttle, turn]
    :return: labels numpy array (n, 3)
    """
    vel = reshapeto2d(obs[:, 0:2])
    ang = reshapeto2d(obs[:, 2])
    return make_labels(lin=vel, ang=ang)


def make_sequential(obs: np.ndarray, length: int) -> (np.ndarray, np.ndarray):
    """
    :param obs: observations. numpy array (n, ...)
    :param length: sequence length
    :return: array of observation sequences. numpy array (n, length, ...)
    """
    sobs = np.zeros((obs.shape[0], length, *obs.shape[1:]))
    for i in range(length):
        sobs[i][length - (i + 1): length] = obs[: i + 1]
    for i in range(length, len(obs)):
        sobs[i] = obs[i - length + 1: i + 1]
    return np.reshape(sobs, (obs.shape[0], length * obs.shape[1]))


def preprocess_observations(params: dict, obs: np.ndarray) -> np.ndarray:
    """
    do filtering, balancing and everything that might help to train NN

    :param params: parameters of data preprocessing
    :param obs: observations. numpy array (n, 5), where second dim
    contains: vel x, vel y, ang vel, throttle, turn
    :return: filtered observations. numpy array (n, 5)
    """
    # TODO filter different types of data with different params
    obs[:, :3] = applyfilter(params=params, data=obs[:, :3])
    return obs


def get_labeled_obs(data: dict, params: dict) -> (np.ndarray, np.ndarray):
    """
    :param data: dictionary with a full dataset
    :param params: parameters of data preprocessing
    :return: tuple of numpy arrays (observations, labels)
    """
    obs = make_observations(data=data)
    obs = preprocess_observations(params=params, obs=obs)
    labels = label_observations(obs=obs)
    if params["sequence_length"] > 1:
        obs = make_sequential(obs=obs, length=params["sequence_length"])
    return obs[:-1], labels[1:]


def get_test_subset(obs: np.ndarray, labels: np.ndarray, interval: (int, int)) -> dict:
    """
    :param obs: observations. numpy array (n, ...)
    :param labels: labels. numpy array (n, ...)
    :param interval: (first, last) indices of an interval
        to be extracted as test data
    :return: test data as a dict: {obs, labels}
    """
    return {DT.obs: obs[interval[0]: interval[1]].copy(),
            DT.labels: labels[interval[0]: interval[1]].copy()}


def get_train_subset(obs: np.ndarray, labels: np.ndarray, interval: (int, int)) -> dict:
    """
    :param obs: observations. numpy array (n, ...)
    :param labels: labels. numpy array (n, ...)
    :param interval: (first, last) indices of a test subset interval
    :return: train data as a dict: {obs, labels}
    """
    return {DT.obs: np.concatenate((obs[:interval[0]], obs[interval[1]:])),
            DT.labels: np.concatenate((labels[:interval[0]], labels[interval[1]:]))}


def get_gathering_data_episode(params: dict, edir: str) -> (dict, dict):
    """
    load data from specified directory,
    split dataset into training and testing subset,
    preprocess training subset

    :param params: parameters of data preprocessing
    :param edir: directory of the episode
    :return: tuple of dicts (train data, test data)
    """
    data = load_dataset(datadir=edir, dts=DT.traintesttypes)
    obs, labels = get_labeled_obs(data=data, params=params)
    interval = random_interval(limit=len(obs), size=int(len(obs) * params["test_size"]))
    testset = get_test_subset(obs=obs, labels=labels, interval=interval)
    trainset = get_train_subset(obs=obs, labels=labels, interval=interval)
    return trainset, testset


def get_data(params: dict) -> (dict, dict, list):
    """
    combine manually gathered data divided into episodes
    into one complete dataset with train and test subsets

    :param params: parameters of data preprocessing
    :return: tuple of dicts (train data, test data, train set normalization constants)
    """
    train, test = {}, {}
    for edir in glob.glob(pathname=os.path.join(Dirs.realdata, "*")):
        tr, ts = get_gathering_data_episode(params=params, edir=edir)
        train = concatenate_dicts(d1=train, d2=tr)
        test = concatenate_dicts(d1=test, d2=ts)
    train[DT.labels], test[DT.labels], normconst = normalizesets(train=train[DT.labels],
                                                                 test=test[DT.labels])
    train[DT.obs], train[DT.labels] = reshape_batch_first(obs=train[DT.obs],
                                                          labels=train[DT.labels],
                                                          batchsize=params["batchsize"])
    return train, test, normconst


def normalize(data: np.ndarray) -> (np.ndarray, float):
    """
    :param data: numpy array (n,)
    :return: array with all values divided by standard deviation; standard deviation
    """
    std = np.std(data)
    return data / std, std


def normalizesets(train: np.ndarray, test: np.ndarray) -> (np.ndarray, np.ndarray, list):
    """
    :param train: train data to normilize
    :param test: test data to normilize
    :return: normilized train data, normilized test data, normalization constants
    """
    normcnst = np.zeros(train.shape[1])
    for i in range(train.shape[1]):
        train[:, i], normcnst[i] = normalize(data=train[:, i])
        test[:, i] /= normcnst[i]
    return train, test, normcnst.tolist()
