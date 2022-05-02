import math as m
import numpy as np
import datetime
import json
import yaml
import sys
import os


def save2json(path: str, data):
    """
    save data to a json file

    :param path: path to a file
    :param data: data to store
    """
    try:
        with open(path, 'w') as f:
            json.dump(data, f)
    except Exception as e:
        pass


def create_directories(path: str):
    """create all directories in the path
    that do not exist and return path"""
    if not os.path.exists(path):
        os.makedirs(path)
    return path


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


def quat2euler(w, x, y, z):
    """
    Convert quaternion (w, x, y, z) to (roll, pitch, yaw) Euler angles

    :return: roll pitch yaw
    """
    pitch = -m.asin(2.0 * (x * z - w * y))
    roll = m.atan2(2.0 * (w * x + y * z), w * w - x * x - y * y + z * z)
    yaw = m.atan2(2.0 * (w * z + x * y), w * w + x * x - y * y - z * z)
    return roll, pitch, yaw


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


def replace_symbols(var: str, to_replace: str, replacement: str):
    """replace all required symbols in the string with desired"""
    for symbol in to_replace:
        var = var.replace(symbol, replacement)
    return var


def gettimestamp():
    """date and time to a string"""
    s = str(datetime.datetime.today())
    return replace_symbols(s, to_replace=' -:.', replacement='_')


if __name__ == "__main__":
    import os.path
    # data = [{"one": 1}, {"one": 2}]
    path = os.path.join("data", "test")
    # save2json(path=path, data=data)
    with open(path, 'r') as f:
        data = json.load(f)
    print(data[0])
