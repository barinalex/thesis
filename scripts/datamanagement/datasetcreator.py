import numpy as np
import glob
from scripts.datamanagement.datamanagementutils import *


def loadruns(datadir: str, drop_first_n: int = 0) -> dict:
    """
    load all data types from all sub directories and return as a dictionary

    :param datadir: main directory
    :param drop_first_n: number of samples to drop from the beginning
    :return: dict with keys as names of runs and values as all observations
    """
    runs = {}
    for rawdir in glob.glob(pathname=f"{datadir}/*"):
        key = rawdir.split('/')[-1]
        runs[key] = load_dataset(datadir=rawdir)
        runs[key] = drop_first(data=runs[key], n=drop_first_n)
    return runs


def create_evaluation_set(datadir: str, savedir: str, subsets_per_run: int = 1, subset_size: int = 500):
    """load raw runs, divide them into trajectories
    and save as a dataset for evaluation purposes"""
    runs = loadruns(datadir=datadir)
    runs = {key: chop_ends(data=runs[key], n=100) for key in runs.keys()}
    for key in runs.keys():
        for _ in range(subsets_per_run):
            start, subset = random_subset(runs[key], size=subset_size)
            save_dataset(data=subset, path=f"{savedir}/{key}_{start}")


def create_train_leave_one_out_set(runs: dict, exclude: str, savedir: str):
    """save all runs for training to train directory"""
    for key in runs.keys():
        if key != exclude:
            save_dataset(data=runs[key], path=f"{savedir}/train/{key}")


def create_test_leave_one_out_set(runs: dict, include: str, savedir: str):
    """save test data to test directory"""
    save_dataset(data=runs[include], path=f"{savedir}/test/{include}")


def create_leave_one_out_train_test_datasets(datadir: str, savedir: str):
    """load raw data, remove first 2 seconds, clip values, save to train/test directories"""
    datasetbasename = "leave_one_out"
    runs = loadruns(datadir=datadir, drop_first_n=200)
    for key in runs.keys():
        datasetdir = f"{savedir}/{datasetbasename}_{key}"
        create_train_leave_one_out_set(runs=runs, exclude=key, savedir=datasetdir)
        create_test_leave_one_out_set(runs=runs, include=key, savedir=datasetdir)


def create_full_dataset(datadir: str, savedir: str):
    """load raw data, remove first 2 seconds, clip values, save to directory"""
    datasetbasename = "full"
    runs = loadruns(datadir=datadir, drop_first_n=200)
    for key in runs.keys():
        save_dataset(data=runs[key], path=f"{savedir}/{datasetbasename}/{key}")


def compute_angular_velocities_from_rotations(rotations: np.ndarray) -> np.ndarray:
    """compute angular velocity robot had to has in order to achieve next orientation in a time step"""
    import scripts.utils.linalg_utils as lau
    n_data = len(rotations)
    ang = np.zeros(n_data)
    ang[0] = lau.compute_angular_velocity_from_orientation(orn_now=np.array([1,0,0,0]),
                                                           orn_next=rotations[0], time_passed=1. / 200.)
    for i in range(1, n_data):
        ang[i] = lau.compute_angular_velocity_from_orientation(orn_now=rotations[i - 1],
                                                               orn_next=rotations[i], time_passed=1. / 200.)
    return ang


def compute_linear_velocities_from_positions_and_rotations(positions: np.ndarray, rotations: np.ndarray) -> np.ndarray:
    """compute velocity robot had to has in order to move to next position in a time step"""
    import scripts.utils.linalg_utils as lau
    n_data = len(positions)
    vel = np.zeros((n_data, 3))
    for i in range(1, n_data):
        vel[i] = lau.compute_linear_velocity_from_position(pos_now=positions[i-1], pos_next=positions[i],
                                                           orn_now=rotations[i], time_passed=1./200.)
    return vel


def lower_data_frequency(datadir: str, savedir: str, del_freq: int = 2):
    """in case data were recorded from sensors with higher frequency than needed,
    remove every nth observation to lower frequency"""
    runs = loadruns(datadir=datadir, drop_first_n=0)
    for k in runs.keys():
        for key in runs[k].keys():
            runs[k][key] = np.delete(runs[k][key], np.arange(1, len(runs[k][key]), del_freq), axis=0)
        save_dataset(data=runs[k], path=f"{savedir}/{k}")


def stack_datasets(datadirs: list, savedir: str):
    """stack data from different datasets to one big dataset"""
    runs = {}
    counter = 1
    for i in range(len(datadirs)):
        subsetruns = loadruns(datadir=datadirs[i], drop_first_n=0)
        for key in subsetruns.keys():
            runs[f"run{counter}"] = subsetruns[key]
            counter += 1
    for k in runs.keys():
        save_dataset(data=runs[k], path=f"{savedir}/{k}")


def merge_raw_data_to_dataset(local_path: str):
    """load all raw episodes, merge them to create a dataset and save to a corresponding directory"""
    create_leave_one_out_train_test_datasets(datadir=f"{pm.paths['rawdata']}/{local_path}", savedir=f"{pm.paths['ds']}/{local_path}")
    create_full_dataset(datadir=f"{pm.paths['rawdata']}/{local_path}", savedir=f"{pm.paths['ds']}/{local_path}")


if __name__ == "__main__":
    create_full_dataset(datadir=f"{pm.paths['rawdata']}/concrete_new",
                        savedir=f"{pm.paths['ds']}/concrete_new")
    pass