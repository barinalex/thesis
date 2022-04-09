import numpy as np
import matplotlib.pyplot as plt
from scripts.datamanagement.datamanagement import loadconfig, get_data
from scripts.datamanagement.datamanagementutils import load_raw_data, reshape_no_batches
from scripts.constants import Dirs, DT
import os

# path = f"{Dirs.realdata}/2022_04_07_15_52_42_186072"
# positions = load_raw_data(path=f"{path}/positions.npy")
# actions = load_raw_data(path=f"{path}/actions.npy")
# linear = load_raw_data(path=f"{path}/linear.npy")
# angular = load_raw_data(path=f"{path}/angular.npy")
# plt.figure()
# plt.plot(np.arange(actions.shape[0]), actions[:, 0])
# plt.show()

path = os.path.join(Dirs.configs, "mlp.yaml")
config = loadconfig(path=path)
train, test = get_data(params=config)
obs, labels = reshape_no_batches(train[DT.obs], train[DT.labels])
plt.figure()
plt.plot(np.arange(obs.shape[0]), labels[:, 0])
plt.show()
