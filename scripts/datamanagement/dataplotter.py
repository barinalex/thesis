import numpy as np
import matplotlib.pyplot as plt
from scripts.datamanagement.datamanagementutils import load_raw_data
from scripts.constants import DATA_DIR

path = f"{DATA_DIR}/real/2022_04_07_15_52_42_186072"
positions = load_raw_data(path=f"{path}/positions.npy")
actions = load_raw_data(path=f"{path}/actions.npy")
linear = load_raw_data(path=f"{path}/linear.npy")
angular = load_raw_data(path=f"{path}/angular.npy")
plt.figure()
plt.plot(np.arange(linear.shape[0]), linear[:, 0])
plt.show()
