from scripts.engine.modelbased import ModelBased
from scripts.models.identitymodel import IdentityModel
from scripts.datamanagement.datamanagementutils import load_raw_data
from scripts.constants import Dirs
import numpy as np


class IdentityEng(ModelBased):
    def __init__(self, datadir: str):
        """
        :param datadir: relative directory with ground truth data
        """
        super().__init__()
        path = f"{Dirs.realdata}/2022_04_07_15_52_42_186072"
        self.linear = load_raw_data(path=f"{path}/linear.npy")
        self.angular = load_raw_data(path=f"{path}/angular.npy")
        self.counter = 1

    def initializemodel(self) -> IdentityModel:
        return IdentityModel()

    def makeobs(self) -> np.ndarray:
        i = self.counter
        dvel = self.linear[i, :2] - self.linear[i-1, :2]
        dang = self.angular[i, 2] - self.angular[i-1, 2]
        return np.hstack((dvel, dang))

    def step(self, throttle: float, turn: float):
        if self.counter < self.linear.shape[0]:
            super(IdentityEng, self).step(throttle=throttle, turn=turn)
            self.counter += 1


if __name__ == "__main__":
    eng = IdentityEng(datadir="")
    n = 4000
    pos = np.zeros((n, 3))
    for i in range(n):
        pos[i] = eng.getpos()
        eng.step(throttle=0, turn=0)

    path = f"{Dirs.realdata}/2022_04_07_15_52_42_186072"
    gtpos = load_raw_data(path=f"{path}/positions.npy")
    import matplotlib.pyplot as plt
    plt.figure()
    plt.title("computed")
    plt.plot(pos[:, 0], pos[:, 1])
    plt.figure()
    plt.title("gt")
    plt.plot(gtpos[:n, 0], gtpos[:n, 1])
    plt.show()
