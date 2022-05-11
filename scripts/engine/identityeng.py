import os.path

from scripts.engine.modelbased import ModelBased
from scripts.models.identitymodel import IdentityModel
from scripts.datamanagement.datamanagementutils import load_raw_data
from scripts.constants import Dirs, DT
import numpy as np


class IdentityEng(ModelBased):
    def __init__(self, datadir: str):
        """
        :param datadir: relative directory with ground truth data
        """
        super().__init__()
        path = os.path.join(Dirs.realdata, datadir)
        lpath = os.path.join(path, f"{DT.lin}.npy")
        apath = os.path.join(path, f"{DT.ang}.npy")
        self.linear = load_raw_data(path=lpath)
        self.angular = load_raw_data(path=apath)
        self.state.set(vel=self.linear[0], ang=self.angular[0])
        self.counter = 1

    def initializemodel(self) -> IdentityModel:
        return IdentityModel()

    def makeobs(self, throttle: float, turn: float) -> np.ndarray:
        i = self.counter
        dvel = self.linear[i, :2] - self.getlin()[:2]
        dang = self.angular[i, 2] - self.getang()[2]
        return np.hstack((dvel, dang))

    def step(self, throttle: float, turn: float):
        if self.counter < self.linear.shape[0]:
            super(IdentityEng, self).step(throttle=throttle, turn=turn)
            self.counter += 1


if __name__ == "__main__":
    datadir = "2022_04_07_15_52_42_186072"
    eng = IdentityEng(datadir=datadir)
    n = 4000
    pos = np.zeros((n, 3))
    for i in range(n):
        pos[i] = eng.getpos()
        eng.step(throttle=0, turn=0)

    path = os.path.join(Dirs.realdata, datadir)
    pospath = os.path.join(path, f"{DT.pos}.npy")
    gtpos = load_raw_data(path=pospath)
    import matplotlib.pyplot as plt
    plt.figure()
    plt.title("computed")
    plt.plot(pos[:, 0], pos[:, 1])
    plt.figure()
    plt.title("gt")
    plt.plot(gtpos[:n, 0], gtpos[:n, 1])
    plt.show()
