import numpy as np

from scripts.simulation.pybulletvisualizer import Visualizer
from scripts.simulation.inputwrapper import InputWrapper
from scripts.engine.modelbased import ModelBased
import time


class Simulator:
    """
    input: either several models and no trajectory following visualization
    or a single engine with trajectory following feature
    """
    def __init__(self, iw: InputWrapper, engine: ModelBased):
        self.iw = iw
        self.engine = engine
        self.timestep = engine.state.timestep
        self.counter = 0

    def step(self, throttle: float, turn: float):
        """
        Update state

        :param throttle: forward action
        :param turn: sideways action
        """
        self.engine.step(throttle=throttle, turn=turn)
        self.counter += 1

    def simulate(self):
        """run a simulation"""
        terminate = False
        positions = []
        while not terminate:
            throttle, turn, terminate = self.iw.getinput()
            positions.append(self.engine.getpos())
            self.step(throttle=throttle, turn=turn)
        return np.asarray(positions)


if __name__ == "__main__":
    from scripts.simulation.joystickinputwrapper import JoystickInputWrapper
    from scripts.engine.mlpbased import MLPBased
    from scripts.engine.tcnnbased import TCNNBased
    from scripts.engine.identityeng import IdentityEng
    from scripts.constants import Dirs
    import os
    path = os.path.join(Dirs.models, "tcnn_2022_04_22_11_27_58_275542")
    # engine = MLPBased(path=path)
    engine = TCNNBased(path=path, visualize=True)

    # engine = IdentityEng(datadir="2022_04_12_15_09_00_833808")

    # sim = Simulator(iw=JoystickInputWrapper(), engine=engine)
    # sim.simulate()
    # exit()

    from scripts.simulation.datainputwrapper import DataWrapper
    from scripts.datamanagement.datamanagement import loadconfig
    from scripts.datamanagement.datamanagementutils import load_raw_data
    config = loadconfig(f"{path}.yaml")

    limit = 1500
    path = os.path.join(Dirs.realdata, "2022_04_10_11_57_44_706120")
    positions = -load_raw_data(path=f"{path}/positions.npy")
    actions = load_raw_data(path=f"{path}/actions.npy")
    linear = load_raw_data(path=f"{path}/linear.npy")
    angular = load_raw_data(path=f"{path}/angular.npy")

    sim = Simulator(iw=DataWrapper(actions=actions[:limit]), engine=engine)
    simpositions = sim.simulate()

    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(positions[:limit, 0], positions[:limit, 1], color='b')
    plt.plot(simpositions[:, 0], simpositions[:, 1], color='r')
    plt.legend(['gt', 'sim'])
    plt.show()
