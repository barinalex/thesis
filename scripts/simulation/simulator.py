import numpy as np

from scripts.simulation.pybulletvisualizer import Visualizer
from scripts.simulation.inputwrapper import InputWrapper
from scripts.engine.engine import Engine
import time


class Simulator:
    """
    input: either several models and no trajectory following visualization
    or a single engine with trajectory following feature
    """
    def __init__(self, iw: InputWrapper, engine: Engine):
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
    from scripts.engine.mujocoengine import MujocoEngine
    from scripts.engine.identityeng import IdentityEng
    from scripts.constants import Dirs
    import os
    path = os.path.join(Dirs.models, "mlp_2022_05_01_12_30_00_981419")
    engine = MLPBased(path=path)
    # engine = TCNNBased(path=path, visualize=False)
    # engine = MujocoEngine(visualize=False)


    # sim = Simulator(iw=JoystickInputWrapper(), engine=engine)
    # sim.simulate()
    # exit()

    from scripts.simulation.datainputwrapper import DataWrapper
    from scripts.datamanagement.datamanagement import loadconfig
    from scripts.datamanagement.datamanagementutils import load_raw_data
    config = loadconfig(f"{path}.yaml")

    limit = 1000
    episodes = ["2022_05_01_11_57_36_659432", "2022_05_01_12_00_50_750831",
                "2022_05_01_12_10_36_731951", "2022_05_01_12_14_00_452235"]

    import matplotlib.pyplot as plt
    figure, axis = plt.subplots(1, len(episodes))
    # tfigure, taxis = plt.subplots(1, len(episodes))
    # sfigure, saxis = plt.subplots(1, len(episodes))
    for i, episode in enumerate(episodes):
        path = os.path.join(Dirs.realdata, episode)
        # engine = IdentityEng(datadir=episode)
        positions = -load_raw_data(path=f"{path}/positions.npy")
        actions = load_raw_data(path=f"{path}/actions.npy")
        actions[:, 0] = actions[:, 0] * 2 - 1
        linear = load_raw_data(path=f"{path}/linear.npy")
        angular = load_raw_data(path=f"{path}/angular.npy")
        sim = Simulator(iw=DataWrapper(actions=actions[:limit]), engine=engine)
        simpositions = sim.simulate()
        engine.reset()
        # axis[i].legend(['gt', 'sim'])
        axis[i].set_xlabel("meters")
        axis[0].set_ylabel("meters")
        axis[i].plot(positions[:limit, 0], positions[:limit, 1], color='b')
        axis[i].plot(simpositions[:, 0], simpositions[:, 1], color='r')
        #
        # taxis[i].set_xlabel("timestamp")
        # taxis[0].set_ylabel("value")
        # taxis[i].plot(np.arange(limit), actions[:limit, 0], color='g')
        #
        # saxis[i].set_xlabel("timestamp")
        # saxis[0].set_ylabel("value")
        # saxis[i].plot(np.arange(limit), actions[:limit, 1], color='g')
    plt.show()
