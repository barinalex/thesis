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
        self.visualizer = Visualizer(engine=engine)
        self.counter = 0

    def __enter__(self):
        return self

    def visualizer_step(self):
        """visualize step if visualization mode is on"""
        if self.visualizer:
            self.visualizer.step()

    def step(self, throttle: float, turn: float):
        """
        Update state

        :param throttle: forward action
        :param turn: sideways action
        """
        self.engine.step(throttle=throttle, turn=turn)
        self.counter += 1
        self.visualizer_step()

    def simulate(self):
        """run a simulation"""
        terminate = False
        while not terminate:
            throttle, turn, terminate = self.iw.getinput()
            start = time.time()
            self.step(throttle=throttle, turn=turn)
            print(self.engine.getlin())
            total = time.time() - start
            if total < self.timestep:
                time.sleep(self.timestep - total)
        self.disconnect_visualizer()

    def disconnect_visualizer(self):
        if self.visualizer:
            self.visualizer.disconnect()

    def __exit__(self, exc_type, exc_value, traceback):
        self.disconnect_visualizer()


if __name__ == "__main__":
    from scripts.simulation.joystickinputwrapper import JoystickInputWrapper
    from scripts.engine.mlpbased import MLPBased
    from scripts.constants import Dirs
    import os
    path = os.path.join(Dirs.models, "mlp_2022_04_11_20_50_31_362568")
    engine = MLPBased(path=path)
    # sim = Simulator(iw=JoystickInputWrapper(), engine=engine)
    # sim.simulate()
    # exit()

    from scripts.simulation.datainputwrapper import DataWrapper
    from scripts.datamanagement.datamanagement import loadconfig
    from scripts.datamanagement.datamanagementutils import load_raw_data
    config = loadconfig(f"{path}.yaml")

    path = os.path.join(Dirs.simdata, "2022_04_11_20_49_05_784963")
    positions = load_raw_data(path=f"{path}/positions.npy")
    actions = load_raw_data(path=f"{path}/actions.npy")
    linear = load_raw_data(path=f"{path}/linear.npy")
    angular = load_raw_data(path=f"{path}/angular.npy")

    # import matplotlib.pyplot as plt
    # plt.figure()
    # plt.plot(positions[:, 0], positions[:, 1])
    # plt.show()
    # exit()

    sim = Simulator(iw=DataWrapper(actions=actions), engine=engine)
    sim.simulate()
