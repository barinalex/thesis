from scripts.simulation.pybulletvisualizer import Visualizer
from scripts.simulation.inputwrapper import InputWrapper
from scripts.engine.modelbased import ModelBased


class Simulator:
    """
    input: either several models and no trajectory following visualization
    or a single engine with trajectory following feature
    """
    def __init__(self, iw: InputWrapper, engine: ModelBased):
        self.iw = iw
        self.engine = engine
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
            print(throttle, turn)
            self.step(throttle=throttle, turn=turn)
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
    path = os.path.join(Dirs.models, "mlp_2022_04_09_20_04_52_223493")
    sim = Simulator(iw=JoystickInputWrapper(), engine=MLPBased(path=path))
    sim.simulate()
