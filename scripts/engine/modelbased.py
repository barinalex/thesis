import numpy as np
from scripts.engine.engine import Engine
from scripts.models.model import Model
from scripts.simulation.pybulletvisualizer import Visualizer
from abc import abstractmethod


class ModelBased(Engine):
    def __init__(self, visualize: bool = False):
        super().__init__()
        self.model: Model = self.initializemodel()
        self.n_wps = 10
        self.viewer = None
        if visualize:
            self.viewer = Visualizer()
            self.viewer.n_wps = self.n_wps
        self.lastwaypoint = np.zeros(2)

    def movewaypoint(self, pos: np.ndarray):
        """
        add new waypoint line and erase old one

        :param pos: waypoint position shape (2,)
        """
        if self.viewer:
            self.viewer.addline(from_=self.lastwaypoint, to_=pos)
            self.lastwaypoint = np.copy(pos)


    @abstractmethod
    def initializemodel(self) -> Model:
        """
        :return: configured model instance
        """
        pass

    @abstractmethod
    def makeobs(self, throttle: float, turn: float) -> np.ndarray:
        """
        :param throttle: forward action
        :param turn: sideways action
        :return: observation vector
        """
        pass

    def step(self, throttle: float, turn: float):
        obs = self.makeobs(throttle=throttle, turn=turn)
        dvel, dang = self.model.predict(obs=obs)
        dvel = np.hstack((dvel, 0))
        dang = np.hstack((np.zeros(2), dang))
        self.state.update_velocities(dvel=dvel, dang=dang)
        self.state.step()
        if self.viewer:
            self.viewer.step(pos=self.getpos(), orn=self.getorn())

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self.viewer:
            self.viewer.disconnect()
