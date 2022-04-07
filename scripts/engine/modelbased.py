import numpy as np
from scripts.engine.engine import Engine
from scripts.models.model import Model
from abc import abstractmethod


class ModelBased(Engine):
    def __init__(self):
        super().__init__()
        self.model: Model = self.initializemodel()

    @abstractmethod
    def initializemodel(self) -> Model:
        """
        :return: configured model instance
        """
        pass

    @abstractmethod
    def makeobs(self) -> np.ndarray:
        """
        :return: observation vector
        """
        pass

    def step(self, throttle: float, turn: float):
        obs = self.makeobs()
        dvel, dang = self.model.predict(obs=obs)
        dvel = np.hstack((dvel, 0))
        dang = np.hstack((np.zeros(2), dang))
        self.state.update_velocities(dvel=dvel, dang=dang)
        self.state.step()
