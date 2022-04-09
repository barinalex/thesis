import numpy as np
from scripts.engine.modelbased import ModelBased
from scripts.models.mlpmodel import MLPModel
from scripts.models.model import Model


class MLPBased(ModelBased):
    def __init__(self, path):
        """
        :param path: path to an existing model parameters and config files.
        """
        self.path = path
        super().__init__()

    def initializemodel(self) -> Model:
        model = MLPModel()
        model.load(path=self.path)
        return model

    def makeobs(self, throttle: float, turn: float) -> np.ndarray:
        lin = self.getlin()[:2]
        ang = self.getang()[2]
        return np.hstack((lin, ang, throttle, turn))
