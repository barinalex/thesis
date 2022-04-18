import numpy as np
from scripts.engine.modelbased import ModelBased
from scripts.models.mlpmodel import MLPModel
from scripts.models.model import Model
from scripts.utils.queuebuffer import QueueBuffer


class MLPBased(ModelBased):
    def __init__(self, path, visualize: bool = False):
        """
        :param path: path to an existing model parameters and config files.
        """
        self.path = path
        super().__init__(visualize=visualize)
        bufsize = self.model.config["sequence_length"]
        initvector = np.array([0, 0, 0, -1, 0])
        self.buffer = QueueBuffer(size=bufsize, initvector=initvector)

    def initializemodel(self) -> Model:
        model = MLPModel()
        model.load(path=self.path)
        return model

    def makeobs(self, throttle: float, turn: float) -> np.ndarray:
        lin = self.getlin()[:2]
        ang = self.getang()[2]
        obs = np.hstack((lin, ang, throttle, turn))
        self.buffer.add(element=obs)
        return self.buffer.get_vector()
