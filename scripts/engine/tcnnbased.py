import numpy as np
from scripts.engine.modelbased import ModelBased
from scripts.models.cnnmodel import CNNModel
from scripts.models.model import Model
from scripts.utils.queuebuffer import QueueBuffer


class TCNNBased(ModelBased):
    def __init__(self, path):
        """
        :param path: path to an existing model parameters and config files.
        """
        self.path = path
        super().__init__()
        self.seqlength = self.model.config["sequence_length"]
        self.inputdim = self.model.config["input_dim"]
        initvector = np.zeros(self.inputdim)
        initvector[-2] = -1
        self.buffer = QueueBuffer(size=self.seqlength, initvector=initvector)

    def initializemodel(self) -> Model:
        model = CNNModel()
        model.load(path=self.path)
        return model

    def makeobs(self, throttle: float, turn: float) -> np.ndarray:
        lin = self.getlin()[:2]
        ang = self.getang()[2]
        obs = np.hstack((lin, ang, throttle, turn))
        self.buffer.add(element=obs)
        obs = self.buffer.get_vector()
        obs = obs.reshape(1, self.seqlength, self.inputdim)
        return np.transpose(obs, (0, 2, 1))
