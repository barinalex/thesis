from torch import nn
from abc import ABC, abstractmethod


class NeuralNetwork(nn.Module, ABC):
    def __init__(self, config: dict):
        super(NeuralNetwork, self).__init__()
        self.config = config

    @abstractmethod
    def predict(self, x):
        """
        :param x: input vector
        """
        pass
