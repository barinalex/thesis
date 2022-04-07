from abc import ABC, abstractmethod
import numpy as np


class Model(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def predict(self, obs: np.ndarray) -> (np.ndarray, np.ndarray):
        """
        :param obs: observation vector
        :return: delta linear velocity shape (2,), delta angular velocity shape (1,)
        """
        pass