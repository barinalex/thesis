import numpy as np
from scripts.models.model import Model


class IdentityModel(Model):
    def __init__(self):
        super().__init__()

    def predict(self, obs: np.ndarray) -> (np.ndarray, np.ndarray):
        return obs[:2], obs[2]

