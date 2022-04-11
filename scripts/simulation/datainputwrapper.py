import numpy as np
from scripts.simulation.inputwrapper import InputWrapper


class DataWrapper(InputWrapper):
    def __init__(self, actions: np.ndarray):
        self.actions = actions
        self.step = 0
        self.n_data = self.actions.shape[0]

    def getinput(self) -> (float, float, bool):
        action = self.actions[self.step]
        self.step += 1
        return action[0], action[1], self.step >= self.n_data - 1

