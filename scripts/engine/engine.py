import numpy as np
from scripts.engine.state import State
from abc import ABC, abstractmethod


class Engine(ABC):
    """abstract engine class"""
    def __init__(self):
        self.state = State()

    def reset(self):
        """set state to point zero"""
        self.state.reset()

    @abstractmethod
    def step(self, throttle: float, turn: float):
        """
        update agent's state

        :param throttle: forward action
        :param turn: sideways action
        """
        pass

    def getpos(self):
        """
        :return: position vector
        """
        return self.state.getpos()

    def getorn(self):
        """
        :return: quaternion in a format w x y z
        """
        return self.state.getorn()

    def getvel(self):
        """
        :return: linear velocity
        """
        return self.state.getvel()

    def getang(self):
        """
        :return: angular velocity
        """
        return self.state.getang()

    def readstate(self):
        """
        :return: state observation: position, velocity, orientation quaternion, angular velocity
        """
        return self.state.getpos(), self.state.getvel(), self.state.getorn(), self.state.getang()

    def get_state_vector(self):
        """
        :return: flat vector: position x y, velocity x y, orientation yaw, angular velocity yaw"""
        return np.hstack((self.getvel()[:2], self.getang()[2]))

    def toselfframe(self, vector: np.ndarray) -> np.ndarray:
        """
        :param vector: vector or vectors in a world frame
        :return: vector in an agent frame
        """
        return self.state.toselfframe(v=vector)[:, :3]

    def setstate(self, pos=None, orn=None, vel=None, ang=None):
        """
        :param pos: position vector
        :param orn: orientation as a quaternion in a format [w x y z]
        :param vel: linear velocity
        :param ang: angular velocity
        """
        self.state.set(pos=pos, orn=orn, vel=vel, ang=ang)
