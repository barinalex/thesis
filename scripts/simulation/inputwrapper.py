from abc import ABC, abstractmethod


class InputWrapper(ABC):
    @abstractmethod
    def getinput(self) -> (float, float, bool):
        """
        :return: throttle, turn, True to end simulation
        """
        pass
