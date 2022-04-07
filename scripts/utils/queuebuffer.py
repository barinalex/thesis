import numpy as np


class QueueBuffer:
    """circular buffer for action=[throttle, turn] storage"""
    def __init__(self, size, dim, initzeros: bool = False):
        self.size = size 
        self.dim = dim
        self.buffer = np.zeros((size, dim))
        self.initzeros = initzeros
        if not self.initzeros:
            # set throttle to -1
            self.buffer[:, -2] = -1
        self.idx = 0

    def reset(self):
        self.buffer = np.zeros((self.size, self.dim))
        if not self.initzeros:
            # set throttle to -1
            self.buffer[:, -2] = -np.ones(self.size)
        self.idx = 0

    def iterate(self, idx):
        """increase index"""
        return (idx + 1) % self.size

    def add(self, element):
        """push data to buffer"""
        if len(element) != self.dim:
            return
        self.buffer[self.idx] = np.copy(element) 
        self.idx = self.iterate(self.idx)

    def get_vector(self):
        """concatenate all buffer data to a flat vector"""
        idx = self.idx
        vector = self.buffer[idx]
        for _ in range(self.size-1):
            idx = self.iterate(idx)
            vector = np.hstack((vector, self.buffer[idx]))
        return vector

    def get_sequential_input(self):
        """return consequential observations as one sequence"""
        vector = self.get_vector()
        return vector.reshape((1, self.size, self.dim))


if __name__ == '__main__':
    buf = QueueBuffer(8, 2)
    print(buf.get_vector())
    buf.add(np.array([1, 2]))
    buf.add(np.array([2, 3]))
    print(buf.get_vector())

