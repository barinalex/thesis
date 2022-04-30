from stable_baselines3.common.noise import ActionNoise
import numpy as np
from opensimplex import OpenSimplex
import time


class SimplexNoise(ActionNoise):
    def __init__(self, dim: int = 1, smoothness: int = 13, multiplier: float = 1.5, clip: bool = True):
        super(SimplexNoise, self).__init__()
        self.idx = 0
        self.dim = dim
        self.smoothness = smoothness
        self.multiplier = multiplier
        self.clip = clip
        self.noisefun = OpenSimplex(seed=int((time.time() % 1) * 10000000))

    def __call__(self) -> np.ndarray:
        self.idx += 1
        noise = np.array([(self.noisefun.noise2(x=self.idx / self.smoothness, y=i))
                         for i in range(self.dim)], dtype=np.float64)
        noise *= self.multiplier
        if self.clip:
            noise = np.clip(noise, 1, 1)
        return noise.astype(dtype=np.float64)

    def __repr__(self) -> str:
        return f"SimplexNoise()"


if __name__ == "__main__":
    def gen_noise(n: int = 1000):
        noisegenerator = SimplexNoise(dim=2, smoothness=100, multiplier=10, clip=False)
        return np.asarray([noisegenerator() for _ in range(n)])

    def plotnoise():
        import matplotlib.pyplot as plt
        n = 1000
        data = gen_noise(n=n)
        data -= data[0]
        plt.scatter(data[:, 0], data[:, 1])
        plt.show()

    plotnoise()
