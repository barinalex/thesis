from stable_baselines3.common.noise import ActionNoise
import numpy as np
from opensimplex import OpenSimplex
import time


class SimplexNoise(ActionNoise):
    def __init__(self, dim: int = 1, smoothness: int = 13, multiplier: float = 1.5):
        super(SimplexNoise, self).__init__()
        self.idx = 0
        self.dim = dim
        self.smoothness = smoothness
        self.multiplier = multiplier
        self.noisefun = OpenSimplex(seed=int((time.time() % 1) * 10000000))

    def __call__(self) -> np.ndarray:
        self.idx += 1
        noise = np.array([(self.noisefun.noise2(x=self.idx / self.smoothness, y=i))
                         for i in range(self.dim)], dtype=np.float64)
        return np.clip(noise * self.multiplier, -1, 1).astype(dtype=np.float64)

    def __repr__(self) -> str:
        return f"SimplexNoise()"


if __name__ == "__main__":
    def gen_noise(n: int = 1000):
        noisegenerator = SimplexNoise(dim=1, smoothness=100, multiplier=2)
        return np.asarray([(noisegenerator() + 1) / 4 + 0.5 for _ in range(n)])

    def plotnoise():
        import matplotlib.pyplot as plt
        n = 1000
        plt.plot(np.arange(n), gen_noise(n=n)[:, ])
        plt.show()

    plotnoise()
