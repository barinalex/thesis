import numpy as np

def filterscan(scan: np.ndarray, bound: int = 300) -> np.ndarray:
    counter = 0
    index = 0
    for i in range(0, len(scan)):
        counter += 1
        if scan[i] == np.inf:
            if counter < bound:
                for j in range(index, i):
                    scan[j] = np.inf
            counter = 0
            index = i
    return scan

"""
import matplotlib.pyplot as plt
def todecart(scan: np.ndarray) -> np.ndarray:
    angleincrement = 2 * np.pi / len(scan)
    decart = np.zeros((len(scan), 2))
    for i in range(len(scan)):
        angle = angleincrement * i
        if scan[i] != np.inf:
            decart[i][1] = np.sin(angle) * scan[i]
            decart[i][0] = np.cos(angle) * scan[i] 
    return decart

plt.figure()
for i in range(15,20):
    scan = np.load(f"scan{i}.npy")
    s = filterscan(scan)
    d = todecart(s)
    plt.scatter(d[:,0], d[:,1])

plt.show()
"""


