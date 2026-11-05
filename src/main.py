# Import Required Modules
import scipy.io as spio
import matplotlib.pyplot as plt
import numpy as np


def importDataset(n):
    # Import MATLAB Dataset
    mat = spio.loadmat(f"data/inputs/D{n}.mat", squeeze_me=True)
    d = mat["d"] if mat["d"].any() else False
    Index = mat["Index"] if n == 1 else False
    Class = mat["Class"] if n == 1 else False
    return d, Index, Class


d, Index, Class = importDataset(1)

plt.plot(d)
plt.show()
