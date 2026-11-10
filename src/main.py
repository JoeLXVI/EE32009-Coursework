# Import Required Modules
import scipy.io as spio
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neural_network import MLPClassifier

# Background info
# five types of neurons i.e. 5 classs
# neurons may be overlapping


# Function to import datasets
# Train on dataset 1 and test
def importDataset(n):
    # Import MATLAB Dataset
    mat = spio.loadmat(f"data/inputs/D{n}.mat", squeeze_me=True)
    d = mat["d"]
    Index = mat["Index"] if n == 1 else False
    Class = mat["Class"] if n == 1 else False
    return d, Index, Class


def plotTimeSeries(d):
    time = np.linspace(0, 1440000 * (1 / 25000), 1440000)
    plt.plot(time, d)
    plt.xlabel("Time / s")
    plt.ylabel("Amplitude / mV")
    plt.show()


d, index, classification = importDataset(1)

clf = MLPClassifier(
    hidden_layer_sizes=(),
    activation="logistic",
    solver="sgd",
    learning_rate_init=0.3,
    max_iter=1000,
)

X = np.empty((0, len(index)))
y = np.empty((0, len(classification)))

inputs = []
targets = []

for i in range(0, 24311):
    inputs.append(d[i])
    targets.append(classification[i])

X = np.vstack((X, inputs))
y = np.vstack((y, targets))

y = np.argmax(y, axis=1)

clf.fit(X, y)
