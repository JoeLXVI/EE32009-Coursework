# Import Required Modules
import scipy.io as spio
from scipy import signal
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import os
from submissionTools import zipSubmission

# Constants
preWindowSize = 20
postWindowSize = 80
referenceOffset = 11
maxShift = 6
nBlocks = 5
peakHeight = 5
minDistance = 30
maxGap = 25
samplingFreq = 25000

# Background info
# five types of neurons i.e. 5 classs
# neurons may be overlapping


# Function to import datasets
# Train on dataset 1 and test
def importDataset(n):
    # Import MATLAB Dataset
    mat = spio.loadmat(f"data/inputs/D{n}.mat", squeeze_me=True)
    d = mat["d"]
    if n != 1:
        return d, None, None
    Index, Class = mat["Index"], mat["Class"]
    order = np.argsort(Index)
    return d, Index[order], Class[order]


def bandpassFilter(d, lowFreq, highFreq, order=3):
    # Butterworth bandpass filter
    sos = signal.butter(
        order, [lowFreq, highFreq], btype="bandpass", output="sos", fs=samplingFreq
    )
    y = signal.sosfiltfilt(sos, d)
    return y


def estimateNoise(d):
    d = np.asarray(d, dtype=float)
    deviation = np.median(np.abs(d - np.median(d)))
    return deviation / 0.6745  # For Gaussian noise, MAD = 0.6745 * sigma


def detectPeaks(filtered, sigma, k=peakHeight, distance=minDistance):
    positions, properties = signal.find_peaks(
        filtered, height=k * sigma, distance=distance
    )
    return positions


def labelPeaks(peakIndex, index, classification, gap=maxGap):
    peakIndex = np.asarray(peakIndex, dtype=np.int64)
    index = np.asarray(index, dtype=np.int64)
    classification = np.asarray(classification, dtype=np.int64)

    pos = np.searchsorted(index, peakIndex, side="right") - 1
    safePos = np.clip(pos, 0, None)
    distance = peakIndex - index[safePos]
    matched = (pos >= 0) & (distance >= 0) & (distance <= gap)
    return np.where(matched, classification[safePos], 0)


def cutWindows(filtered, positions, pre=preWindowSize, post=postWindowSize):
    positions = np.asarray(positions, dtype=np.int64)
    keep = (positions >= pre) & (positions + post <= len(filtered))
    idx = positions[keep][:, None] + np.arange(-pre, post)[None, :]
    return filtered[idx], keep


def blockOf(positions, nSamples, nBlocks=nBlocks):
    positions = np.asarray(positions, dtype=np.int64)
    blocks = positions * nBlocks // nSamples
    return np.clip(blocks, 0, nBlocks - 1)


def buildTrainingSet(filtered, index, classification, maxShift, offset=referenceOffset):
    index = np.asarray(index, dtype=np.int64)
    classification = np.asarray(classification, dtype=np.int64)
    reference = index + offset

    usable = (reference - maxShift - preWindowSize >= 0) & (
        reference + maxShift + postWindowSize <= len(filtered)
    )
    reference, labels = reference[usable], classification[usable]

    windows, y, refPos = [], [], []
    for shift in range(-maxShift, maxShift + 1):
        X, _ = cutWindows(filtered, reference + shift)
        windows.append(X)
        y.append(labels)
        refPos.append(reference)
    return np.concatenate(windows), np.concatenate(y), np.concatenate(refPos)


def makeModel(estimator, nComponents=10):
    return make_pipeline(StandardScaler(), PCA(nComponents), estimator)


def crossValidate(
    filtered,
    index,
    classification,
    peakIndex,
    peakLabels,
    esitmatorFactory,
    maxShift=maxShift,
    buffer=200,
):
    trainX, trainY, trainPos = buildTrainingSet(
        filtered, index, classification, maxShift
    )

    valX, keep = cutWindows(filtered, peakIndex)
    valY = np.asarray(peakLabels)[keep]
    valPos = np.asarray(peakIndex)[keep]
    labelled = valY > 0
    valX, valY, valPos = valX[labelled], valY[labelled], valPos[labelled]

    nSamples = len(filtered)
    accuracies, trueAll, predAll = [], [], []
    for block in range(nBlocks):
        valMask = blockOf(valPos, nSamples) == block
        lo, hi = block * nSamples // nBlocks, (block + 1) * nSamples // nBlocks
        trainMask = (trainPos < lo - buffer) | (trainPos >= hi + buffer)

        model = makeModel(esitmatorFactory()).fit(trainX[trainMask], trainY[trainMask])
        predicted = model.predict(valX[valMask])
        accuracies.append(float(np.mean(predicted == valY[valMask])))
        trueAll.append(valY[valMask])
        predAll.append(predicted)

    cm = confusion_matrix(
        np.concatenate(trueAll), np.concatenate(predAll), labels=[1, 2, 3, 4, 5]
    )
    return accuracies, cm


def trainFinalModel(
    filtered, index, classification, estimatorFactory, maxShift=maxShift
):
    X, y, _ = buildTrainingSet(filtered, index, classification, maxShift)
    return makeModel(estimatorFactory()).fit(X, y)


def predictSpikes(
    filtered, model, k=peakHeight, distance=minDistance, offset=referenceOffset
):
    sigma = estimateNoise(filtered)
    peakIndex = detectPeaks(filtered, sigma, k, distance)
    X, keep = cutWindows(filtered, peakIndex)
    predictedClass = model.predict(X)
    spikeIndex = peakIndex[keep] - offset
    return spikeIndex, predictedClass


def crossFittedPredict(
    filtered,
    index,
    classification,
    estimatorFactory,
    k=peakHeight,
    distance=minDistance,
    maxShift=maxShift,
    buffer=200,
):
    sigma = estimateNoise(filtered)
    peakIndex = detectPeaks(filtered, sigma, k, distance)
    X, keep = cutWindows(filtered, peakIndex)
    peakIndex = peakIndex[keep]
    trainX, trainY, trainPos = buildTrainingSet(
        filtered, index, classification, maxShift
    )
    n = len(filtered)
    predicted = np.zeros(len(peakIndex), dtype=np.int64)
    for block in range(nBlocks):
        mask = blockOf(peakIndex, n) == block
        lo, hi = block * n // nBlocks, (block + 1) * n // nBlocks
        trainMask = (trainPos < lo - buffer) | (trainPos >= hi + buffer)
        model = makeModel(estimatorFactory()).fit(trainX[trainMask], trainY[trainMask])
        predicted[mask] = model.predict(X[mask])
    return peakIndex - referenceOffset, predicted


def saveSubmission(path, index, classification):
    index = np.asarray(index)
    order = np.argsort(index)
    spio.savemat(
        path,
        {
            "Index": index[order].astype(np.int32).reshape(1, -1),
            "Class": np.asarray(classification)[order].astype(np.int32).reshape(1, -1),
        },
    )


def runClassifierExperiment(filteredD, index, classification):
    """Time-block cross-validation of each classifier, with confusion matrices."""
    sigma = estimateNoise(filteredD)
    peakIndex = detectPeaks(filteredD, sigma)
    peakLabels = labelPeaks(peakIndex, index, classification)
    print(
        "peaks:",
        len(peakIndex),
        "labels (0 = unmatched):",
        np.bincount(peakLabels, minlength=6),
    )

    models = {"LDA": lambda: LinearDiscriminantAnalysis(), "SVC": lambda: SVC()}
    for name, factory in models.items():
        accuracies, cm = crossValidate(
            filteredD, index, classification, peakIndex, peakLabels, factory, maxShift
        )
        print(
            f"{name}: mean accuracy {np.mean(accuracies):.3f}, per block {np.round(accuracies, 3)}"
        )
        print(cm)
        ConfusionMatrixDisplay(cm, display_labels=[1, 2, 3, 4, 5]).plot()
        plt.title(f"{name}, maxShift = {maxShift}")


def regenerateD1(filteredD, index, classification, path="data/outputs/D1.mat"):
    """Task 1: regenerate Index and Class for D1 and save them (out-of-sample predictions)."""
    spikeIndex, spikeClass = crossFittedPredict(
        filteredD, index, classification, lambda: SVC()
    )
    os.makedirs(os.path.dirname(path), exist_ok=True)
    saveSubmission(path, spikeIndex, spikeClass)
    print(f"saved {len(spikeIndex)} spikes to {path}")


def generateSubmission(n, model, k, distance, outDir="data/outputs"):
    d, _, _ = importDataset(n)
    filtered = bandpassFilter(d, 5, 1500)
    spikeIndex, spikeClass = predictSpikes(filtered, model, k, distance)
    os.makedirs(outDir, exist_ok=True)
    saveSubmission(f"{outDir}/D{n}.mat", spikeIndex, spikeClass)
    return len(spikeIndex)


if __name__ == "__main__":
    d1, index1, classification1 = importDataset(1)
    filteredD1 = bandpassFilter(d1, 5, 1500)
    model = trainFinalModel(filteredD1, index1, classification1, lambda: SVC())
    generateSubmission(2, model, k=4, distance=20)
    zipSubmission(["data/outputs/D2.mat"])
    # runClassifierExperiment(filteredD, index, classification)
    # regenerateD1(filteredD, index, classification)
    # plt.show()
