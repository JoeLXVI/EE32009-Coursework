"""Checks on the real D1 recording. The fast ones run by default; model training is marked slow."""

import numpy as np
import pytest
from sklearn.svm import SVC

from autoMarker import scoreDataset
from main import (
    buildTrainingSet,
    crossFittedPredict,
    cutWindows,
    detectPeaks,
    estimateNoise,
    labelPeaks,
)


def test_filteredNoiseLevelOfD1(filteredD1):
    assert estimateNoise(filteredD1) == pytest.approx(0.100, abs=0.005)


def test_peakCountAndLabelsOnD1(filteredD1, d1Data):
    _, index, classification = d1Data
    peaks = detectPeaks(filteredD1, estimateNoise(filteredD1), k=5, distance=30)
    assert len(peaks) == 2324
    labels = labelPeaks(peaks, index, classification)
    assert list(np.bincount(labels, minlength=6)) == [0, 458, 471, 450, 485, 460]


def test_windowsAroundDetectedPeaks(filteredD1):
    peaks = detectPeaks(filteredD1, estimateNoise(filteredD1), k=5, distance=30)
    windows, keep = cutWindows(filteredD1, peaks)
    assert windows.shape == (2324, 100)
    assert (
        np.mean(windows.argmax(axis=1) == 20) > 0.9
    )  # the peak sits at column 20 in most windows


@pytest.mark.parametrize(
    "shift, rows", [(0, 2432), (3, 17024), (6, 31616), (10, 51072)]
)
def test_trainingSetRowCounts(filteredD1, d1Data, shift, rows):
    _, index, classification = d1Data
    X, y, refPos = buildTrainingSet(filteredD1, index, classification, shift)
    assert X.shape == (rows, 100)


@pytest.mark.slow
def test_crossFittedScoreOnD1(filteredD1, d1Data):
    """Regression guard: the clean-D1 result from the earlier experiments (combined about 0.956)."""
    _, index, classification = d1Data
    predIndex, predClass = crossFittedPredict(
        filteredD1, index, classification, lambda: SVC(), k=5, distance=30
    )
    s = scoreDataset(index, classification, predIndex, predClass)
    assert s["precision"] > 0.99
    assert s["recall"] > 0.95
    assert s["accuracy"] > 0.95
    assert s["fDatasetMacro"] > 0.94
