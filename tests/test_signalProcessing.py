"""Tests of the detection, labelling and windowing functions on small synthetic signals."""

import numpy as np
import pytest

from main import (
    blockOf,
    buildTrainingSet,
    cutWindows,
    detectPeaks,
    labelPeaks,
    postWindowSize,
    preWindowSize,
)


def makeSignal(length, peaks):
    """Zeros with a short triangular bump of the given height at each position."""
    x = np.zeros(length)
    for position, height in peaks:
        x[position - 1], x[position], x[position + 1] = height / 2, height, height / 2
    return x


def test_detectPeaksFindsInjectedPeaks():
    x = makeSignal(10000, [(1000, 10), (3000, 10), (7000, 10)])
    assert list(detectPeaks(x, sigma=1.0, k=5, distance=30)) == [1000, 3000, 7000]


def test_detectPeaksIgnoresBumpsBelowThreshold():
    x = makeSignal(10000, [(1000, 10), (3000, 3)])
    assert list(detectPeaks(x, sigma=1.0, k=5, distance=30)) == [1000]


def test_detectPeaksDistanceKeepsTheLargerOfTwoClosePeaks():
    x = makeSignal(5000, [(1000, 10), (1010, 9)])
    assert list(detectPeaks(x, 1.0, k=5, distance=30)) == [1000]
    assert list(detectPeaks(x, 1.0, k=5, distance=5)) == [1000, 1010]


def test_cutWindowsShapeAndAlignment():
    x = np.arange(1000, dtype=float)
    windows, keep = cutWindows(x, [500])
    assert windows.shape == (1, preWindowSize + postWindowSize)
    assert windows[0, preWindowSize] == 500  # the position sits at column `pre`


def test_cutWindowsDropsPositionsTooCloseToTheEnds():
    x = np.zeros(1000)
    windows, keep = cutWindows(
        x, [0, 5, preWindowSize, 500, 1000 - postWindowSize, 1000 - postWindowSize + 1]
    )
    assert list(keep) == [False, False, True, True, True, False]
    assert len(windows) == keep.sum()


def test_cutWindowsOnNoPositions():
    windows, keep = cutWindows(np.zeros(1000), [])
    assert windows.shape[0] == 0 and len(keep) == 0


def test_blockOfBoundaries():
    positions = [0, 287999, 288000, 1439999]
    assert list(blockOf(positions, 1440000)) == [0, 0, 1, 4]


def test_blockOfEmptyInput():
    assert len(blockOf([], 1440000)) == 0


def test_labelPeaksRules():
    index, classification = [100, 1000, 5000], [1, 2, 3]
    peaks = [50, 105, 125, 126, 1011, 4999]
    #   before first spike -> 0 | 5 after -> 1 | 25 after (gap limit) -> 1 | 26 after -> 0 | 11 after -> 2 | far from anything -> 0
    assert list(labelPeaks(peaks, index, classification, gap=25)) == [0, 1, 1, 0, 2, 0]


def test_labelPeaksOnEmptyInput():
    assert len(labelPeaks([], [100, 1000], [1, 2])) == 0


def test_buildTrainingSetRowCounts():
    x = np.zeros(5000)
    X, y, refPos = buildTrainingSet(x, [500, 2000, 4000], [1, 2, 3], maxShift=2)
    assert X.shape == (3 * 5, preWindowSize + postWindowSize)
    assert len(y) == len(refPos) == 15
    assert sorted(set(y)) == [1, 2, 3]


def test_buildTrainingSetDropsWholeSpikesNearTheEdge():
    x = np.zeros(5000)
    X, y, refPos = buildTrainingSet(x, [5, 2000], [1, 2], maxShift=2)
    assert X.shape[0] == 5 and set(y) == {2}


def test_shiftedCopiesPlaceTheSpikeAtTheExpectedColumn():
    x = np.zeros(5000)
    reference = 2000 + 11  # Index + referenceOffset
    x[reference] = 1.0
    X, y, refPos = buildTrainingSet(x, [2000], [1], maxShift=2)
    shifts = np.arange(-2, 3)
    assert list(X.argmax(axis=1)) == list(
        preWindowSize - shifts
    )  # window cut at reference + shift
    assert set(refPos) == {reference}  # unshifted position, for the block split
