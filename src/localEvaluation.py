"""localEvaluation.py - score the pipeline on noisy copies of D1 calibrated to D2-D6.

Only D1 has labels, so the closest local test of D2-D6 is D1 with white noise added until its
FILTERED noise level matches each real dataset. Spikes are always predicted by a model that
never saw them (time-block cross-fitting).
"""

import numpy as np
from sklearn.svm import SVC

from main import (
    bandpassFilter,
    estimateNoise,
    detectPeaks,
    cutWindows,
    blockOf,
    buildTrainingSet,
    makeModel,
    nBlocks,
    maxShift,
    referenceOffset,
)
from autoMarker import scoreDataset, finalScore

lowCut, highCut = 5, 1500
# filtered noise level (5 to 1500 Hz, estimateNoise) measured on the real D2 to D6
realFilteredSigma = {2: 0.167, 3: 0.237, 4: 0.385, 5: 0.711, 6: 1.055}


def measureUnitSigma(nSamples, seed=123):
    """Filtered sigma that unit-variance white noise has after the band-pass."""
    rng = np.random.default_rng(seed)
    return estimateNoise(bandpassFilter(rng.normal(size=nSamples), lowCut, highCut))


def simulateFiltered(d1, targetSigma, seed):
    """Filtered D1 with white noise added so the filtered sigma is about targetSigma."""
    cleanSigma = estimateNoise(bandpassFilter(d1, lowCut, highCut))
    extra = np.sqrt(max(targetSigma**2 - cleanSigma**2, 0.0)) / measureUnitSigma(
        len(d1)
    )
    rng = np.random.default_rng(seed)
    return bandpassFilter(d1 + rng.normal(0.0, extra, size=len(d1)), lowCut, highCut)


def crossFittedPredict(
    trainFiltered,
    testFiltered,
    index,
    classification,
    estimatorFactory,
    k,
    distance,
    shifts=maxShift,
    buffer=200,
):
    """Detect peaks in testFiltered and classify them; each time block is classified by a
    model trained (on trainFiltered) from the other blocks only."""
    peakIndex = detectPeaks(testFiltered, estimateNoise(testFiltered), k, distance)
    X, keep = cutWindows(testFiltered, peakIndex)
    peakIndex = peakIndex[keep]
    trainX, trainY, trainPos = buildTrainingSet(
        trainFiltered, index, classification, shifts
    )
    n = len(testFiltered)
    predicted = np.zeros(len(peakIndex), dtype=np.int64)
    for block in range(nBlocks):
        mask = blockOf(peakIndex, n) == block
        lo, hi = block * n // nBlocks, (block + 1) * n // nBlocks
        trainMask = (trainPos < lo - buffer) | (trainPos >= hi + buffer)
        model = makeModel(estimatorFactory()).fit(trainX[trainMask], trainY[trainMask])
        predicted[mask] = model.predict(X[mask])
    return peakIndex - referenceOffset, predicted


def evaluateLevel(
    d1,
    index,
    classification,
    level,
    k,
    distance,
    trainOn="clean",
    seed=0,
    estimatorFactory=lambda: SVC(),
):
    """Score one simulated dataset (level = 2..6). trainOn: 'clean' or 'matched' (noise
    matched to this level, using a different noise realisation from the test copy)."""
    testFiltered = simulateFiltered(d1, realFilteredSigma[level], seed)
    if trainOn == "clean":
        trainFiltered = bandpassFilter(d1, lowCut, highCut)
    else:
        trainFiltered = simulateFiltered(d1, realFilteredSigma[level], seed + 1000)
    predIdx, predCls = crossFittedPredict(
        trainFiltered,
        testFiltered,
        index,
        classification,
        estimatorFactory,
        k,
        distance,
    )
    return scoreDataset(index, classification, predIdx, predCls)


def evaluateAllLevels(
    d1,
    index,
    classification,
    ks,
    distance=20,
    trainOn="clean",
    seeds=(0, 1),
    estimatorFactory=lambda: SVC(),
):
    """ks: {level: threshold multiplier}. Returns {level: averaged scores} and prints a table."""
    results = {}
    for level, k in ks.items():
        runs = [
            evaluateLevel(
                d1,
                index,
                classification,
                level,
                k,
                distance,
                trainOn,
                s,
                estimatorFactory,
            )
            for s in seeds
        ]
        avg = {
            key: float(np.mean([r[key] for r in runs]))
            for key in (
                "precision",
                "recall",
                "fIdent",
                "accuracy",
                "macroF1",
                "fDatasetMacro",
                "fDatasetAccuracy",
            )
        }
        avg["perClassF1"] = np.mean([r["perClassF1"] for r in runs], axis=0)
        results[level] = avg
        print(
            f"D{level} k={k}: precision {avg['precision']:.3f} recall {avg['recall']:.3f} "
            f"matched acc {avg['accuracy']:.3f} macroF1 {avg['macroF1']:.3f} "
            f"score {avg['fDatasetMacro']:.3f}/{avg['fDatasetAccuracy']:.3f}"
        )
    if set(results) == {2, 3, 4, 5, 6}:
        print(
            f"estimated final score: macro {finalScore(results):.3f}, "
            f"accuracy {finalScore(results, 'fDatasetAccuracy'):.3f}"
        )
    return results
