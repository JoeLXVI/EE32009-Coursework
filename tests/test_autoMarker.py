"""Tests of the local scorer using synthetic spikes (fast, no data files needed)."""

import numpy as np
import pytest

from autoMarker import (
    classScores,
    datasetWeights,
    finalScore,
    matchSpikes,
    scoreDataset,
    tolerance,
)


def test_perfectPredictionsScoreOne(syntheticSpikes):
    index, classification = syntheticSpikes
    s = scoreDataset(index, classification, index, classification)
    assert s["precision"] == 1 and s["recall"] == 1
    assert s["macroF1"] == 1 and s["fDatasetMacro"] == 1


def test_orderOfPredictionsDoesNotMatter(syntheticSpikes):
    index, classification = syntheticSpikes
    shuffled = np.random.default_rng(1).permutation(index.size)
    s = scoreDataset(index, classification, index[shuffled], classification[shuffled])
    assert s["fDatasetMacro"] == 1


@pytest.mark.parametrize(
    "shift, expectedRecall",
    [(10, 1.0), (tolerance, 1.0), (tolerance + 1, 0.0), (-60, 0.0)],
)
def test_toleranceWindow(syntheticSpikes, shift, expectedRecall):
    index, classification = syntheticSpikes
    s = scoreDataset(index, classification, index + shift, classification)
    assert s["recall"] == expectedRecall


def test_missingSpikesLowerRecallOnly(syntheticSpikes):
    index, classification = syntheticSpikes
    keep = np.random.default_rng(2).random(index.size) > 0.2
    s = scoreDataset(index, classification, index[keep], classification[keep])
    assert s["precision"] == 1
    assert s["recall"] == pytest.approx(keep.mean())
    assert s["accuracy"] == 1  # matched pulses are all classified correctly


def test_duplicatedPredictionsHalvePrecision(syntheticSpikes):
    index, classification = syntheticSpikes
    s = scoreDataset(
        index,
        classification,
        np.concatenate([index, index]),
        np.concatenate([classification, classification]),
    )
    assert s["precision"] == pytest.approx(0.5)
    assert s["recall"] == 1


def test_oneTrueSpikeCanOnlyBeMatchedOnce():
    pairs = matchSpikes([1000], [990, 1000, 1010])
    assert len(pairs) == 1


def test_wrongClassesKeepDetectionButZeroAccuracy(syntheticSpikes):
    index, classification = syntheticSpikes
    s = scoreDataset(index, classification, index, classification % 5 + 1)
    assert s["recall"] == 1 and s["accuracy"] == 0


def test_emptyPredictionsDoNotCrash(syntheticSpikes):
    index, classification = syntheticSpikes
    empty = np.array([], dtype=int)
    s = scoreDataset(index, classification, empty, empty)
    assert s["fDatasetMacro"] == 0


def test_confusionMatrixCountsMatchedPulsesOnly(syntheticSpikes):
    index, classification = syntheticSpikes
    keep = np.arange(index.size) % 2 == 0  # predict every other spike
    s = scoreDataset(index, classification, index[keep], classification[keep])
    assert s["confusion"].sum() == keep.sum()


def test_datasetWeightsSumToOne():
    assert sum(datasetWeights.values()) == pytest.approx(1.0)


def test_finalScoreOfPerfectDatasetsIsOne(syntheticSpikes):
    index, classification = syntheticSpikes
    perfect = scoreDataset(index, classification, index, classification)
    assert finalScore({n: perfect for n in range(2, 7)}) == pytest.approx(1.0)


def test_classScoresMatchTheRealAutomarkerFeedback():
    """D2 and D6 confusion matrices copied from the Moodle feedback of 16 Nov 2026."""
    d2 = np.array(
        [
            [535, 11, 2, 1, 5],
            [7, 548, 0, 0, 7],
            [6, 10, 541, 4, 4],
            [5, 8, 2, 540, 13],
            [4, 5, 0, 3, 518],
        ]
    )
    d6 = np.array(
        [
            [267, 126, 0, 3, 12],
            [83, 441, 2, 2, 12],
            [17, 16, 0, 0, 8],
            [36, 34, 3, 12, 15],
            [65, 124, 7, 12, 131],
        ]
    )
    assert classScores(d2)["accuracy"] == pytest.approx(0.965, abs=0.001)
    assert classScores(d6)["accuracy"] == pytest.approx(0.596, abs=0.001)
    assert classScores(d6)["macroF1"] == pytest.approx(0.398, abs=0.001)
