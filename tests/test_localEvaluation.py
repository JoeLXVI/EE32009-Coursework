"""Checks that the noisy-copy simulation behaves, and that it still agrees with the real automarker."""

import numpy as np
import pytest

from main import estimateNoise
from localEvaluation import evaluateLevel, realFilteredSigma, simulateFiltered


@pytest.mark.parametrize("level", [2, 4, 6])
def test_simulatedNoiseLevelIsCloseToTarget(d1Data, level):
    sigma = estimateNoise(simulateFiltered(d1Data[0], realFilteredSigma[level], seed=0))
    assert sigma == pytest.approx(realFilteredSigma[level], rel=0.12)


# recall and matched accuracy the Moodle automarker reported on 16 Nov 2026 for k=4, distance 20,
# model trained on clean D1
realFeedback = {2: (0.956, 0.965), 4: (0.801, 0.898)}


@pytest.mark.slow
@pytest.mark.parametrize("level", [2, 4])
def test_simulationAgreesWithTheRealAutomarker(d1Data, level):
    _, index, classification = d1Data
    d1 = d1Data[0]
    s = evaluateLevel(
        d1, index, classification, level, k=4, distance=20, trainOn="clean", seed=0
    )
    realRecall, realAccuracy = realFeedback[level]
    assert s["recall"] == pytest.approx(realRecall, abs=0.04)
    assert s["accuracy"] == pytest.approx(realAccuracy, abs=0.05)
