"""Shared fixtures. pytest loads this file automatically."""

import os
from pathlib import Path

import numpy as np
import pytest

projectRoot = Path(__file__).resolve().parents[1]
os.chdir(
    projectRoot
)  # importDataset reads the relative path data/inputs/, so always run from the project root


@pytest.fixture(scope="session")
def d1Data():
    """(d, index, classification) for D1, loaded once per test session."""
    from main import importDataset

    if not (projectRoot / "data" / "inputs" / "D1.mat").exists():
        pytest.skip("data/inputs/D1.mat not found")
    d, index, classification = importDataset(1)
    return d, index.astype(np.int64), classification.astype(np.int64)


@pytest.fixture(scope="session")
def filteredD1(d1Data):
    """D1 after the 5 to 1500 Hz band-pass."""
    from main import bandpassFilter

    return bandpassFilter(d1Data[0], 5, 1500)


@pytest.fixture
def syntheticSpikes():
    """2400 well-separated fake spikes with random classes (seeded, so repeatable)."""
    rng = np.random.default_rng(0)
    index = np.cumsum(rng.integers(200, 1000, size=2400))
    classification = rng.integers(1, 6, size=index.size)
    return index, classification
