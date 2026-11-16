"""Tests that the files written for Moodle are in the format the automarker expects."""

import zipfile

import numpy as np
import scipy.io as spio

from main import saveSubmission
from submissionTools import validateSubmission, zipSubmission


def writeValid(path):
    saveSubmission(str(path), [300, 100, 200], [3, 1, 2])  # deliberately unsorted input


def test_savedFileIsSortedRowVectorsWithOnlyIndexAndClass(tmp_path):
    path = tmp_path / "D2.mat"
    writeValid(path)
    mat = spio.loadmat(str(path))
    assert sorted(k for k in mat if not k.startswith("__")) == ["Class", "Index"]
    assert mat["Index"].shape == mat["Class"].shape == (1, 3)
    assert list(mat["Index"].ravel()) == [100, 200, 300]
    assert list(mat["Class"].ravel()) == [
        1,
        2,
        3,
    ]  # classes follow their spikes when sorted


def test_validSubmissionHasNoProblems(tmp_path):
    path = tmp_path / "D2.mat"
    writeValid(path)
    assert validateSubmission(str(path)) == []


def test_validatorCatchesTheCommonMistakes(tmp_path):
    cases = {
        "includes d": {
            "d": np.zeros((1, 5)),
            "Index": np.array([[5]]),
            "Class": np.array([[1]]),
        },
        "unsorted": {"Index": np.array([[50, 10]]), "Class": np.array([[1, 2]])},
        "class zero": {"Index": np.array([[10, 50]]), "Class": np.array([[0, 2]])},
        "length mismatch": {"Index": np.array([[10, 50]]), "Class": np.array([[1]])},
        "out of range": {
            "Index": np.array([[10, 1440005]]),
            "Class": np.array([[1, 2]]),
        },
        "column vectors": {
            "Index": np.array([[10], [50]]),
            "Class": np.array([[1], [2]]),
        },
    }
    for name, content in cases.items():
        path = tmp_path / "bad.mat"
        spio.savemat(str(path), content)
        assert validateSubmission(str(path)) != [], name


def test_zipHasFilesAtTheTopLevel(tmp_path):
    first, second = tmp_path / "D2.mat", tmp_path / "D3.mat"
    writeValid(first)
    writeValid(second)
    archive = tmp_path / "submission.zip"
    zipSubmission([str(first), str(second)], str(archive))
    with zipfile.ZipFile(archive) as z:
        assert sorted(z.namelist()) == ["D2.mat", "D3.mat"]
