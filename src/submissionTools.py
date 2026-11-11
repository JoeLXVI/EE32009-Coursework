"""Checks and zipping for the Moodle submission (paste into main.py).

Needs: import os, zipfile, numpy as np, scipy.io as spio
"""

import os
import zipfile

import numpy as np
import scipy.io as spio

recordingLength = 1440000  # samples in every dataset


def validateSubmission(path, nSamples=recordingLength):
    """Return a list of problems with a D{n}.mat file (an empty list means it looks fine)."""
    problems = []
    mat = spio.loadmat(path)
    variables = sorted(k for k in mat if not k.startswith("__"))
    if variables != ["Class", "Index"]:
        problems.append(
            f"variables are {variables}; expected exactly ['Class', 'Index'] (no d)"
        )
        return problems

    index, classification = mat["Index"], mat["Class"]
    if index.ndim != 2 or index.shape[0] != 1 or classification.shape != index.shape:
        problems.append(
            f"shapes {index.shape} and {classification.shape}; expected (1, N) for both"
        )
    index, classification = index.ravel().astype(np.int64), classification.ravel()
    if len(index) == 0:
        problems.append("no spikes in the file")
        return problems
    if index.min() < 0 or index.max() >= nSamples:
        problems.append(f"Index outside the recording: {index.min()} to {index.max()}")
    if np.any(np.diff(index) <= 0):
        problems.append(
            "Index is not strictly increasing (unsorted or duplicate positions)"
        )
    if not np.all(np.isin(classification, [1, 2, 3, 4, 5])):
        problems.append(
            f"Class values must be 1 to 5, found {np.unique(classification)}"
        )
    return problems


def zipSubmission(matPaths, zipPath="data/outputs/submission.zip"):
    """Zip D{n}.mat files with no folder inside the archive. Refuses invalid files."""
    for path in matPaths:
        problems = validateSubmission(path)
        if problems:
            raise ValueError(f"{path}: " + "; ".join(problems))
    with zipfile.ZipFile(zipPath, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in matPaths:
            archive.write(
                path, arcname=os.path.basename(path)
            )  # 'D2.mat', not 'data/outputs/D2.mat'
    print(
        f"wrote {zipPath} ({os.path.getsize(zipPath)} bytes) containing "
        f"{[os.path.basename(p) for p in matPaths]}"
    )
