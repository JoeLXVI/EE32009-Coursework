"""autoMarker.py - local approximation of the EE32009 automarker (numpy only).

Detection: precision / recall of predicted Index against the true Index, one-to-one, within
+/- 50 samples. Classification: confusion matrix over the MATCHED pulses only (as the
feedback describes). The brief does not say how the class score is formed from that matrix,
so both a macro-F1 and an accuracy version are reported.
"""

import numpy as np

tolerance = 50
identWeight, classWeight = 0.3, 0.7
datasetWeights = {2: 0.10, 3: 0.15, 4: 0.20, 5: 0.25, 6: 0.30}
nClasses = 5


def f1Score(precision, recall):
    """Harmonic mean of precision and recall (0 if both are 0)."""
    return (
        2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    )


def matchSpikes(trueIdx, predIdx, tol=tolerance):
    """One-to-one match within +/- tol samples. Returns (n, 2) array of
    (position in trueIdx, position in predIdx), referring to the ORIGINAL arrays."""
    trueIdx = np.atleast_1d(np.asarray(trueIdx)).astype(np.int64)
    predIdx = np.atleast_1d(np.asarray(predIdx)).astype(np.int64)
    orderT, orderP = np.argsort(trueIdx, kind="stable"), np.argsort(
        predIdx, kind="stable"
    )
    ts, ps = trueIdx[orderT], predIdx[orderP]

    pairs, i, j = [], 0, 0
    while i < len(ts) and j < len(ps):
        gap = ps[j] - ts[i]
        if gap < -tol:  # prediction is too early to match any remaining true spike
            j += 1
        elif gap > tol:  # true spike was missed
            i += 1
        else:  # within tolerance: pair them, use each only once
            pairs.append((orderT[i], orderP[j]))
            i += 1
            j += 1
    return np.array(pairs, dtype=np.int64).reshape(-1, 2)


def confusionOnMatched(trueCls, predCls, pairs):
    """5x5 counts over matched pulses only. Rows = actual class, columns = predicted class."""
    trueCls = np.atleast_1d(np.asarray(trueCls)).astype(np.int64)
    predCls = np.atleast_1d(np.asarray(predCls)).astype(np.int64)
    cm = np.zeros((nClasses, nClasses), dtype=np.int64)
    for t, p in pairs:
        cm[trueCls[t] - 1, predCls[p] - 1] += 1
    return cm


def classScores(cm):
    """Per-class F1, macro F1 and accuracy from a confusion matrix."""
    tp = np.diag(cm).astype(float)
    precision = np.divide(
        tp, cm.sum(axis=0), out=np.zeros(nClasses), where=cm.sum(axis=0) > 0
    )
    recall = np.divide(
        tp, cm.sum(axis=1), out=np.zeros(nClasses), where=cm.sum(axis=1) > 0
    )
    perClass = np.array([f1Score(p, r) for p, r in zip(precision, recall)])
    accuracy = tp.sum() / cm.sum() if cm.sum() > 0 else 0.0
    return {
        "perClassF1": perClass,
        "macroF1": float(perClass.mean()),
        "accuracy": float(accuracy),
    }


def scoreDataset(trueIdx, trueCls, predIdx, predCls, tol=tolerance):
    """Everything the automarker reports for one dataset, plus the combined score."""
    nTrue, nPred = len(np.atleast_1d(trueIdx)), len(np.atleast_1d(predIdx))
    pairs = matchSpikes(trueIdx, predIdx, tol)
    precision = len(pairs) / nPred if nPred else 0.0
    recall = len(pairs) / nTrue if nTrue else 0.0
    fIdent = f1Score(precision, recall)

    cm = confusionOnMatched(trueCls, predCls, pairs)
    cs = classScores(cm)
    return {
        "precision": precision,
        "recall": recall,
        "fIdent": fIdent,
        "confusion": cm,
        "perClassF1": cs["perClassF1"],
        "macroF1": cs["macroF1"],
        "accuracy": cs["accuracy"],
        "fDatasetMacro": identWeight * fIdent + classWeight * cs["macroF1"],
        "fDatasetAccuracy": identWeight * fIdent + classWeight * cs["accuracy"],
        "nPred": nPred,
        "nTrue": nTrue,
    }


def finalScore(datasetScores, key="fDatasetMacro"):
    """Weighted score over datasets: {dataset number (2-6): scoreDataset result}."""
    return sum(datasetWeights[n] * s[key] for n, s in datasetScores.items())


def printReport(name, s):
    """Print one dataset the way the feedback PDF presents it."""
    print(
        f"{name}: precision {s['precision']:.4f}  recall {s['recall']:.4f}  "
        f"(predicted {s['nPred']}, true {s['nTrue']})"
    )
    print(
        f"     matched accuracy {s['accuracy']:.3f}  macro F1 {s['macroF1']:.3f}  "
        f"per-class F1 {np.round(s['perClassF1'], 2)}  dataset score {s['fDatasetMacro']:.3f} / "
        f"{s['fDatasetAccuracy']:.3f}"
    )
    print(s["confusion"])
