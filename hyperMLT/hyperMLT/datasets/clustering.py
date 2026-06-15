from __future__ import annotations

import numpy as np
from sklearn.cluster import DBSCAN


def hierarchical_cluster(
    X,
    *,
    verbose: bool = False,
    eps: float = 0.1,
    min_samples: int = 100,
):

    X = np.asarray(X, dtype=np.float64)

    xmin = np.min(X, axis=0)
    xmax = np.max(X, axis=0)
    xmean = np.mean(X, axis=0)

    if X.shape[1] >= 3:
        xmin[2] = 0.0
        xmax[2] = 1.0

    scale = np.maximum(xmax - xmin, 1e-12)
    X_norm = (X - xmean[None, :]) / scale[None, :]

    npoints = len(X_norm[:, 0])

    if verbose:
        print("Performing clustering ...")

    model = DBSCAN(eps=float(eps), min_samples=int(min_samples))
    cluster_labels = model.fit_predict(X_norm)
    outliers = cluster_labels == -1

    if np.count_nonzero(outliers) < 0.1 * npoints:
        valid = ~outliers
    else:
        valid = np.ones(npoints, dtype=bool)

    if verbose:
        print(f"\t {np.count_nonzero(valid)}/{npoints}")

    return valid
