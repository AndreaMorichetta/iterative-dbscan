"""Silhouette computation on precomputed distance matrices, with noise exclusion.

This is a re-implementation of the LENTA silhouette logic from
Morichetta & Mellia (IEEE TNSM 2019). Differences from the original:

- No monkey-patching of sklearn internals.
- Operates on the public sklearn API only.
- Explicit handling of the noise label (-1) instead of relying on a label-encoder
  side-effect that mapped -1 to position 0.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

NOISE_LABEL = -1


def silhouette_samples_with_noise(
    distance_matrix: NDArray[np.floating],
    labels: NDArray[np.integer],
    noise_label: int = NOISE_LABEL,
) -> NDArray[np.float64]:
    """Per-sample silhouette on a precomputed distance matrix, ignoring noise.

    Parameters
    ----------
    distance_matrix : (n, n) symmetric distance matrix.
    labels          : (n,) cluster labels. ``noise_label`` (default -1) is excluded
                      from both intra and inter-cluster distance computations.
    noise_label     : the label value treated as noise.

    Returns
    -------
    (n,) array of silhouette scores. Noise samples and singleton clusters get 0.
    """
    distance_matrix = np.asarray(distance_matrix, dtype=np.float64)
    labels = np.asarray(labels)
    n = distance_matrix.shape[0]
    if distance_matrix.shape != (n, n):
        raise ValueError(f"distance_matrix must be square, got {distance_matrix.shape}")
    if labels.shape != (n,):
        raise ValueError(f"labels must have length {n}, got {labels.shape}")

    unique_labels = np.array([lbl for lbl in np.unique(labels) if lbl != noise_label])
    sil = np.zeros(n, dtype=np.float64)
    if len(unique_labels) < 2:
        return sil  # silhouette undefined with fewer than 2 real clusters

    intra = np.zeros(n, dtype=np.float64)
    inter = np.full(n, np.inf, dtype=np.float64)

    for lbl in unique_labels:
        mask = labels == lbl
        size = int(mask.sum())
        rows = distance_matrix[mask]  # (size, n)

        if size > 1:
            intra[mask] = rows[:, mask].sum(axis=1) / (size - 1)

        for other in unique_labels:
            if other == lbl:
                continue
            other_mask = labels == other
            mean_to_other = rows[:, other_mask].mean(axis=1)
            inter[mask] = np.minimum(inter[mask], mean_to_other)

    denom = np.maximum(intra, inter)
    with np.errstate(invalid="ignore", divide="ignore"):
        sil = np.where(denom > 0, (inter - intra) / denom, 0.0)

    # singletons and noise get 0
    counts = np.bincount(np.where(labels == noise_label, 0, labels + 1))
    singleton_mask = np.array([counts[lbl + 1] == 1 if lbl != noise_label else True for lbl in labels])
    sil[singleton_mask] = 0.0
    sil[labels == noise_label] = 0.0

    return sil


def mean_silhouette_per_cluster(
    distance_matrix: NDArray[np.floating],
    labels: NDArray[np.integer],
    noise_label: int = NOISE_LABEL,
) -> dict[int, float]:
    """Mean silhouette for each non-noise cluster."""
    sil = silhouette_samples_with_noise(distance_matrix, labels, noise_label=noise_label)
    out: dict[int, float] = {}
    for lbl in np.unique(labels):
        if lbl == noise_label:
            continue
        out[int(lbl)] = float(sil[labels == lbl].mean())
    return out
