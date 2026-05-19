"""Automatic epsilon selection via k-distance graph.

For each point, find the distance to its k-th nearest neighbour, then take the
``percentile``-th percentile of these distances as eps. This is the original
LENTA heuristic for adapting DBSCAN's eps to local density.

The original code parallelised this with a ThreadPoolExecutor and used np.float16
for the sorted array. Modern numpy can do the whole thing vectorised in one call,
and float32 keeps precision without exploding memory for the problem sizes this
package targets (up to a few tens of thousands of points).
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray


def kth_neighbour_distances(
    distance_matrix: NDArray[np.floating],
    k: int,
) -> NDArray[np.float64]:
    """Distance to the k-th nearest neighbour for each point.

    ``k`` is 1-indexed (k=1 is the nearest neighbour after self). The diagonal
    is ignored.
    """
    distance_matrix = np.asarray(distance_matrix, dtype=np.float64)
    n = distance_matrix.shape[0]
    if k < 1 or k >= n:
        raise ValueError(f"k must be in [1, {n - 1}], got {k}")

    # partial sort: column k after we fill the diagonal with inf so self isn't counted
    d = distance_matrix.copy()
    np.fill_diagonal(d, np.inf)
    partitioned = np.partition(d, kth=k - 1, axis=1)
    return partitioned[:, k - 1]


def auto_eps(
    distance_matrix: NDArray[np.floating],
    min_samples: int,
    percentile_to_cluster: float = 65.0,
    round_decimals: int = 2,
) -> float:
    """Pick an eps so that approximately ``percentile_to_cluster`` percent of
    points have their k-th neighbour within eps, with k = ``min_samples``.

    This is the original LENTA criterion. ``percentile_to_cluster`` ranges 0-100;
    lower means stricter (smaller eps, more noise), higher means looser.
    """
    if not 0 < percentile_to_cluster < 100:
        raise ValueError(f"percentile_to_cluster must be in (0, 100), got {percentile_to_cluster}")

    k_dist = kth_neighbour_distances(distance_matrix, k=min_samples)
    k_dist_sorted = np.sort(k_dist)
    idx = int((distance_matrix.shape[0] / 100.0) * percentile_to_cluster)
    idx = min(idx, len(k_dist_sorted) - 1)
    raw_eps = float(k_dist_sorted[idx])

    factor = 10 ** round_decimals
    return math.ceil(raw_eps * factor) / factor
