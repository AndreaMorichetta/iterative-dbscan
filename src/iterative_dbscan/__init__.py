"""Iterative DBSCAN with silhouette-guided recursive refinement.

A modernised, pip-installable implementation of the LENTA algorithm
(Morichetta & Mellia, IEEE TNSM 2019).

Quick start
-----------
>>> import numpy as np
>>> from iterative_dbscan import IterativeDBSCAN
>>> X = np.random.RandomState(0).randn(100, 10)
>>> idb = IterativeDBSCAN(metric="euclidean", min_samples=5, smin=0.1)
>>> labels = idb.fit_predict(X)
>>> print(idb.cluster_tree_)
"""

from .core import IterativeDBSCAN
from .eps import auto_eps, kth_neighbour_distances
from .silhouette import (
    NOISE_LABEL,
    mean_silhouette_per_cluster,
    silhouette_samples_with_noise,
)
from .tree import ClusterNode, ClusterTree

__all__ = [
    "IterativeDBSCAN",
    "ClusterTree",
    "ClusterNode",
    "auto_eps",
    "kth_neighbour_distances",
    "silhouette_samples_with_noise",
    "mean_silhouette_per_cluster",
    "NOISE_LABEL",
]

__version__ = "0.1.0"
