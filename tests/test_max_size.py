"""Tests for the max_cluster_size split trigger (extension to original LENTA)."""

import numpy as np

from iterative_dbscan import IterativeDBSCAN


def test_max_cluster_size_triggers_split_on_isolated_mega_cluster():
    """An isolated heterogeneous cluster keeps high silhouette but should still
    be split if max_cluster_size is set."""
    rng = np.random.RandomState(0)
    A = rng.randn(30, 2) * 0.3 + np.array([0, 0])
    # B is two sub-blobs at (30, 5) and (30, 8) — separable, but far from A
    B1 = rng.randn(20, 2) * 0.2 + np.array([30, 5])
    B2 = rng.randn(20, 2) * 0.2 + np.array([30, 8])
    X = np.vstack([A, B1, B2])

    # smin can't trigger because cluster B is well-isolated and has high silhouette
    no_size_cap = IterativeDBSCAN(
        metric="euclidean", min_samples=3, smin=0.9, eps=4.0, max_cluster_size=None
    )
    no_size_cap.fit(X)
    # B1+B2 should be one big cluster (40 points)
    sizes_no_cap = sorted([(no_size_cap.labels_ == lbl).sum() for lbl in np.unique(no_size_cap.labels_) if lbl != -1])
    assert 40 in sizes_no_cap

    # But with max_cluster_size=35, the 40-point cluster gets split
    with_size_cap = IterativeDBSCAN(
        metric="euclidean", min_samples=3, smin=-1.0, eps=4.0, max_cluster_size=35
    )
    with_size_cap.fit(X)
    sizes_with_cap = sorted([(with_size_cap.labels_ == lbl).sum() for lbl in np.unique(with_size_cap.labels_) if lbl != -1])
    assert with_size_cap.n_iterations_ >= 1, "should have triggered at least one split"
    assert max(sizes_with_cap) < 40, f"expected no cluster >= 40, got sizes {sizes_with_cap}"


def test_max_cluster_size_default_is_no_size_trigger():
    """When max_cluster_size is None, the algorithm uses only the silhouette trigger."""
    rng = np.random.RandomState(0)
    X = rng.randn(50, 2)
    idb = IterativeDBSCAN(metric="euclidean", min_samples=3, smin=-1.0, max_cluster_size=None)
    idb.fit(X)
    # smin=-1 disables sil-trigger; max_cluster_size=None disables size-trigger; so no splits
    assert idb.n_iterations_ == 0
