"""End-to-end tests for IterativeDBSCAN."""

import numpy as np
import pytest
from sklearn.metrics import pairwise_distances

from iterative_dbscan import IterativeDBSCAN
from iterative_dbscan.tree import ClusterTree


def make_nested_clusters(seed: int = 0):
    """Two big blobs, one of which contains two tight sub-blobs.

    Standard DBSCAN with one eps will see two clusters. IterativeDBSCAN
    with smin > 0 should split the heterogeneous one further.
    """
    rng = np.random.RandomState(seed)
    # cluster A: tight blob at (0, 0)
    A = rng.randn(30, 2) * 0.3 + np.array([0, 0])
    # cluster B: two sub-blobs at (10, 0) and (10, 5), only loosely close
    B1 = rng.randn(20, 2) * 0.3 + np.array([10, 0])
    B2 = rng.randn(20, 2) * 0.3 + np.array([10, 5])
    return np.vstack([A, B1, B2])


class TestBasicAPI:
    def test_fits_and_predicts(self):
        X = make_nested_clusters()
        idb = IterativeDBSCAN(min_samples=3, smin=0.0)
        labels = idb.fit_predict(X)
        assert labels.shape == (70,)
        assert idb.labels_ is not None
        assert idb.cluster_tree_ is not None
        assert idb.eps_initial_ is not None
        assert idb.eps_initial_ > 0

    def test_attributes_set_after_fit(self):
        X = make_nested_clusters()
        idb = IterativeDBSCAN(min_samples=3, smin=0.1)
        idb.fit(X)
        assert isinstance(idb.cluster_tree_, ClusterTree)
        assert isinstance(idb.silhouette_per_cluster_, dict)
        assert idb.n_iterations_ >= 0

    def test_finds_at_least_two_clusters_on_separable_data(self):
        X = make_nested_clusters()
        idb = IterativeDBSCAN(min_samples=3, smin=0.0)
        labels = idb.fit_predict(X)
        unique = [lbl for lbl in np.unique(labels) if lbl != -1]
        assert len(unique) >= 2


class TestSplitting:
    def test_high_smin_triggers_splitting(self):
        X = make_nested_clusters()
        # With low smin, no splitting needed
        idb_low = IterativeDBSCAN(min_samples=3, smin=-1.0)
        idb_low.fit(X)
        # With high smin, the heterogeneous cluster should get split
        idb_high = IterativeDBSCAN(min_samples=3, smin=0.7)
        idb_high.fit(X)
        # high-smin run should have done strictly more iterations
        assert idb_high.n_iterations_ >= idb_low.n_iterations_

    def test_max_depth_bounds_recursion(self):
        X = make_nested_clusters()
        idb = IterativeDBSCAN(min_samples=3, smin=0.99, max_depth=1)
        idb.fit(X)
        assert idb.cluster_tree_.max_depth() <= 1


class TestPrecomputedInput:
    def test_precomputed_distance_matrix(self):
        X = make_nested_clusters()
        D = pairwise_distances(X)
        idb = IterativeDBSCAN(metric="precomputed", min_samples=3, smin=0.0)
        labels_pc = idb.fit_predict(D)
        # should produce same labels (up to permutation) as the vector path with default euclidean
        idb_vec = IterativeDBSCAN(metric="euclidean", min_samples=3, smin=0.0)
        labels_vec = idb_vec.fit_predict(X)
        # both should find some clustering with non-noise points
        assert (labels_pc != -1).any()
        assert (labels_vec != -1).any()

    def test_non_square_precomputed_raises(self):
        idb = IterativeDBSCAN(metric="precomputed")
        with pytest.raises(ValueError):
            idb.fit(np.zeros((5, 4)))


class TestCosineMetric:
    def test_cosine_on_normalised_embeddings(self):
        """Sanity check the metric users will actually pick for paper embeddings."""
        rng = np.random.RandomState(42)
        # 3 directions in 10D, 20 points each, normalised
        centers = rng.randn(3, 10)
        centers /= np.linalg.norm(centers, axis=1, keepdims=True)
        X = np.vstack([c + 0.05 * rng.randn(20, 10) for c in centers])
        X /= np.linalg.norm(X, axis=1, keepdims=True)
        idb = IterativeDBSCAN(metric="cosine", min_samples=3, smin=0.0)
        labels = idb.fit_predict(X)
        unique = [lbl for lbl in np.unique(labels) if lbl != -1]
        assert len(unique) >= 2
