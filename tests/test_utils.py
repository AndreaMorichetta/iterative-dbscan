"""Tests for silhouette and eps utilities."""

import numpy as np
import pytest
from sklearn.metrics import pairwise_distances

from iterative_dbscan.eps import auto_eps, kth_neighbour_distances
from iterative_dbscan.silhouette import (
    mean_silhouette_per_cluster,
    silhouette_samples_with_noise,
)


def make_two_blobs(seed: int = 0):
    rng = np.random.RandomState(seed)
    A = rng.randn(30, 2) + np.array([0, 0])
    B = rng.randn(30, 2) + np.array([10, 10])
    X = np.vstack([A, B])
    labels = np.array([0] * 30 + [1] * 30)
    return X, labels


class TestKthNeighbourDistances:
    def test_simple_2d_case(self):
        X = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [10.0, 10.0]])
        D = pairwise_distances(X)
        k1 = kth_neighbour_distances(D, k=1)
        # nearest non-self distances: 1.0, 1.0, 1.0, sqrt(82)
        assert np.isclose(k1[0], 1.0)
        assert np.isclose(k1[1], 1.0)
        assert np.isclose(k1[2], 1.0)
        assert k1[3] > 10.0

    def test_diagonal_ignored(self):
        D = np.array([[0.0, 5.0], [5.0, 0.0]])
        k1 = kth_neighbour_distances(D, k=1)
        assert np.allclose(k1, [5.0, 5.0])

    def test_k_out_of_range(self):
        D = np.zeros((3, 3))
        with pytest.raises(ValueError):
            kth_neighbour_distances(D, k=3)
        with pytest.raises(ValueError):
            kth_neighbour_distances(D, k=0)


class TestAutoEps:
    def test_returns_positive_value_on_real_data(self):
        X, _ = make_two_blobs()
        D = pairwise_distances(X)
        eps = auto_eps(D, min_samples=3, percentile_to_cluster=65)
        assert eps > 0

    def test_higher_percentile_gives_larger_eps(self):
        X, _ = make_two_blobs()
        D = pairwise_distances(X)
        eps_low = auto_eps(D, min_samples=3, percentile_to_cluster=30)
        eps_high = auto_eps(D, min_samples=3, percentile_to_cluster=90)
        assert eps_high >= eps_low

    def test_invalid_percentile(self):
        D = pairwise_distances(np.random.randn(10, 2))
        with pytest.raises(ValueError):
            auto_eps(D, min_samples=3, percentile_to_cluster=0)
        with pytest.raises(ValueError):
            auto_eps(D, min_samples=3, percentile_to_cluster=100)


class TestSilhouette:
    def test_well_separated_clusters_have_high_silhouette(self):
        X, labels = make_two_blobs()
        D = pairwise_distances(X)
        sils = mean_silhouette_per_cluster(D, labels)
        for sil in sils.values():
            assert sil > 0.5

    def test_noise_excluded(self):
        X, labels = make_two_blobs()
        # mark some points as noise
        labels = labels.copy()
        labels[0:5] = -1
        D = pairwise_distances(X)
        sils = mean_silhouette_per_cluster(D, labels)
        assert -1 not in sils
        # noise samples themselves should get silhouette 0
        sample_sil = silhouette_samples_with_noise(D, labels)
        assert np.all(sample_sil[labels == -1] == 0.0)

    def test_single_cluster_returns_empty_or_zero(self):
        X, _ = make_two_blobs()
        D = pairwise_distances(X)
        labels = np.zeros(60, dtype=int)
        # only one real cluster -> silhouette undefined, should return zeros
        sils = silhouette_samples_with_noise(D, labels)
        assert np.all(sils == 0.0)

    def test_shape_mismatch_raises(self):
        D = np.zeros((5, 5))
        with pytest.raises(ValueError):
            silhouette_samples_with_noise(D, np.zeros(4, dtype=int))
        with pytest.raises(ValueError):
            silhouette_samples_with_noise(np.zeros((5, 4)), np.zeros(5, dtype=int))
