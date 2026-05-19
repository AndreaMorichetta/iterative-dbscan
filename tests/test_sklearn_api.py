"""Sklearn API compliance tests.

These verify that IterativeDBSCAN behaves like other sklearn clusterers:
- get_params / set_params work
- sklearn.base.clone() produces an unfitted copy with the same params
- It can be used in tools that expect the sklearn estimator API
"""

import numpy as np
import pytest
from sklearn.base import BaseEstimator, ClusterMixin, clone
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from iterative_dbscan import IterativeDBSCAN


def make_data(seed: int = 0):
    rng = np.random.RandomState(seed)
    A = rng.randn(20, 2) * 0.3 + np.array([0, 0])
    B = rng.randn(20, 2) * 0.3 + np.array([10, 10])
    return np.vstack([A, B])


class TestSklearnInheritance:
    def test_is_a_baseestimator(self):
        idb = IterativeDBSCAN()
        assert isinstance(idb, BaseEstimator)

    def test_is_a_clustermixin(self):
        idb = IterativeDBSCAN()
        assert isinstance(idb, ClusterMixin)


class TestGetSetParams:
    def test_get_params_returns_init_args(self):
        idb = IterativeDBSCAN(min_samples=7, smin=0.2, metric="cosine")
        params = idb.get_params()
        assert params["min_samples"] == 7
        assert params["smin"] == 0.2
        assert params["metric"] == "cosine"
        # all init params should be there
        expected_keys = {
            "metric", "min_samples", "smin", "percentile_to_cluster",
            "max_depth", "eps", "max_cluster_size", "demote_unsplittable",
        }
        assert set(params.keys()) == expected_keys

    def test_set_params_updates(self):
        idb = IterativeDBSCAN()
        idb.set_params(min_samples=10, smin=0.5)
        assert idb.min_samples == 10
        assert idb.smin == 0.5

    def test_set_params_returns_self_for_chaining(self):
        idb = IterativeDBSCAN()
        result = idb.set_params(min_samples=10)
        assert result is idb

    def test_set_params_rejects_unknown(self):
        idb = IterativeDBSCAN()
        with pytest.raises(ValueError):
            idb.set_params(not_a_real_param=42)


class TestClone:
    def test_clone_preserves_params(self):
        original = IterativeDBSCAN(min_samples=7, smin=0.2, metric="cosine")
        cloned = clone(original)
        assert cloned.get_params() == original.get_params()

    def test_clone_is_unfitted(self):
        """A cloned estimator must not carry over fitted state."""
        original = IterativeDBSCAN(min_samples=3)
        original.fit(make_data())
        cloned = clone(original)
        # cloned should not have labels_ attribute
        assert not hasattr(cloned, "labels_")

    def test_clone_can_be_fit_independently(self):
        original = IterativeDBSCAN(min_samples=3)
        original.fit(make_data(seed=0))
        cloned = clone(original)
        cloned.fit(make_data(seed=1))
        # both should be fitted now with potentially different labels
        assert hasattr(original, "labels_")
        assert hasattr(cloned, "labels_")


class TestFitPredictContract:
    def test_fit_returns_self(self):
        idb = IterativeDBSCAN(min_samples=3)
        result = idb.fit(make_data())
        assert result is idb

    def test_fit_predict_returns_labels(self):
        """fit_predict is inherited from ClusterMixin and should work."""
        idb = IterativeDBSCAN(min_samples=3)
        labels = idb.fit_predict(make_data())
        assert isinstance(labels, np.ndarray)
        assert labels.shape == (40,)

    def test_fit_accepts_y_none(self):
        """sklearn convention: unsupervised estimators must accept y=None."""
        idb = IterativeDBSCAN(min_samples=3)
        idb.fit(make_data(), y=None)  # should not raise
        assert hasattr(idb, "labels_")


class TestSklearnRepr:
    def test_repr_shows_non_default_params(self):
        """sklearn's __repr__ via BaseEstimator should show non-default params."""
        idb = IterativeDBSCAN(min_samples=7, smin=0.3)
        s = repr(idb)
        assert "IterativeDBSCAN" in s
        # at least one of the non-default values should appear
        assert "7" in s or "0.3" in s
