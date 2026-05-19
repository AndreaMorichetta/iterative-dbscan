"""IterativeDBSCAN: silhouette-guided recursive refinement of DBSCAN clusters.

The algorithm (originally introduced as LENTA-DBSCAN in Morichetta & Mellia,
IEEE TNSM 2019):

1. Run DBSCAN with auto-selected eps on the full data.
2. For each resulting cluster, compute its mean silhouette (with noise excluded).
3. If a cluster's silhouette is below ``smin`` and it has enough points, extract
   its sub-distance-matrix, pick a new eps for that local neighbourhood, and
   re-cluster within. Replace the old cluster with the new sub-clusters.
4. Repeat until no cluster needs splitting, or ``max_depth`` is reached.

The recursion is depth-bounded per cluster, so total work is bounded.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.cluster import DBSCAN
from sklearn.metrics import pairwise_distances

from .eps import auto_eps
from .silhouette import NOISE_LABEL, mean_silhouette_per_cluster
from .tree import ClusterNode, ClusterTree


@dataclass
class _SplitResult:
    """Internal: what happened when we tried to split a cluster."""
    success: bool
    new_labels: dict[int, int]  # old sub-label -> new global label
    eps_used: float | None


class IterativeDBSCAN(ClusterMixin, BaseEstimator):
    """Iterative DBSCAN with silhouette-guided splitting.

    Follows the scikit-learn estimator API: inherits from ``BaseEstimator``
    and ``ClusterMixin``, supports ``get_params`` / ``set_params``, and the
    ``fit`` / ``fit_predict`` contract.

    Parameters
    ----------
    metric : str, default 'euclidean'
        Distance metric. Use ``'precomputed'`` if you pass a distance matrix
        directly to ``fit``. ``'cosine'`` is recommended for normalised
        embeddings. Any metric accepted by ``sklearn.metrics.pairwise_distances``
        works.
    min_samples : int, default 5
        DBSCAN ``min_samples`` (also the k for the k-distance auto-eps heuristic).
    smin : float, default 0.0
        Silhouette threshold. Clusters with mean silhouette below ``smin`` are
        split further. Set to ``-1.0`` to disable splitting; set higher to be
        more aggressive about decomposing weak clusters.
    percentile_to_cluster : float, default 65.0
        Percentile used in auto-eps selection. Lower is stricter (more noise),
        higher is looser (more inclusion).
    max_depth : int, default 10
        Maximum recursion depth for any single cluster lineage.
    eps : float or None, default None
        If set, used as the initial eps for the top-level DBSCAN, overriding
        the auto-eps heuristic. Sub-clusters still use auto-eps locally.
    max_cluster_size : int or None, default None
        If set, also split any cluster larger than this, *regardless of its
        silhouette*. This is an extension beyond the original LENTA criterion,
        useful when a mega-cluster is internally heterogeneous but well
        isolated from other clusters (in which case silhouette stays high
        and the original criterion wouldn't trigger a split).
    noisify_unsplittable : bool, default False
        If a cluster has low silhouette but cannot be split (DBSCAN returns
        only one cluster on its sub-matrix), what to do? If ``True``, transforms
        its points to noise (the original LENTA behaviour). If ``False``,
        leave the cluster intact.

    Attributes
    ----------
    labels_ : ndarray of shape (n_samples,)
        Final cluster labels. -1 indicates noise. Set after ``fit``.
    cluster_tree_ : ClusterTree
        Hierarchy of splits. Set after ``fit``.
    silhouette_per_cluster_ : dict[int, float]
        Mean silhouette per final cluster. Set after ``fit``.
    n_iterations_ : int
        Number of successful splits performed. Set after ``fit``.
    eps_initial_ : float
        The eps used at the top level. Set after ``fit``.
    """

    def __init__(
        self,
        metric: str = "euclidean",
        min_samples: int = 5,
        smin: float = 0.0,
        percentile_to_cluster: float = 65.0,
        max_depth: int = 10,
        eps: float | None = None,
        max_cluster_size: int | None = None,
        noisify_unsplittable: bool = False,
    ) -> None:
        self.metric = metric
        self.min_samples = min_samples
        self.smin = smin
        self.percentile_to_cluster = percentile_to_cluster
        self.max_depth = max_depth
        self.eps = eps
        self.max_cluster_size = max_cluster_size
        self.noisify_unsplittable = noisify_unsplittable

    # ------------------------------------------------------------------ public

    def fit(self, X: ArrayLike, y: ArrayLike | None = None) -> "IterativeDBSCAN":
        """Run iterative DBSCAN on ``X``.

        ``X`` is either an (n, d) feature matrix (if ``metric != 'precomputed'``)
        or an (n, n) symmetric distance matrix (if ``metric == 'precomputed'``).

        The ``y`` parameter is accepted but ignored (sklearn convention for
        unsupervised estimators).
        """
        del y  # unsupervised; signature kept for sklearn compatibility
        D = self._to_distance_matrix(X)
        n = D.shape[0]

        # 1. initial clustering
        if self.eps is None:
            self.eps_initial_ = auto_eps(D, self.min_samples, self.percentile_to_cluster)
        else:
            self.eps_initial_ = float(self.eps)

        labels = DBSCAN(
            eps=self.eps_initial_,
            min_samples=self.min_samples,
            metric="precomputed",
        ).fit_predict(D).astype(np.int64)

        # 2. build initial tree
        tree = ClusterTree()
        for lbl in np.unique(labels):
            if lbl == NOISE_LABEL:
                continue
            tree.add(
                ClusterNode(
                    label=int(lbl),
                    size=int((labels == lbl).sum()),
                    depth=0,
                    parent=None,
                    eps_used=self.eps_initial_,
                )
            )
        sils = mean_silhouette_per_cluster(D, labels)
        for lbl, sil in sils.items():
            tree.update_silhouette(lbl, sil)

        # 3. iterative refinement (BFS over candidate splits)
        next_label = int(labels.max()) + 1 if (labels != NOISE_LABEL).any() else 0
        queue: list[int] = sorted(tree.nodes.keys())
        iterations = 0

        while queue:
            lbl = queue.pop(0)
            if lbl not in tree.nodes:
                continue
            node = tree.nodes[lbl]
            if node.depth >= self.max_depth:
                continue
            if node.size <= self.min_samples:
                continue

            sil = node.silhouette
            sil_triggers_split = sil is not None and sil < self.smin
            size_triggers_split = (
                self.max_cluster_size is not None and node.size > self.max_cluster_size
            )
            if not (sil_triggers_split or size_triggers_split):
                continue

            result, next_label = self._try_split(
                D=D,
                labels=labels,
                target_label=lbl,
                tree=tree,
                next_label=next_label,
            )
            if result.success:
                iterations += 1
                sils = mean_silhouette_per_cluster(D, labels)
                for new_lbl in result.new_labels.values():
                    if new_lbl in sils:
                        tree.update_silhouette(new_lbl, sils[new_lbl])
                        queue.append(new_lbl)

        # 4. final stats
        self.labels_ = labels
        self.cluster_tree_ = tree
        self.silhouette_per_cluster_ = mean_silhouette_per_cluster(D, labels)
        self.n_iterations_ = iterations
        return self

    # fit_predict is inherited from ClusterMixin and does exactly the same thing
    # as: self.fit(X); return self.labels_

    # ----------------------------------------------------------------- internal

    def _to_distance_matrix(self, X: ArrayLike) -> NDArray[np.float64]:
        X = np.asarray(X)
        if self.metric == "precomputed":
            if X.ndim != 2 or X.shape[0] != X.shape[1]:
                raise ValueError(
                    f"With metric='precomputed', X must be a square matrix; got shape {X.shape}"
                )
            return X.astype(np.float64, copy=False)
        if X.ndim != 2:
            raise ValueError(f"X must be 2D; got shape {X.shape}")
        return pairwise_distances(X, metric=self.metric).astype(np.float64, copy=False)

    def _try_split(
        self,
        *,
        D: NDArray[np.float64],
        labels: NDArray[np.int64],
        target_label: int,
        tree: ClusterTree,
        next_label: int,
    ) -> tuple[_SplitResult, int]:
        """Attempt to split the cluster ``target_label``. Mutates ``labels`` and ``tree`` on success."""
        mask = labels == target_label
        idx_global = np.where(mask)[0]
        D_sub = D[np.ix_(idx_global, idx_global)]

        try:
            sub_eps = auto_eps(D_sub, self.min_samples, self.percentile_to_cluster)
        except ValueError:
            return _SplitResult(False, {}, None), next_label

        sub_labels = DBSCAN(
            eps=sub_eps,
            min_samples=self.min_samples,
            metric="precomputed",
        ).fit_predict(D_sub)

        unique_sub_clusters = [lbl for lbl in np.unique(sub_labels) if lbl != NOISE_LABEL]
        if len(unique_sub_clusters) <= 1:
            # cannot meaningfully split
            if self.noisify_unsplittable:
                labels[mask] = NOISE_LABEL
                # remove from tree's effective state (keep node for traceability but mark)
                tree.nodes[target_label].silhouette = None  # ineligible to revisit
            return _SplitResult(False, {}, sub_eps), next_label

        # assign new global labels for each sub-cluster
        new_label_map: dict[int, int] = {}
        for sub_lbl in unique_sub_clusters:
            new_label_map[int(sub_lbl)] = next_label
            next_label += 1

        # rewrite labels for the sub-points
        for i, sub_lbl in enumerate(sub_labels):
            g = idx_global[i]
            if sub_lbl == NOISE_LABEL:
                labels[g] = NOISE_LABEL
            else:
                labels[g] = new_label_map[int(sub_lbl)]

        # add children to the tree
        parent_depth = tree.nodes[target_label].depth
        for sub_lbl, new_lbl in new_label_map.items():
            size = int((sub_labels == sub_lbl).sum())
            tree.add(
                ClusterNode(
                    label=new_lbl,
                    size=size,
                    depth=parent_depth + 1,
                    parent=target_label,
                    eps_used=sub_eps,
                )
            )
        # parent is now an internal node, no longer a leaf cluster
        tree.nodes[target_label].silhouette = None

        return _SplitResult(True, new_label_map, sub_eps), next_label
