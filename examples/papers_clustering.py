"""End-to-end example: cluster synthetic 'paper embeddings' that mirror
the ERC sources structure — a few topical clusters, one of which is
heterogeneous and should get split iteratively.
"""

import numpy as np

from iterative_dbscan import IterativeDBSCAN


def make_paper_embeddings(seed: int = 42) -> np.ndarray:
    """Three topic clusters in 384-D 'embedding space':
    - 30 papers on topic A (tight)
    - 30 papers on topic B (tight)
    - 60 papers nominally on topic C but actually two sub-topics C1 and C2
      that are close in embedding space but separable
    """
    rng = np.random.RandomState(seed)
    d = 384

    def cluster(center: np.ndarray, n: int, spread: float) -> np.ndarray:
        pts = center + spread * rng.randn(n, d)
        pts /= np.linalg.norm(pts, axis=1, keepdims=True)
        return pts

    center_A = rng.randn(d); center_A /= np.linalg.norm(center_A)
    center_B = rng.randn(d); center_B /= np.linalg.norm(center_B)
    center_C1 = rng.randn(d); center_C1 /= np.linalg.norm(center_C1)
    # C2 is close to C1 in cosine space (only 0.3 displacement)
    center_C2 = center_C1 + 0.3 * rng.randn(d)
    center_C2 /= np.linalg.norm(center_C2)

    A = cluster(center_A, 30, 0.05)
    B = cluster(center_B, 30, 0.05)
    C1 = cluster(center_C1, 30, 0.08)
    C2 = cluster(center_C2, 30, 0.08)
    return np.vstack([A, B, C1, C2])


def main() -> None:
    X = make_paper_embeddings()
    print(f"Generated {X.shape[0]} 'paper embeddings' in {X.shape[1]}-D\n")

    print("=" * 60)
    print("Plain DBSCAN (single eps) via IterativeDBSCAN with smin=-1 (no splitting):")
    print("=" * 60)
    plain = IterativeDBSCAN(metric="cosine", min_samples=3, smin=-1.0)
    plain.fit_predict(X)
    _summarise(plain)

    print("\n" + "=" * 60)
    print("IterativeDBSCAN with smin=0.15 (split low-silhouette clusters):")
    print("=" * 60)
    idb = IterativeDBSCAN(metric="cosine", min_samples=3, smin=0.15, max_depth=5)
    idb.fit_predict(X)
    _summarise(idb)

    print("\nCluster hierarchy:")
    print(idb.cluster_tree_)


def _summarise(idb: IterativeDBSCAN) -> None:
    labels = idb.labels_
    assert labels is not None
    assert idb.silhouette_per_cluster_ is not None
    unique = [lbl for lbl in np.unique(labels) if lbl != -1]
    print(f"  eps_initial={idb.eps_initial_:.4f}, n_iterations={idb.n_iterations_}")
    print(f"  n_clusters={len(unique)}, n_noise={(labels == -1).sum()}")
    for lbl in unique:
        size = (labels == lbl).sum()
        sil = idb.silhouette_per_cluster_.get(int(lbl), float("nan"))
        print(f"  cluster {lbl}: n={size}, silhouette={sil:.3f}")


if __name__ == "__main__":
    main()
