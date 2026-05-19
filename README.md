# iterative-dbscan

Silhouette-guided recursive refinement of DBSCAN. A modern, pip-installable
implementation of the **LENTA** algorithm from:

> Morichetta & Mellia, *LENTA: Longitudinal Exploration for Network Traffic Analysis from Passive Data*,
> IEEE Transactions on Network and Service Management, 2019.
> [DOI](https://ieeexplore.ieee.org/document/8493073)

## What problem does it solve?

Standard DBSCAN with a single `eps` finds a partition that is appropriate at
that one density scale. When the data has clusters at multiple density scales,
or when one cluster is internally heterogeneous, you get either:

- one giant blob that should be several clusters, or
- everything fragmented into noise.

`IterativeDBSCAN` runs DBSCAN, then for any cluster whose **mean silhouette
falls below `smin`** it extracts that cluster's local distance matrix, picks a
new eps from its own k-distance graph, and re-clusters within. Repeat until
every cluster is tight enough, or `max_depth` is reached.

## Install

```bash
pip install iterative-dbscan
```

For development:

```bash
git clone https://github.com/AndreaMorichetta/iterative-dbscan
cd iterative-dbscan
pip install -e ".[dev]"
pytest
```

## Quick start

### From feature vectors

```python
import numpy as np
from iterative_dbscan import IterativeDBSCAN

X = np.random.RandomState(0).randn(200, 10)

idb = IterativeDBSCAN(
    metric="euclidean",
    min_samples=5,
    smin=0.1,                  # split clusters whose silhouette < 0.1
    percentile_to_cluster=65,  # auto-eps percentile (LENTA default)
    max_depth=10,
)
labels = idb.fit_predict(X)
print(idb.cluster_tree_)        # readable hierarchy of splits
print(idb.silhouette_per_cluster_)
```

### From a precomputed distance matrix

```python
from sklearn.metrics import pairwise_distances
D = pairwise_distances(X, metric="cosine")
idb = IterativeDBSCAN(metric="precomputed", min_samples=5, smin=0.1)
labels = idb.fit_predict(D)
```


## Parameters

The parameters follow the ones used in classic DBSCAN. The main differences are:
- `smin`: by changing this parameter you decide which threshold of [silhouette score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.silhouette_score.html) would trigger a split. 
- `percentile_to_cluster`: important to extract the epsilon value from the k-distance graph.
- `max_depth`: the maximun number of interactions
- `noisify_unsplittable`: it is a boolean that let you decide to directly put low-silhouette clusters in noise.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `metric` | `"euclidean"` | Distance metric, or `"precomputed"` to pass a distance matrix. |
| `min_samples` | 5 | DBSCAN min_samples (also k for the auto-eps k-distance graph). |
| `smin` | 0.0 | Silhouette threshold. Clusters below this get split further. |
| `percentile_to_cluster` | 65.0 | Auto-eps percentile (0-100, exclusive). |
| `max_depth` | 10 | Maximum recursion depth per cluster lineage. |
| `eps` | None | Override the top-level auto-eps if set. |
| `noisify_unsplittable` | False | If True, low-silhouette clusters that can't be split become noise. |

## Output attributes (after `fit`)

- `labels_`: final cluster labels, -1 for noise
- `cluster_tree_`: `ClusterTree` with parent-child relationships and per-node silhouette/size/depth
- `silhouette_per_cluster_`: dict mapping each final label to its mean silhouette
- `n_iterations_`: total number of successful splits
- `eps_initial_`: the eps used at the top level

## Notes
This package is a refactor of
[AndreaMorichetta/compute_clustering](https://github.com/AndreaMorichetta/compute_clustering)
that fixes import-time issues with modern `scikit-learn` and `numpy`, and makes the algorithm safe to use as a library. It started from my need to reuse I-DBSCAN for future projects. The main differences are:

- I dropped `pyclustering` (abandoned) and external `hdbscan`/`OPTICS` dependencies. 
    - You can use sklearn's `HDBSCAN` directly if you want it.
- Supports raw feature vectors directly, not only precomputed distance matrices.
- Returns a `ClusterTree` that records the cluster-parents branch.
- Made it more robust adding tests, and a better packaging (`pyproject.toml`, `src/` layout).

## License

MIT.
