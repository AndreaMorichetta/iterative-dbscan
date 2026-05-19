"""Cluster hierarchy tree.

Records the parent-child relationship between clusters as iterative splits happen.
Useful for inspection: which big cluster did this small one come from? At what
depth was the split performed?
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ClusterNode:
    label: int
    size: int
    silhouette: float | None = None
    depth: int = 0
    parent: int | None = None
    children: list[int] = field(default_factory=list)
    eps_used: float | None = None


class ClusterTree:
    """Tracks the iterative-split hierarchy of cluster labels.

    Labels are integers as produced by DBSCAN (-1 for noise, 0..k for clusters).
    Noise is not stored as a node; only real clusters.
    """

    def __init__(self) -> None:
        self.nodes: dict[int, ClusterNode] = {}

    def add(self, node: ClusterNode) -> None:
        self.nodes[node.label] = node
        if node.parent is not None and node.parent in self.nodes:
            parent = self.nodes[node.parent]
            if node.label not in parent.children:
                parent.children.append(node.label)

    def update_silhouette(self, label: int, silhouette: float) -> None:
        if label in self.nodes:
            self.nodes[label].silhouette = silhouette

    def roots(self) -> list[int]:
        return [n.label for n in self.nodes.values() if n.parent is None]

    def leaves(self) -> list[int]:
        return [n.label for n in self.nodes.values() if not n.children]

    def max_depth(self) -> int:
        if not self.nodes:
            return 0
        return max(n.depth for n in self.nodes.values())

    def __len__(self) -> int:
        return len(self.nodes)

    def __contains__(self, label: int) -> bool:
        return label in self.nodes

    def __repr__(self) -> str:
        lines = [f"ClusterTree(n_clusters={len(self.nodes)}, max_depth={self.max_depth()})"]

        def walk(label: int, prefix: str = "", is_last: bool = True) -> None:
            node = self.nodes[label]
            connector = "└── " if is_last else "├── "
            sil_str = f"sil={node.silhouette:.3f}" if node.silhouette is not None else "sil=?"
            lines.append(f"{prefix}{connector}cluster {label} (n={node.size}, {sil_str}, depth={node.depth})")
            child_prefix = prefix + ("    " if is_last else "│   ")
            children = node.children
            for i, child_label in enumerate(children):
                walk(child_label, child_prefix, i == len(children) - 1)

        roots = sorted(self.roots())
        for i, root in enumerate(roots):
            walk(root, "", i == len(roots) - 1)
        return "\n".join(lines)
