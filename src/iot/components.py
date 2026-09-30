"""
CN: 支撑图 S 的连通分量与必需的连接边集合 R（spec 第 3 节）。
EN: Connected components of the support graph S and the mandatory connector edge set R
    (spec section 3).

CN: 若 B = S 而没有 R，则 master 中 delta 无上界、分量平移无界；因此 R 必须保留。
EN: Without R the master has no bound on delta and componentwise shifts of (f, g) are
    unbounded, so R must never be pruned.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .edges import EdgeSet, edge_keys
from .problem import InverseOTError, InverseOTProblem


class _UnionFind:
    """
    CN: 带路径压缩的并查集，节点数 m + n。
    EN: Union-find with path compression over m + n nodes.
    """

    def __init__(self, size: int) -> None:
        self._parent = np.arange(int(size), dtype=np.int64)

    def find(self, node: int) -> int:
        parent = self._parent
        root = int(node)
        while parent[root] != root:
            root = int(parent[root])
        while parent[node] != root:
            parent[node], node = root, int(parent[node])
        return root

    def union(self, left: int, right: int) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root != right_root:
            self._parent[right_root] = left_root


@dataclass(frozen=True)
class ComponentPartition:
    """
    CN: 支撑图的连通分量划分及每个分量的代表源点/目标点。
    EN: Partition of the support graph into connected components with a representative
        source and target per component.
    """

    labels: np.ndarray
    n_components: int
    component_sources: tuple[int, ...]
    component_targets: tuple[int, ...]

    def __post_init__(self) -> None:
        labels = np.ascontiguousarray(np.asarray(self.labels, dtype=np.int64).reshape(-1))
        if labels.size < 1:
            raise ValueError("labels must be non-empty")
        if len(self.component_sources) != int(self.n_components):
            raise ValueError("component_sources must have one entry per component")
        if len(self.component_targets) != int(self.n_components):
            raise ValueError("component_targets must have one entry per component")
        object.__setattr__(self, "labels", labels)


def support_components(problem: InverseOTProblem) -> ComponentPartition:
    """
    CN: 用并查集计算以 S 为边的二分图连通分量。
    EN: Compute connected components of the bipartite graph with edges S via union-find.
    """
    n_source = int(problem.n_source)
    n_target = int(problem.n_target)
    rows, cols, _values = problem.support()
    union_find = _UnionFind(n_source + n_target)
    for row, col in zip(rows.tolist(), cols.tolist(), strict=True):
        union_find.union(int(row), n_source + int(col))

    roots = np.asarray([union_find.find(node) for node in range(n_source + n_target)], dtype=np.int64)
    unique_roots, labels = np.unique(roots, return_inverse=True)
    n_components = int(unique_roots.size)

    component_sources = [-1] * n_components
    component_targets = [-1] * n_components
    for source in range(n_source):
        component = int(labels[source])
        if component_sources[component] < 0:
            component_sources[component] = source
    for target in range(n_target):
        component = int(labels[n_source + target])
        if component_targets[component] < 0:
            component_targets[component] = target
    if any(value < 0 for value in component_sources) or any(value < 0 for value in component_targets):
        raise InverseOTError("every support component must contain at least one source and one target")
    return ComponentPartition(
        labels=labels,
        n_components=n_components,
        component_sources=tuple(int(value) for value in component_sources),
        component_targets=tuple(int(value) for value in component_targets),
    )


def _first_edge_outside(problem: InverseOTProblem, support: EdgeSet) -> tuple[int, int]:
    """
    CN: 返回行主序下第一条不在 S 中的边；不存在时抛错。
    EN: Return the first edge outside S in row-major order; raise when none exists.
    """
    n_target = int(problem.n_target)
    columns = np.arange(n_target, dtype=np.int64)
    for row in range(int(problem.n_source)):
        candidate_keys = edge_keys(np.full(n_target, row, dtype=np.int64), columns, n_target=n_target)
        missing = ~support.contains(candidate_keys)
        if bool(missing.any()):
            col = int(columns[int(np.argmax(missing))])
            return row, col
    raise InverseOTError(
        "support S is the full edge set, so the master problem is unbounded; "
        "the constraint-generation guarantee does not apply here"
    )


def connector_edges(problem: InverseOTProblem, partition: ComponentPartition) -> EdgeSet:
    """
    CN: 构造连接边集合 R：p >= 2 时连接分量环，p = 1 时取任一条 S 外边。
    EN: Build the connector set R: link components in a cycle when p >= 2, or take one
        edge outside S when p = 1.
    """
    n_target = int(problem.n_target)
    if int(partition.n_components) == 1:
        row, col = _first_edge_outside(problem, EdgeSet.from_arrays(*problem.support()[:2], n_target=n_target))
        return EdgeSet.from_arrays(np.array([row]), np.array([col]), n_target=n_target)
    rows = []
    cols = []
    p = int(partition.n_components)
    for index in range(p):
        rows.append(partition.component_sources[index])
        cols.append(partition.component_targets[(index + 1) % p])
    return EdgeSet.from_arrays(np.asarray(rows, dtype=np.int64), np.asarray(cols, dtype=np.int64), n_target=n_target)


def mandatory_edges(problem: InverseOTProblem) -> tuple[EdgeSet, ComponentPartition]:
    """
    CN: 返回 B = S ∪ R 及分量划分；B 在循环中永不被剪枝。
    EN: Return B = S ∪ R and the component partition; B is never pruned in the loop.
    """
    support_rows, support_cols, _values = problem.support()
    support = EdgeSet.from_arrays(support_rows, support_cols, n_target=int(problem.n_target))
    partition = support_components(problem)
    connectors = connector_edges(problem, partition)
    return support.union(connectors), partition


__all__ = [
    "ComponentPartition",
    "connector_edges",
    "mandatory_edges",
    "support_components",
]
