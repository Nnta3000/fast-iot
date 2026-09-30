"""
CN: 稀疏边集合及其编码，供 master 与约束生成循环使用。
EN: Sparse edge sets and their key encoding, shared by the master and the
    constraint-generation loop.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def edge_keys(rows: np.ndarray, cols: np.ndarray, *, n_target: int) -> np.ndarray:
    """
    CN: 将边 (i, j) 编码为唯一键 `i * n_target + j`。
    EN: Encode edge (i, j) as the unique key `i * n_target + j`.
    """
    row_index = np.asarray(rows, dtype=np.int64).reshape(-1)
    col_index = np.asarray(cols, dtype=np.int64).reshape(-1)
    if row_index.size != col_index.size:
        raise ValueError("rows and cols must have the same length")
    return row_index * int(n_target) + col_index


@dataclass(frozen=True)
class EdgeSet:
    """
    CN: 去重、按键升序排列的边集合；不可变，所有集合运算返回新对象。
    EN: Deduplicated edge set sorted by key; immutable, every set operation returns a
        new object.
    """

    n_target: int
    rows: np.ndarray
    cols: np.ndarray
    keys: np.ndarray = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if int(self.n_target) < 1:
            raise ValueError("n_target must be >= 1")
        row_index = np.ascontiguousarray(np.asarray(self.rows, dtype=np.int64).reshape(-1))
        col_index = np.ascontiguousarray(np.asarray(self.cols, dtype=np.int64).reshape(-1))
        if row_index.size != col_index.size:
            raise ValueError("rows and cols must have the same length")
        if row_index.size and (row_index.min() < 0 or col_index.min() < 0):
            raise ValueError("edge indices must be nonnegative")
        if row_index.size:
            all_keys = row_index * int(self.n_target) + col_index
            unique_keys, first = np.unique(all_keys, return_index=True)
            row_index = np.ascontiguousarray(row_index[first])
            col_index = np.ascontiguousarray(col_index[first])
        else:
            unique_keys = np.zeros(0, dtype=np.int64)
        order = np.argsort(unique_keys, kind="stable")
        unique_keys = np.ascontiguousarray(unique_keys[order])
        object.__setattr__(self, "rows", np.ascontiguousarray(row_index[order]))
        object.__setattr__(self, "cols", np.ascontiguousarray(col_index[order]))
        object.__setattr__(self, "keys", unique_keys)

    @property
    def size(self) -> int:
        return int(self.keys.size)

    def to_arrays(self) -> tuple[np.ndarray, np.ndarray]:
        return self.rows, self.cols

    def contains(self, query_keys: np.ndarray) -> np.ndarray:
        """
        CN: 返回 query_keys 中每个键是否属于本集合。
        EN: Return, for each queried key, whether it belongs to this set.
        """
        keys = np.asarray(query_keys, dtype=np.int64).reshape(-1)
        if self.keys.size == 0:
            return np.zeros(keys.size, dtype=bool)
        position = np.searchsorted(self.keys, keys)
        clipped = np.minimum(position, self.keys.size - 1)
        return (position < self.keys.size) & (self.keys[clipped] == keys)

    def union(self, other: "EdgeSet") -> "EdgeSet":
        return EdgeSet(
            n_target=int(self.n_target),
            rows=np.concatenate([self.rows, other.rows]),
            cols=np.concatenate([self.cols, other.cols]),
        )

    def difference(self, other: "EdgeSet") -> "EdgeSet":
        keep = ~other.contains(self.keys)
        return EdgeSet(n_target=int(self.n_target), rows=self.rows[keep], cols=self.cols[keep])

    def select_mask(self, mask: np.ndarray) -> "EdgeSet":
        selected = np.asarray(mask, dtype=bool).reshape(-1)
        if selected.size != self.size:
            raise ValueError("mask must have one entry per edge")
        return EdgeSet(n_target=int(self.n_target), rows=self.rows[selected], cols=self.cols[selected])

    @classmethod
    def from_arrays(cls, rows: np.ndarray, cols: np.ndarray, *, n_target: int) -> "EdgeSet":
        return cls(n_target=int(n_target), rows=rows, cols=cols)


__all__ = [
    "EdgeSet",
    "edge_keys",
]
