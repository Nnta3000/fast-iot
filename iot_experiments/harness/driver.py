"""
CN: **驱动** —— 把一个配置展开成若干 `RunRecord`。所有划分、循环与落盘都在这里，研究脚本不再自己写循环。

    CN: 划分约定（全树统一）：`order = rng(seed=draw).permutation(n_subjects)`，
        前 `m` 个为 labelled，其余为 unlabelled/pool。因此同一 `draw` 在任何研究里都指同一切分。

EN: **Driver** -- expand one configuration into `RunRecord`s. All splitting, looping and writing happen
    here; study scripts no longer write their own loops.

    Split convention (uniform across the tree): `order = rng(seed=draw).permutation(n_subjects)`, the first
    `m` are labelled and the rest form the unlabelled/pool set, so the same `draw` means the same split in
    every study.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from itertools import product
from typing import Callable, Sequence

import numpy as np
from jaxtyping import Float, Int

from .config import RunConfig
from .records import RunRecord, write_records
from .registry import geometry
from .solver import diagonal_cost, fit_full_lp, matching_hit, rank_metrics, uniform_cosine


class DriverError(RuntimeError):
    """CN: 驱动级错误（例如没有可跑的单元）。EN: driver-level error (e.g. no cells to run)."""


@dataclass(frozen=True)
class Split:
    """
    CN: 一次切分：`labelled` 是 `m` 个下标，`unlabelled` 是其余 `n` 个。
    EN: one split: `labelled` holds `m` indices, `unlabelled` the remaining `n`.
    """

    labelled: Int[np.ndarray, "m"]
    unlabelled: Int[np.ndarray, "n"]


def make_split(n_subjects: int, m: int, *, draw: int) -> Split:
    """
    CN: 按统一约定生成切分。EN: build a split under the shared convention.
    """
    if not 1 <= int(m) < int(n_subjects):
        raise DriverError(f"m must be in [1, n_subjects); got m={m}, n_subjects={n_subjects}")
    order: Int[np.ndarray, "subjects"] = np.random.default_rng(int(draw)).permutation(int(n_subjects))
    return Split(labelled=order[: int(m)], unlabelled=order[int(m):])


def score_arms(x: Float[np.ndarray, "subjects d"], y: Float[np.ndarray, "subjects d"], *,
               geometry_name: str, split: Split, arms: Sequence[str], draw: int,
               config: RunConfig, box_relative: float = 0.25) -> list[RunRecord]:
    """
    CN: 在一个几何与一次切分上评估若干臂，返回**未聚合**的记录。
        支持的臂：`cosine`（= uniform θ，registry 已断言一致）与 `learned`（完整 LP）。
        失败/超时的 learned 记为 `excluded=True`，**不**回退成 uniform。
    EN: Evaluate several arms on one geometry and one split; returns un-aggregated records. Supported
        arms: `cosine` (= uniform theta, asserted consistent by the registry) and `learned` (full LP).
        A failed/timed-out learned fit is recorded with `excluded=True`; it is **not** replaced by uniform.
    """
    geo = geometry(x, y, geometry_name, labelled=split.labelled, unlabelled=split.unlabelled)
    xu, yu = geo.x_unlabelled, geo.y_unlabelled
    records: list[RunRecord] = []
    snapshot = config.to_dict()

    if "cosine" in arms:
        c_cos = uniform_cosine(xu, yu)
        metrics = {"hit": matching_hit(c_cos, seed=int(draw)), **rank_metrics(c_cos)}
        records.append(RunRecord(study=str(config.study), geometry=str(geometry_name), arm="cosine",
                                 k=int(x.shape[1]), m=int(split.labelled.size), draw=int(draw),
                                 metrics=metrics, config=snapshot))

    if "learned" in arms:
        fit = fit_full_lp(geo.x_label, geo.y_label, solver=str(config.solver),
                          solve_limit_s=float(config.solve_limit_s), box_relative=float(box_relative))
        excluded = fit.status != "ok"
        if excluded:
            metrics: dict[str, float] = {}
        else:
            c_lrn = diagonal_cost(xu, yu, fit.theta)
            metrics = {"hit": matching_hit(c_lrn, seed=int(draw)), **rank_metrics(c_lrn)}
        records.append(RunRecord(study=str(config.study), geometry=str(geometry_name),
                                 arm="learned", k=int(x.shape[1]), m=int(split.labelled.size),
                                 draw=int(draw), metrics=metrics, config=snapshot, excluded=excluded,
                                 status=fit.status,
                                 extra={"R_prime": fit.R_prime, "R_uniform": fit.R_uniform,
                                        "R_ratio": fit.R_ratio, "nnz": fit.nnz,
                                        "build_s": fit.build_s, "solve_s": fit.solve_s}))
    return records


def run_cells(features_by_k: Callable[[int], tuple[Float[np.ndarray, "subjects d"],
                                                   Float[np.ndarray, "subjects d"]]],
              *, config: RunConfig, ks: Sequence[int], ms: Sequence[int],
              geometries: Sequence[str], arms: Sequence[str],
              provenance: dict[str, object] | None = None) -> list[RunRecord]:
    """
    CN: 遍历 `K x m x draw x geometry` 网格。`features_by_k(K)` 返回未后处理的 `(T1, T2)`。
        `draw` 真正参与切分（`make_split(..., draw=draw)`），并写入每条记录。
    EN: Loop the `K x m x draw x geometry` grid. `features_by_k(K)` returns raw `(T1, T2)` arrays; `draw`
        drives the split via `make_split` and is stored on every record.
    """
    if not ks or not ms:
        raise DriverError("ks and ms must be non-empty")
    records: list[RunRecord] = []
    t_all = time.perf_counter()
    for k in ks:
        x, y = features_by_k(int(k))
        n_subjects = int(x.shape[0])
        for m, geom in product(ms, geometries):
            for draw in config.draws:
                split = make_split(n_subjects, int(m), draw=int(draw))
                for rec in score_arms(x, y, geometry_name=str(geom), split=split, arms=arms,
                                      draw=int(draw), config=config):
                    records.append(rec)
                    print(f"  K={k:>3} m={m:>3} draw={draw} {geom:>3} {rec.arm:>8} "
                          f"hit={rec.metrics.get('hit', float('nan')):.4f} "
                          f"R@1={rec.metrics.get('R@1', float('nan')):.4f}"
                          f"{'' if not rec.excluded else '  EXCLUDED: ' + rec.status}", flush=True)
    payload = dict(provenance or {})
    payload["config"] = config.to_dict()
    payload["wall_s"] = round(time.perf_counter() - t_all, 1)
    write_records(config.run_dir / "records.json", records, provenance=payload)
    return records


__all__ = ["DriverError", "Split", "make_split", "run_cells", "score_arms"]
