"""
CN: **池匹配研究（置信度）** —— 问题：耦合（EMD）选出的配对中，**哪些可信**？
    在给定几何与臂（cosine / learned）上，报告 precision-at-coverage，用两种置信度：

      (a) `mutual_nn`   —— 互最近邻一致性（行 argmin 与列 argmin 互相指向对方）；
      (b) `margin`      —— 行内/列内"最优与次优之差"的较小者。

    CN: **断言 `P@100% == matching hit`** —— 全覆盖时 precision 就是命中率，这条断言若失败说明
        计分索引错了（此前确实错过一次：用打乱后的列号而非 `argsort(perm)`）。

EN: **Pool-matching study (confidence)** -- which of the pairs the coupling selects can be trusted?
    Reports precision-at-coverage under two confidences: (a) `mutual_nn` (row argmin and column argmin
    agree), (b) `margin` (min of the row and column best-vs-second gap).

    **Asserts `P@100% == matching hit`**; a failure means the scoring index is wrong (this happened before:
    shuffled column ids were used instead of `argsort(perm)`).
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import ot
from jaxtyping import Bool, Float, Int

_REPO_ROOT = Path(__file__).resolve().parents[4]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from iot_experiments.harness.config import RunConfig, build_parser  # noqa: E402
from iot_experiments.harness.driver import make_split  # noqa: E402
from iot_experiments.harness.records import RunRecord, format_table, summarise, write_records  # noqa: E402
from iot_experiments.harness.registry import GeometryResult, geometry  # noqa: E402
from iot_experiments.harness.solver import (  # noqa: E402
    diagonal_cost,
    fit_full_lp,
    matching_hit,
    uniform_cosine,
)

COVERAGE: tuple[float, ...] = (0.25, 0.50, 0.75, 1.00)

PROVENANCE: dict[str, object] = {
    "standardisation_fit_set": "pool images only (transductive, label-free)",
    "pca_fit_set": "all T1+T2 images (label-free)",
    "note": "standardisation and PCA use the pool images (transductive, no labels)",
}


class PoolError(RuntimeError):
    """CN: 池匹配读数的前置条件失败。EN: a precondition for the pool readout failed."""


def mutual_nn(cost: Float[np.ndarray, "n n"]) -> Bool[np.ndarray, "n"]:
    """
    CN: 互最近邻：行 argmin 选 j，且列的 argmin 也回到 i。索引无关（不做打乱）。
    EN: mutual nearest neighbour: row argmin picks j and the column argmin returns to i (no shuffling).
    """
    values: Float[np.ndarray, "n n"] = np.asarray(cost, dtype=np.float64)
    row_best: Int[np.ndarray, "n"] = np.argmin(values, axis=1)
    col_best: Int[np.ndarray, "n"] = np.argmin(values, axis=0)
    return col_best[row_best] == np.arange(int(values.shape[0]))


def margin(cost: Float[np.ndarray, "n n"]) -> Float[np.ndarray, "n"]:
    """
    CN: 裕度 = `min(行内最优-次优, 列内最优-次优)`；越大越可信。
    EN: margin = `min(row best-vs-second, column best-vs-second)`; larger is more trustworthy.
    """
    values: Float[np.ndarray, "n n"] = np.asarray(cost, dtype=np.float64)
    row_sorted: Float[np.ndarray, "n n"] = np.sort(values, axis=1)
    col_sorted: Float[np.ndarray, "n n"] = np.sort(values, axis=0)
    row_gap: Float[np.ndarray, "n"] = row_sorted[:, 1] - row_sorted[:, 0]
    col_gap: Float[np.ndarray, "n"] = col_sorted[1, :] - col_sorted[0, :]
    return np.minimum(row_gap, col_gap)


def precision_at_coverage(hits: Bool[np.ndarray, "n"], score: Float[np.ndarray, "n"]
                          ) -> dict[str, float]:
    """
    CN: 按 `score` 降序（最可信在前）取前 25/50/75/100% 的平均命中。
        `P@100` 必须等于整体命中率，否则抛错。
    EN: sort by descending `score` (most trusted first) and average the hits over the top 25/50/75/100%.
        `P@100` must equal the overall hit rate, else raise.
    """
    order: Int[np.ndarray, "n"] = np.argsort(-np.asarray(score, dtype=np.float64), kind="stable")
    hits_f: Float[np.ndarray, "n"] = np.asarray(hits, dtype=np.float64)
    out: dict[str, float] = {}
    for frac in COVERAGE:
        k = max(1, int(round(frac * hits_f.size)))
        out[f"P@{int(round(frac * 100))}"] = float(np.mean(hits_f[order[:k]]))
    overall = float(np.mean(hits_f))
    if abs(out["P@100"] - overall) > 1e-12:
        raise PoolError(f"P@100 ({out['P@100']!r}) != overall hit ({overall!r}); scoring index is wrong")
    return out


def _tracked_hits(cost: Float[np.ndarray, "n n"], *, seed: int) -> Bool[np.ndarray, "n"]:
    """
    CN: 打乱列后的 EMD，返回**每行的命中布尔**（与 `matching_hit` 同一索引约定）。
    EN: EMD after shuffling columns; returns the **per-row hit mask** under the same index convention as
        `matching_hit`.
    """
    values: Float[np.ndarray, "n n"] = np.asarray(cost, dtype=np.float64)
    h = int(values.shape[0])
    if values.shape[1] != h:
        raise PoolError(f"expected a square matrix, got {values.shape}")
    perm: Int[np.ndarray, "n"] = np.random.default_rng(int(seed)).permutation(h)
    plan: Float[np.ndarray, "n n"] = ot.emd(np.ones(h) / h, np.ones(h) / h, values[:, perm],
                                            numItermax=10_000_000)
    return np.argmax(plan, axis=1) == np.argsort(perm)


@dataclass(frozen=True)
class PoolConfig(RunConfig):
    """CN: 池匹配研究自己的旋钮。EN: the pool study's own knobs."""

    section = "pool"
    study: str = "pool"
    feature: str = "dinov2-pool4"
    ks: tuple[int, ...] = (256,)
    ms: tuple[int, ...] = (100, 300)
    geometries: tuple[str, ...] = ("G2",)
    arms: tuple[str, ...] = ("cosine", "learned")

    @classmethod
    def add_study_arguments(cls, parser: argparse.ArgumentParser) -> None:
        """CN: 只加本研究特有的参数。EN: only this study's flags."""
        parser.add_argument("--feature", default="dinov2-pool4",
                            choices=["dinov2-pool4", "dinov2-cls", "block"])
        parser.add_argument("--ks", type=int, nargs="+", default=[256])
        parser.add_argument("--ms", type=int, nargs="+", default=[100, 300])
        parser.add_argument("--geometries", nargs="+", default=["G2"],
                            choices=["G0", "G1", "G2", "G3"])
        parser.add_argument("--arms", nargs="+", default=["cosine", "learned"],
                            choices=["cosine", "learned"])


def _evaluate(cost: Float[np.ndarray, "n n"], *, draw: int) -> dict[str, float]:
    """
    CN: 一个臂的完整读数：命中率、R@1/R@5/中位、两种置信度下的 P@coverage。
    EN: full read-out for one arm: hit, R@1/R@5/median, and P@coverage under both confidences.
    """
    hits = _tracked_hits(cost, seed=int(draw))
    hit = float(np.mean(hits))
    # CN: 与 `matching_hit` 交叉核对，保证两处索引约定一致。
    # EN: cross-check against `matching_hit` so both share one index convention.
    if abs(hit - matching_hit(cost, seed=int(draw))) > 1e-12:
        raise PoolError("per-row hits disagree with matching_hit; index convention diverged")
    out = {"hit": hit}
    out.update(precision_at_coverage(hits, margin(cost)))
    out.update({"mnn_" + k: v for k, v in precision_at_coverage(hits, mutual_nn(cost).astype(float)).items()})
    return out


def main(argv: list[str] | None = None) -> int:
    """CN: 入口。EN: entry point."""
    parser = build_parser(PoolConfig)
    args = parser.parse_args(list(argv) if argv is not None else None)
    config = PoolConfig.from_namespace(args, parser)

    from iot_experiments.harness.registry import features  # noqa: PLC0415

    records: list[RunRecord] = []
    for k in config.ks:
        x, y = features(str(config.feature), dim=int(k))
        n_subjects = int(x.shape[0])
        for m in config.ms:
            for draw in config.draws:
                split = make_split(n_subjects, int(m), draw=int(draw))
                for geom_name in config.geometries:
                    geo: GeometryResult = geometry(x, y, geom_name, labelled=split.labelled,
                                                   unlabelled=split.unlabelled)
                    costs: dict[str, Float[np.ndarray, "n n"]] = {}
                    if "cosine" in config.arms:
                        costs["cosine"] = uniform_cosine(geo.x_unlabelled, geo.y_unlabelled)
                    if "learned" in config.arms:
                        fit = fit_full_lp(geo.x_label, geo.y_label, solver=str(config.solver),
                                          solve_limit_s=float(config.solve_limit_s))
                        if fit.status != "ok":
                            print(f"  K={k} m={m} d={draw} {geom_name} learned EXCLUDED: {fit.status}",
                                  flush=True)
                            records.append(RunRecord(study=str(config.study), geometry=geom_name,
                                                     arm="learned", k=int(k), m=int(m),
                                                     draw=int(draw), metrics={},
                                                     config=config.to_dict(), excluded=True,
                                                     status=fit.status))
                            continue
                        costs["learned"] = diagonal_cost(geo.x_unlabelled, geo.y_unlabelled, fit.theta)
                    for arm, cost in costs.items():
                        metrics = _evaluate(cost, draw=int(draw))
                        records.append(RunRecord(study=str(config.study), geometry=geom_name,
                                                 arm=arm, k=int(k), m=int(m), draw=int(draw),
                                                 metrics=metrics, config=config.to_dict()))
                        print(f"  K={k} m={m} d={draw} {geom_name} {arm:>8} hit={metrics['hit']:.4f} "
                              f"P@25={metrics['P@25']:.4f} P@50={metrics['P@50']:.4f}", flush=True)

    rows = summarise(records, group=("geometry", "arm"),
                     metrics=("hit", "P@25", "P@50", "P@75", "P@100",
                              "mnn_P@25", "mnn_P@50", "mnn_P@75", "mnn_P@100"))
    columns = ("geometry", "arm", "n_draws", "hit_mean", "P@25_mean", "P@50_mean", "P@75_mean",
               "P@100_mean", "mnn_P@25_mean", "mnn_P@50_mean")
    print("\n" + format_table(rows, columns))
    print("\n# P@100_mean must equal hit_mean (asserted per record)")
    write_records(config.run_dir / "records.json", records, provenance=PROVENANCE)
    print(f"\n[pool] wrote {config.run_dir / 'records.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
