"""
EN: **Diagnostic D2 -- does the body want different weights in different places?**

    Every result so far fits one cost for the whole shape. If hands, torso and feet want different
    descriptors, that is a hypothesis-class limitation, not an objective problem -- and it would explain why a
    global `theta` transfers so poorly across pairs (§4h) and under deformation (§4g).

    This module fits `theta` separately on `R` spatial regions of the same training pair and reports how far
    apart the regional thetas are. The regions are k-means clusters of the subsampled vertices in normalised
    coordinates; because the two meshes share the index set, one label field serves both shapes.

    Reading: compare the regional L1 distances with the **between-pair** distances already measured (0.5-1.5
    in §4h, where two plans made materially different demands). Regional thetas within ~0.1 of each other mean
    a regional cost has nothing to exploit.

    A secondary readout -- clearly preliminary -- is whether a *regional field* (row `i` uses its own region's
    theta) beats the global theta on held-out pairs. It is the shape of direction B, not the experiment: it is
    fitted and tested at one seed, with no regional supervised control, so a gain here would have to be
    re-established against `regional InfoNCE` before it means anything.

    CN: **诊断 D2**：把 θ 按空间分区分别拟合，比较各分区 θ 之间的 L1 距离（与 §4h 实测的跨对距离 0.5–1.5 对照）。
        另附一个初步读数：用"每行取本区 θ"的区域代价场是否优于全局 θ——这只是方向 B 的形状，不是实验本身。

    Run:
        .venv/bin/python -u -m iot_experiments.studies.real.faust.regional --n-sub 2000
"""

from __future__ import annotations

import dataclasses
import sys
import time
from pathlib import Path

import numpy as np
from jaxtyping import Float, Int

_REPO_ROOT = Path(__file__).resolve().parents[4]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from iot.core.features import SquaredDifferenceFeatures  # noqa: E402
from iot_experiments.harness.config import RunConfig, build_parser  # noqa: E402
from iot_experiments.harness.faust import (  # noqa: E402
    pair_geometry,
    read_ply,
    shared_index,
)
from iot_experiments.harness.faust_match import cost_matrix, geodesic_errors, match  # noqa: E402
from iot_experiments.harness.preflight import check as preflight_check  # noqa: E402
from iot_experiments.harness.records import RunLog, RunRecord, write_records  # noqa: E402
from iot_experiments.studies.real.faust.correspondence import default_pairs, fit_theta  # noqa: E402

DEFAULT_ROOT = "/home/agent/Documents/fast-iot/data/faust_extract/MPI-FAUST/training/registrations"

PROVENANCE: dict[str, object] = {
    "question": "do spatial regions of the body require different descriptor weights? (go/no-go for a "
                "piecewise-constant cost field)",
    "regions": "k-means (k-means++, 30 Lloyd iterations) on the standardised xyz of the subsampled training "
               "vertices; the same index set is used on both shapes, so one label field serves both",
    "thetas": "one simplex-only inverse-OT fit per region on that region's sub-block of the identity plan, "
              "plus one global fit for reference",
    "diagnostic": "pairwise L1 between the regional thetas, compared with the 0.5-1.5 L1 measured BETWEEN PAIRS "
                  "in section 4h; within ~0.1 means regions do not disagree",
    "secondary": "PRELIMINARY: the regional field (row i uses its region's theta) vs the global theta on 8 "
                 "held-out pairs. Single seed, no regional supervised control -- direction B, not the result",
    "memory": "one n x n cost and one n x n geodesic matrix; ~64 MB each at n=2000",
}


@dataclasses.dataclass(frozen=True)
class RegionalConfig(RunConfig):
    """CN: 旋钮。EN: knobs."""

    study: str = "faust_regional"
    section: str = "faust_regional"
    root: str = DEFAULT_ROOT
    train_pair: tuple[int, ...] = (0, 13)
    test_subjects: tuple[int, ...] = (5, 6, 7, 8, 9)
    test_pair_count: int = 8
    n_sub: int = 2000
    regions: int = 6
    max_iterations: int = 30

    @classmethod
    def add_study_arguments(cls, parser) -> None:
        """CN: 只加本研究特有的参数。EN: only this study's flags."""
        parser.add_argument("--root", default=DEFAULT_ROOT)
        parser.add_argument("--train-pair", type=int, nargs=2, default=[0, 13])
        parser.add_argument("--test-subjects", type=int, nargs="+", default=[5, 6, 7, 8, 9])
        parser.add_argument("--test-pair-count", type=int, default=8)
        parser.add_argument("--n-sub", type=int, default=2000)
        parser.add_argument("--regions", type=int, default=6)
        parser.add_argument("--max-iterations", type=int, default=30)


def kmeans(features: Float[np.ndarray, "n d"], count: int, *, iterations: int = 30, seed: int = 0
           ) -> Int[np.ndarray, "n"]:
    """
    EN: k-means++ initialisation then Lloyd iterations. Written here because no clustering library is
        installed; deterministic given `seed`.
    CN: k-means++ 初始化 + Lloyd 迭代；环境里没有聚类库，因此自带实现，给定 `seed` 时确定。
    """
    rng = np.random.default_rng(int(seed))
    n = int(features.shape[0])
    centres = [features[int(rng.integers(n))]]
    for _ in range(1, int(count)):
        dist = np.min(((features[:, None, :] - np.array(centres)[None, :, :]) ** 2).sum(-1), axis=1)
        total = float(dist.sum())
        probability = dist / total if total > 0 else np.full(n, 1.0 / n)
        centres.append(features[int(rng.choice(n, p=probability))])
    centre_array = np.array(centres, dtype=np.float64)
    labels = np.zeros(n, dtype=np.int64)
    for _ in range(int(iterations)):
        dist = ((features[:, None, :] - centre_array[None, :, :]) ** 2).sum(-1)
        labels = np.argmin(dist, axis=1).astype(np.int64)
        for c in range(int(count)):
            member = labels == c
            if member.any():
                centre_array[c] = features[member].mean(axis=0)
    return labels


def regional_cost(a: Float[np.ndarray, "n k"], b: Float[np.ndarray, "n k"],
                  thetas: list[Float[np.ndarray, "k"]], labels: Int[np.ndarray, "n"]
                  ) -> Float[np.ndarray, "n n"]:
    """
    EN: the piecewise-constant field: row `i` is scored with its own region's weights. Still a well-defined
        `n x n` matrix, so ordinary OT decodes it.
    CN: 分段常数的代价场：第 i 行用其所属区的权重。仍是一个良定义的 `n x n` 矩阵，可直接用普通 OT 解码。
    """
    out = np.empty((int(a.shape[0]), int(b.shape[0])), dtype=np.float64)
    for c, theta in enumerate(thetas):
        rows = np.where(labels == c)[0]
        if rows.size:
            out[rows] = cost_matrix(a[rows], b, theta)
    return out


def main(argv: list[str] | None = None) -> int:
    """CN: 入口。EN: entry point."""
    parser = build_parser(RegionalConfig)
    args = parser.parse_args(list(argv) if argv is not None else None)
    config = RegionalConfig.from_namespace(args, parser)
    run_dir = config.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    log = RunLog(run_dir / "run.log")
    pre = preflight_check(run_tiny=False, need_gpu=False)
    pre.write(run_dir / "preflight.json")
    log.line(f"preflight ok={pre.ok} mem={pre.mem_available_gb:.1f} GB")

    root = Path(config.root)
    train_pair = (int(config.train_pair[0]), int(config.train_pair[1]))
    test_pairs = default_pairs(tuple(int(v) for v in config.test_subjects), int(config.test_pair_count))
    a_tr = read_ply(root / f"tr_reg_{train_pair[0]:03d}.ply")
    b_tr = read_ply(root / f"tr_reg_{train_pair[1]:03d}.ply")
    index = shared_index(a_tr, int(config.n_sub), 0)
    a, b = pair_geometry(a_tr, b_tr, index)[:2]
    k = int(a.shape[1])
    log.line(f"train pair {train_pair}, n={index.size}, K={k}, regions={config.regions}")

    # CN: 分区用标准化后的 xyz（即描述子的前三列），因此与代价族的坐标一致。
    # EN: cluster on the standardised xyz (the first three descriptor columns), so the labels live in the
    #     same coordinate frame the cost uses.
    labels = kmeans(np.ascontiguousarray(a[:, :3]), int(config.regions), seed=0)
    sizes = np.bincount(labels, minlength=int(config.regions))
    log.line(f"region sizes: {sizes.tolist()}")

    started = time.perf_counter()
    global_fit = fit_theta(SquaredDifferenceFeatures(a, b), rel=None,
                           max_iterations=int(config.max_iterations))
    global_theta = np.asarray(global_fit["theta"], dtype=np.float64)
    log.line(f"global theta fitted in {time.perf_counter() - started:.1f}s")

    thetas: list[Float[np.ndarray, "k"]] = []
    for c in range(int(config.regions)):
        rows = np.where(labels == c)[0]
        started = time.perf_counter()
        fit = fit_theta(SquaredDifferenceFeatures(a[rows], b[rows]), rel=None,
                        max_iterations=int(config.max_iterations))
        thetas.append(np.asarray(fit["theta"], dtype=np.float64))
        log.line(f"   region {c}: n={rows.size:>5}  zeros={int(np.sum(thetas[-1] <= 1e-12)):>2}  "
                 f"L1-to-global={float(np.abs(thetas[-1] - global_theta).sum()):.3f}  "
                 f"({time.perf_counter() - started:.1f}s)")

    distances = np.zeros((int(config.regions), int(config.regions)), dtype=np.float64)
    for i in range(int(config.regions)):
        for j in range(int(config.regions)):
            distances[i, j] = float(np.abs(thetas[i] - thetas[j]).sum())
    upper = distances[np.triu_indices(int(config.regions), 1)]
    log.line(f"\npairwise regional L1: mean {upper.mean():.3f} | median {np.median(upper):.3f} | "
             f"max {upper.max():.3f} | min {upper.min():.3f}")
    log.line("per-region L1 to the global theta: "
             + ", ".join(f"{float(np.abs(t - global_theta).sum()):.3f}" for t in thetas))
    log.line("matrix:")
    for i in range(int(config.regions)):
        log.line("   " + " ".join(f"{distances[i, j]:6.3f}" for j in range(int(config.regions))))

    rows: list[RunRecord] = [
        RunRecord(study=str(config.study), geometry="positions", arm=f"region{c}", k=k, m=int(sizes[c]),
                  draw=0, metrics={"size": float(sizes[c]),
                                   "l1_to_global": float(np.abs(thetas[c] - global_theta).sum()),
                                   "zeros": float(np.sum(thetas[c] <= 1e-12)),
                                   "max": float(thetas[c].max())},
                  config=config.to_dict(), extra={"theta": thetas[c].tolist()})
        for c in range(int(config.regions))
    ]
    rows.append(RunRecord(study=str(config.study), geometry="spread", arm="regional_l1", k=k,
                          m=int(index.size), draw=0,
                          metrics={"mean": float(upper.mean()), "median": float(np.median(upper)),
                                   "max": float(upper.max()), "min": float(upper.min()),
                                   "mean_to_global": float(np.mean([np.abs(t - global_theta).sum()
                                                                    for t in thetas]))},
                          config=config.to_dict()))

    # CN: 初步读数：区域代价场 vs 全局 θ，8 个留出对，每对只算一次测地矩阵。
    # EN: preliminary readout -- regional field vs the global theta on the held-out pairs.
    log.line("\n--- preliminary: regional field vs global theta on held-out pairs ---")
    field_scores: list[float] = []
    global_scores: list[float] = []
    for pair in test_pairs:
        ta = read_ply(root / f"tr_reg_{pair[0]:03d}.ply")
        tb = read_ply(root / f"tr_reg_{pair[1]:03d}.ply")
        test_index = shared_index(ta, int(config.n_sub), 0)
        da, db, geo, norm, mirror = pair_geometry(ta, tb, test_index)
        global_pred = match(cost_matrix(da, db, global_theta))
        global_scores.append(float(geodesic_errors(global_pred, geo, norm, mirror)["miss_5pct"]))
        field_pred = match(regional_cost(da, db, thetas, labels))
        field_scores.append(float(geodesic_errors(field_pred, geo, norm, mirror)["miss_5pct"]))
        log.line(f"   {pair}: global {global_scores[-1]:.4f}  field {field_scores[-1]:.4f}")
        del geo
    delta = np.array(field_scores) - np.array(global_scores)
    log.line(f"field - global: mean {delta.mean():+.4f}, wins {int((delta < 0).sum())}/{delta.size}")
    for arm, values in (("global", global_scores), ("field", field_scores)):
        rows.append(RunRecord(study=str(config.study), geometry="test", arm=arm, k=k, m=int(index.size),
                              draw=0, metrics={"miss_5pct": float(np.mean(values)),
                                               "mean_pairs": float(np.mean(values))},
                              config=config.to_dict()))
    write_records(run_dir / "records.json", rows, provenance=PROVENANCE)
    log.line(f"wrote {run_dir / 'records.json'} ({len(rows)} rows)")
    log.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
