"""
EN: **Direction A -- select inside the near-optimal set, on validation pairs.**

    D1 (`face.py`) showed that thetas which are near-equally good on the *training* pair differ by 0.089
    `miss@5 %` on held-out pairs, and that the shipped max-margin point ranks 20th of 71. But D1's "best" was
    chosen on the **test** pairs, so it is a ceiling, not a method. This module runs the method:

      1. fit the reference (max-margin) theta on the training pair;
      2. sample the near-optimal set (exact regret within `slack_rel` of the reference) with the same walk;
      3. score **every** candidate on validation pairs drawn from the TRAINING subjects;
      4. take the validation-best theta to the held-out test pairs, against the reference, a selection-free
         aggregate (coordinate-wise median of the set) and uniform.

    Two questions, in order of importance:

      - **does validation selection beat the default?** (the practical claim)
      - **does a selection-free aggregate beat both?** (if the median is as good as the argmax, the spread is
        noise being amplified by the selection, not signal being harvested -- the same lesson as the averaged
        theta in §4e)

    The validation/test split is subject-disjoint and the test pairs take no part in the selection.

    CN: **方向 A**：在近最优集合内用**验证对**选 θ。D1 的"最好"是在测试对上选的（上界），本模块把它变成方法：
        训练对拟合参考 θ → 游走采样近最优集 → 在训练受试者的验证对上给每个候选打分 → 把验证最优带到留出对，
        并与参考 θ、无选择的中位数聚合、以及均匀权重比较。

    Run:
        .venv/bin/python -u -m iot_experiments.studies.real.faust.select --n-sub 4000
"""

from __future__ import annotations

import dataclasses
import sys
import time
from pathlib import Path

import numpy as np
from jaxtyping import Float

_REPO_ROOT = Path(__file__).resolve().parents[4]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from iot.core.features import SquaredDifferenceFeatures  # noqa: E402
from iot_experiments.harness.config import RunConfig, build_parser  # noqa: E402
from iot_experiments.harness.faust import pair_geometry, read_ply, shared_index  # noqa: E402
from iot_experiments.harness.faust_match import (  # noqa: E402
    bootstrap_contrast,
    duality_gap,
    map_metrics,
    near_optimal_walk,
)
from iot_experiments.harness.preflight import check as preflight_check  # noqa: E402
from iot_experiments.harness.records import RunLog, RunRecord, summarise, write_records  # noqa: E402
from iot_experiments.studies.real.faust.correspondence import default_pairs, fit_theta  # noqa: E402

DEFAULT_ROOT = "/home/agent/Documents/fast-iot/data/faust_extract/MPI-FAUST/training/registrations"

PROVENANCE: dict[str, object] = {
    "question": "does selecting inside the near-optimal set on VALIDATION pairs beat the shipped max-margin "
                "theta on held-out pairs, and does a selection-free aggregate beat both?",
    "candidates": "the reference plus every theta admitted by the random walk with exact regret within "
                  "slack_rel of the reference's (same sampler as D1 / face.py)",
    "selection": "mean miss@5% over 6 validation pairs drawn from the TRAINING subjects (0-4), which are "
                 "subject-disjoint from the test subjects (5-9); the test pairs take no part in it",
    "arms": "reference (shipped max-margin), selected (validation-best), median (coordinate-wise median of the "
            "admitted set, renormalised -- selection-free), uniform",
    "metric": "geodesic miss@5% on the 8 held-out pairs, paired bootstrap over pairs",
    "memory": "one n x n cost and one n x n geodesic matrix at a time; ~128 MB each at n=4000",
}


@dataclasses.dataclass(frozen=True)
class SelectConfig(RunConfig):
    """CN: 旋钮。EN: knobs."""

    study: str = "faust_select"
    section: str = "faust_select"
    root: str = DEFAULT_ROOT
    train_pair: tuple[int, ...] = (0, 13)
    train_subjects: tuple[int, ...] = (0, 1, 2, 3, 4)
    test_subjects: tuple[int, ...] = (5, 6, 7, 8, 9)
    test_pair_count: int = 8
    val_pair_count: int = 6
    n_sub: int = 4000
    proposals: int = 150
    slack_rel: float = 0.25
    step: float = 0.08
    max_iterations: int = 30

    @classmethod
    def add_study_arguments(cls, parser) -> None:
        """CN: 只加本研究特有的参数。EN: only this study's flags."""
        parser.add_argument("--root", default=DEFAULT_ROOT)
        parser.add_argument("--train-pair", type=int, nargs=2, default=[0, 13])
        parser.add_argument("--train-subjects", type=int, nargs="+", default=[0, 1, 2, 3, 4])
        parser.add_argument("--test-subjects", type=int, nargs="+", default=[5, 6, 7, 8, 9])
        parser.add_argument("--test-pair-count", type=int, default=8)
        parser.add_argument("--val-pair-count", type=int, default=6)
        parser.add_argument("--n-sub", type=int, default=4000)
        parser.add_argument("--proposals", type=int, default=150)
        parser.add_argument("--slack-rel", type=float, default=0.25)
        parser.add_argument("--step", type=float, default=0.08)
        parser.add_argument("--max-iterations", type=int, default=30)


def main(argv: list[str] | None = None) -> int:
    """CN: 入口。EN: entry point."""
    parser = build_parser(SelectConfig)
    args = parser.parse_args(list(argv) if argv is not None else None)
    config = SelectConfig.from_namespace(args, parser)
    run_dir = config.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    log = RunLog(run_dir / "run.log")
    pre = preflight_check(run_tiny=False, need_gpu=False)
    pre.write(run_dir / "preflight.json")
    log.line(f"preflight ok={pre.ok} mem={pre.mem_available_gb:.1f} GB")

    root = Path(config.root)
    train_pair = (int(config.train_pair[0]), int(config.train_pair[1]))
    test_pairs = default_pairs(tuple(int(v) for v in config.test_subjects), int(config.test_pair_count))
    pool = default_pairs(tuple(int(v) for v in config.train_subjects),
                         4 * int(config.val_pair_count) + 16)
    val_pairs = [pair for pair in pool if pair != train_pair and pair[::-1] != train_pair
                 ][: int(config.val_pair_count)]
    for pair in val_pairs:
        for vertex_id in pair:
            if vertex_id // 10 not in set(int(v) for v in config.train_subjects):
                raise SystemExit(f"validation pair {pair} leaves the training subjects")
    log.line(f"train pair {train_pair}; validation (training subjects) {val_pairs}; test {test_pairs}")

    a_tr = read_ply(root / f"tr_reg_{train_pair[0]:03d}.ply")
    b_tr = read_ply(root / f"tr_reg_{train_pair[1]:03d}.ply")
    index = shared_index(a_tr, int(config.n_sub), 0)
    a, b = pair_geometry(a_tr, b_tr, index)[:2]
    k = int(a.shape[1])

    started = time.perf_counter()
    fit = fit_theta(SquaredDifferenceFeatures(a, b), rel=None, max_iterations=int(config.max_iterations))
    if fit["theta"] is None:
        log.line("the reference fit failed; nothing to select from")
        log.close()
        return 1
    reference = np.asarray(fit["theta"], dtype=np.float64)
    reference_regret, reference_wrong = duality_gap(reference, a, b)
    tolerance = abs(reference_regret) * (1.0 + float(config.slack_rel))
    log.line(f"reference fitted in {time.perf_counter() - started:.1f}s; regret={reference_regret:.4g}, "
             f"training map wrong at {reference_wrong}/{index.size}; tolerance={tolerance:.4g}")

    admitted = near_optimal_walk(reference, a, b, tolerance=tolerance, proposals=int(config.proposals),
                                 step=float(config.step), seed=0)
    thetas = np.array(admitted, dtype=np.float64)
    log.line(f"admitted {len(admitted) - 1}/{int(config.proposals)} proposals -> {len(admitted)} candidates")
    median_theta = np.median(thetas, axis=0)
    median_theta = median_theta / median_theta.sum()

    # CN: **验证**：训练受试者的验证对，全部候选都打分。每对只算一次测地矩阵。
    # EN: **validation**: score every candidate on the training-subject validation pairs, one geodesic matrix
    #     per pair.
    log.line(f"\n--- validation on {len(val_pairs)} training-subject pairs ({len(admitted)} candidates) ---")
    val_scores = np.zeros((len(admitted), len(val_pairs)), dtype=np.float64)
    for slot, pair in enumerate(val_pairs):
        va = read_ply(root / f"tr_reg_{pair[0]:03d}.ply")
        vb = read_ply(root / f"tr_reg_{pair[1]:03d}.ply")
        v_index = shared_index(va, int(config.n_sub), 0)
        va_desc, vb_desc, geo, norm, mirror = pair_geometry(va, vb, v_index)
        for candidate, theta in enumerate(admitted):
            val_scores[candidate, slot] = map_metrics(va_desc, vb_desc, theta, geo, norm, mirror)["miss_5pct"]
        del geo
        log.line(f"   validation pair {pair}: scored {len(admitted)} candidates")
    val_mean = val_scores.mean(axis=1)
    best = int(np.argmin(val_mean))
    reference_rank = int(np.where(np.argsort(val_mean) == 0)[0][0]) + 1
    log.line(f"validation miss@5%: reference {val_mean[0]:.4f} (rank {reference_rank}/{len(admitted)}); "
             f"best candidate {val_mean[best]:.4f} (index {best}); "
             f"spread {val_mean.max() - val_mean.min():.4f}")
    selected = admitted[best]

    # CN: **测试**：只带四个臂去留出对，绝不在测试对上挑选。
    # EN: **test**: only four arms go to the held-out pairs; nothing is selected on them.
    log.line("\n--- evaluation on held-out pairs ---")
    arms: dict[str, Float[np.ndarray, "k"]] = {
        "reference": reference,
        "selected": selected,
        "median": np.asarray(median_theta, dtype=np.float64),
        "uniform": np.full(k, 1.0 / k),
    }
    rows: list[RunRecord] = []
    per_pair: dict[str, dict[str, float]] = {}
    for pair in test_pairs:
        ta = read_ply(root / f"tr_reg_{pair[0]:03d}.ply")
        tb = read_ply(root / f"tr_reg_{pair[1]:03d}.ply")
        geometry = f"pair{pair[0]:03d}-{pair[1]:03d}"
        t_index = shared_index(ta, int(config.n_sub), 0)
        da, db, geo, norm, mirror = pair_geometry(ta, tb, t_index)
        for name, theta in arms.items():
            metrics = map_metrics(da, db, theta, geo, norm, mirror)
            rows.append(RunRecord(study=str(config.study), geometry=geometry, arm=name, k=k,
                                  m=int(t_index.size), draw=0, metrics=metrics, config=config.to_dict()))
            per_pair.setdefault(geometry, {})[name] = metrics["miss_5pct"]
        log.line(f"   {geometry}: " + "  ".join(f"{name}={per_pair[geometry][name]:.4f}" for name in arms))
        del geo

    for base in ("selected", "median"):
        contrasts = bootstrap_contrast(per_pair, base=base, others=["reference", "uniform"])
        for other, stats in contrasts.items():
            log.line(f"   {base:<9} - {other:<9} mean {stats['mean']:+.4f}  "
                     f"95% CI [{stats['ci_low']:+.4f}, {stats['ci_high']:+.4f}]  "
                     f"wins {int(stats['wins'])}/{int(stats['pairs'])}")
            rows.append(RunRecord(study=str(config.study), geometry="contrast", arm=f"{base}_minus_{other}",
                                  k=k, m=int(index.size), draw=-1, metrics=stats, config=config.to_dict()))
    rows.append(RunRecord(study=str(config.study), geometry="selection", arm="validation", k=k,
                          m=int(index.size), draw=0,
                          metrics={"candidates": float(len(admitted)),
                                   "admitted_fraction": float((len(admitted) - 1) / int(config.proposals)),
                                   "val_reference": float(val_mean[0]),
                                   "val_best": float(val_mean[best]),
                                   "val_reference_rank": float(reference_rank),
                                   "val_spread": float(val_mean.max() - val_mean.min()),
                                   "reference_regret": float(reference_regret),
                                   "reference_wrong": float(reference_wrong)},
                          config=config.to_dict(),
                          extra={"selected_theta": selected.tolist(),
                                 "median_theta": median_theta.tolist(),
                                 "validation_scores": val_mean.tolist()}))
    table = summarise(rows, group=("arm",), metrics=("mean", "miss_5pct"))
    log.line("\n" + str(table))
    write_records(run_dir / "records.json", rows, provenance=PROVENANCE)
    log.line(f"wrote {run_dir / 'records.json'} ({len(rows)} rows)")
    log.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
