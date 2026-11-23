"""
EN: **Where the correspondence is wrong, on one held-out FAUST pair** -- per-vertex error for three weightings,
    written as a record plus the per-vertex error arrays a figure can draw.

    This exists to make one picture possible: a human surface with the wrongly matched vertices marked, so a
    reader sees *where* the cost fails instead of reading a miss rate. It is deliberately small (n = 600
    vertices for the fit and the evaluation) so it costs one CPU LP solve and one geodesic matrix, and it
    reuses every primitive in the harness rather than re-implementing the pipeline.

    The three weightings are fixed, no hyperparameter is chosen: `position_only` (xyz only), `uniform_theta`
    (equal weights on all 13 descriptors) and `learned` (the simplex fit on the training pair, HiGHS master).

    CN: **一个留出 FAUST 对上"错在哪里"** —— 三种加权下逐顶点误差，写成记录 + 供绘图使用的误差数组。
        规模刻意小（拟合与评估都用 n = 600 个顶点），代价是一次 CPU LP 与一个测地矩阵；全部复用 harness。
        三种加权固定：仅位置、等权、在训练对上拟合的 learned θ。

    Run:
        .venv/bin/python -u -m iot_experiments.studies.iot.real.faust.error_points
"""

from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

import numpy as np
from jaxtyping import Float

_REPO_ROOT = Path(__file__).resolve().parents[4]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from iot.core.features import SquaredDifferenceFeatures  # noqa: E402
from iot_experiments.harness.config import RunConfig, build_parser  # noqa: E402
from iot_experiments.harness.faust import (  # noqa: E402
    default_registrations_root,
    descriptors,
    pair_geometry,
    read_ply,
    shared_index,
)
from iot_experiments.harness.faust_match import cost_matrix, match  # noqa: E402
from iot_experiments.harness.records import RunLog, RunRecord, write_records  # noqa: E402
from iot_experiments.studies.iot.real.faust.correspondence import fit_theta, standardise  # noqa: E402

DEFAULT_ROOT = str(default_registrations_root())

PROVENANCE: dict[str, object] = {
    "question": "on a held-out FAUST pair, which vertices are matched wrongly, and what does the surface look "
                "like where they are?",
    "scale": "n = 600 vertices for both the fit and the evaluation; the images mark the wrongly matched "
             "vertices on the target surface, they do not score the whole mesh",
    "arms": "position_only (xyz only), uniform_theta (equal weights on 13 descriptors), learned (simplex fit "
            "on the training pair with the HiGHS master)",
    "error": "per-vertex geodesic distance between the matched target vertex and the true one, divided by "
             "sqrt(surface area) of the target",
    "why_highs": "the master is only fitted once and the question is where the errors are; the exact CPU "
                 "master keeps the fit reproducible to the last digit",
}


@dataclasses.dataclass(frozen=True)
class ErrorPointsConfig(RunConfig):
    """CN: 旋钮。EN: knobs."""

    study: str = "faust_error_points"
    section: str = "faust_error_points"
    root: str = DEFAULT_ROOT
    train_pair: tuple[int, ...] = (0, 13)
    pair: tuple[int, ...] = (50, 66)
    n_sub: int = 600
    max_iterations: int = 20
    threshold: float = 0.05

    @classmethod
    def add_study_arguments(cls, parser) -> None:
        """CN: 只加本研究特有的参数。EN: only this study's flags."""
        parser.add_argument("--root", default=DEFAULT_ROOT)
        parser.add_argument("--train-pair", type=int, nargs=2, default=[0, 13])
        parser.add_argument("--pair", type=int, nargs=2, default=[50, 66])
        parser.add_argument("--n-sub", type=int, default=600)
        parser.add_argument("--max-iterations", type=int, default=20)
        parser.add_argument("--threshold", type=float, default=0.05)


def main(argv: list[str] | None = None) -> int:
    """CN: 入口。EN: entry point."""
    parser = build_parser(ErrorPointsConfig)
    args = parser.parse_args(list(argv) if argv is not None else None)
    config = ErrorPointsConfig.from_namespace(args, parser)
    run_dir = config.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    log = RunLog(run_dir / "run.log")
    root = Path(config.root)
    train_pair = (int(config.train_pair[0]), int(config.train_pair[1]))
    pair = (int(config.pair[0]), int(config.pair[1]))
    log.line(f"train pair {train_pair}; evaluated pair {pair}; n_sub={config.n_sub}")

    a_tr, b_tr = read_ply(root / f"tr_reg_{train_pair[0]:03d}.ply"), read_ply(root / f"tr_reg_{train_pair[1]:03d}.ply")
    train_index = shared_index(a_tr, int(config.n_sub), 0)
    da_tr, db_tr = standardise(descriptors(a_tr)[train_index], descriptors(b_tr)[train_index])
    fit = fit_theta(SquaredDifferenceFeatures(da_tr, db_tr), rel=None,
                    max_iterations=int(config.max_iterations), backend="highs")
    theta = np.asarray(fit["theta"], dtype=np.float64)
    log.line(f"learned theta={np.round(theta, 3).tolist()} regret={fit['regret']:.4g} "
             f"converged={fit['converged']}")

    k = int(da_tr.shape[1])
    arms: dict[str, Float[np.ndarray, "k"]] = {
        "position_only": np.asarray([1.0, 1.0, 1.0] + [0.0] * (k - 3), dtype=np.float64),
        "uniform_theta": np.full(k, 1.0 / k, dtype=np.float64),
        "learned": theta,
    }

    ta, tb = read_ply(root / f"tr_reg_{pair[0]:03d}.ply"), read_ply(root / f"tr_reg_{pair[1]:03d}.ply")
    index = shared_index(ta, int(config.n_sub), 0)
    da, db, geo, norm, _mirror = pair_geometry(ta, tb, index)
    rows: list[RunRecord] = []
    errors: dict[str, np.ndarray] = {}
    for arm, weights in arms.items():
        pred = match(cost_matrix(da, db, weights))
        per_vertex = np.asarray(geo)[np.arange(index.size), pred] / max(norm, 1e-12)
        errors[arm] = per_vertex
        metrics = {"mean": float(per_vertex.mean()),
                   "miss_5pct": float(np.mean(per_vertex > float(config.threshold))),
                   "wrong_vertices": float(np.sum(per_vertex > float(config.threshold)))}
        rows.append(RunRecord(study=str(config.study), geometry=f"pair{pair[0]:03d}-{pair[1]:03d}", arm=arm,
                              k=k, m=int(index.size), draw=0, metrics=metrics, config=config.to_dict(),
                              extra={"theta": weights.tolist()}))
        log.line(f"{arm:<14} mean={metrics['mean']:.4f} miss5%={metrics['miss_5pct']:.4f} "
                 f"wrong={int(metrics['wrong_vertices'])}")

    np.savez(run_dir / f"errors_pair{pair[0]:03d}-{pair[1]:03d}.npz",
             **{arm: errors[arm].astype(np.float64) for arm in errors},
             index=index.astype(np.int64), pair=np.asarray(pair, dtype=np.int64),
             threshold=np.float64(config.threshold))
    write_records(run_dir / "records.json", rows, provenance=PROVENANCE)
    log.line(f"wrote {run_dir / 'records.json'} and errors_pair{pair[0]:03d}-{pair[1]:03d}.npz")
    log.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
