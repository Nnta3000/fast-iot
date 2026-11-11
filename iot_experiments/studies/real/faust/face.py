"""
EN: **Diagnostic D1 -- is the admissible set worth more than one point?**

    `theta` is identified only up to a cone: many weights certify the same training plan, and the shipped
    solver returns the max-margin vertex of that cone by convention, not by evidence. The corner analysis
    flagged this early; nothing has ever measured whether the choice matters for *generalisation*.

    This module walks inside the cone and scores every point it finds on held-out pairs. Membership is
    checked **exactly** and without labels: `theta` is admissible iff the training plan is optimal for
    `c(theta)`, i.e. iff the regret `R(theta) = <c(theta), pi_hat> - OT(c(theta))` is zero. Rejection
    sampling from the simplex would almost never land inside, so the walk starts at the fitted vertex and
    takes normalised Gaussian steps, accepting only feasible proposals.

    Readings:

    - **fraction of proposals admitted** -- how thick the cone is;
    - **spread of `miss@5 %` over admitted points** -- the quantity the diagnostic turns on. If it is within
      validation noise, the max-margin convention is harmless; if it is large, selecting inside the cone is a
      real lever;
    - **where the max-margin point sits** in that spread, as a rank.

    CN: **诊断 D1**：θ 只在锥内可识别，而已交付的解只是在锥上取极大裕度顶点。本模块在锥内做拒绝采样（可行性
        用精确 OT 的 regret 是否为 0 判定，无需标签），把每个可行点都在留出对上打分，报告可行率、`miss@5%`
        的离散度，以及极大裕度点在其中的排名。

    Run:
        .venv/bin/python -u -m iot_experiments.studies.real.faust.face --n-sub 4000
"""

from __future__ import annotations

import dataclasses
import sys
import time
from pathlib import Path

import numpy as np
import ot
from jaxtyping import Float

_REPO_ROOT = Path(__file__).resolve().parents[4]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from iot.core.features import SquaredDifferenceFeatures  # noqa: E402
from iot_experiments.harness.config import RunConfig, build_parser  # noqa: E402
from iot_experiments.harness.faust import pair_geometry, read_ply, shared_index  # noqa: E402
from iot_experiments.harness.faust_match import cost_matrix, map_metrics  # noqa: E402
from iot_experiments.harness.preflight import check as preflight_check  # noqa: E402
from iot_experiments.harness.records import RunLog, RunRecord, summarise, write_records  # noqa: E402
from iot_experiments.studies.real.faust.correspondence import default_pairs, fit_theta  # noqa: E402

DEFAULT_ROOT = "/home/agent/Documents/fast-iot/data/faust_extract/MPI-FAUST/training/registrations"

PROVENANCE: dict[str, object] = {
    "question": "do different admissible thetas (same training plan, different weights) generalise "
                "differently? i.e. is the max-margin vertex a good selection rule?",
    "admissibility": "measured, not assumed: the reference theta does NOT make the training plan optimal "
                     "(its decoded map is wrong at thousands of vertices and its exact regret is ~1% of "
                     "R_uniform). Admissibility is therefore defined RELATIVE to the shipped reference: "
                     "regret <= R(reference) * (1 + slack_rel) -- 'equally good on the training pair'",
    "why_relative": "an absolute criterion was tried first and is empty at the reference, so it would have "
                    "reported 'one point' for the wrong reason. The discrete test (decoded map == identity) "
                    "is reported alongside as the honest strength of the certificate",
    "sampler": "normalised Gaussian random walk from the fitted vertex; only feasible proposals are kept, so "
               "every scored theta certifies the SAME plan",
    "metric": "geodesic miss@5% on the 8 held-out pairs; the spread across admitted thetas is the diagnostic",
    "go_no_go": "GO if the spread exceeds ~0.01 (the seed-to-seed noise) and the max-margin point is not "
                "near the top; NO-GO if every admitted theta scores within noise",
}


@dataclasses.dataclass(frozen=True)
class FaceConfig(RunConfig):
    """CN: 旋钮。EN: knobs."""

    study: str = "faust_face"
    section: str = "faust_face"
    root: str = DEFAULT_ROOT
    train_pair: tuple[int, ...] = (0, 13)
    test_subjects: tuple[int, ...] = (5, 6, 7, 8, 9)
    test_pair_count: int = 8
    n_sub: int = 4000
    proposals: int = 120
    slack_rel: float = 0.25
    step: float = 0.08
    tol_rel: float = 1e-5
    max_iterations: int = 30

    @classmethod
    def add_study_arguments(cls, parser) -> None:
        """CN: 只加本研究特有的参数。EN: only this study's flags."""
        parser.add_argument("--root", default=DEFAULT_ROOT)
        parser.add_argument("--train-pair", type=int, nargs=2, default=[0, 13])
        parser.add_argument("--test-subjects", type=int, nargs="+", default=[5, 6, 7, 8, 9])
        parser.add_argument("--test-pair-count", type=int, default=8)
        parser.add_argument("--n-sub", type=int, default=4000)
        parser.add_argument("--proposals", type=int, default=120)
        parser.add_argument("--slack-rel", type=float, default=0.25)
        parser.add_argument("--step", type=float, default=0.08)
        parser.add_argument("--tol-rel", type=float, default=1e-5)
        parser.add_argument("--max-iterations", type=int, default=30)


def diagnose(theta: Float[np.ndarray, "k"], a: Float[np.ndarray, "n k"], b: Float[np.ndarray, "n k"]
             ) -> tuple[bool, float, int]:
    """
    EN: admissibility and its two diagnostics, all label-free.

        `admissible` is **discrete and exact**: decode the exact OT plan under `c(theta)` and ask whether the
        induced map is the identity on every vertex. That is the statement the study actually needs -- "this
        theta reproduces the training correspondence" -- and it is robust to the 1e-13 numerical noise that
        makes a *regret* threshold a knife edge (the reference's own regret is ~1e-4, so a tolerance below
        that rejects every perturbation).

        `regret` is `R(theta) = <c(theta), I/n> - OT(c(theta))`, reported for context, and `wrong` is the
        number of vertices the decoded map gets wrong (0 for an admitted theta).

    CN: `admissible` 用**离散精确**判据：在 `c(θ)` 下解精确 OT，检查诱导映射是否逐顶点为单位映射——这正是本研究
        需要的"该 θ 复现训练对应"。`regret` 与错配顶点数一并返回作为诊断量。
    """
    cost = cost_matrix(a, b, theta)
    n = int(cost.shape[0])
    mass = np.full(n, 1.0 / n)
    plan = ot.emd(mass, mass, cost, numItermax=10_000_000)
    predicted = np.argmax(plan, axis=1).astype(np.int64)
    wrong = int(np.sum(predicted != np.arange(n)))
    transport = float(np.sum(cost * plan))
    value = float(np.trace(cost) / n) - transport
    return wrong == 0, value, wrong


def main(argv: list[str] | None = None) -> int:
    """CN: 入口。EN: entry point."""
    parser = build_parser(FaceConfig)
    args = parser.parse_args(list(argv) if argv is not None else None)
    config = FaceConfig.from_namespace(args, parser)
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
    log.line(f"train pair {train_pair}, n={index.size}, K={k}")

    started = time.perf_counter()
    fit = fit_theta(SquaredDifferenceFeatures(a, b), rel=None, max_iterations=int(config.max_iterations))
    if fit["theta"] is None:
        log.line("the fit failed; nothing to sample around")
        log.close()
        return 1
    reference = np.asarray(fit["theta"], dtype=np.float64)
    log.line(f"reference (max-margin) theta fitted in {time.perf_counter() - started:.1f}s; "
             f"regret={fit['regret']:.4g} zeros={int(np.sum(reference <= 1e-12))}")

    uniform_admissible, uniform_regret, uniform_wrong = diagnose(np.full(k, 1.0 / k), a, b)
    ref_admissible, reference_regret, ref_wrong = diagnose(reference, a, b)
    log.line(f"R_uniform={uniform_regret:.4g} (decoded map wrong at {uniform_wrong} vertices); "
             f"R(reference)={reference_regret:.4g} (wrong at {ref_wrong}); "
             f"reference admissible: {ref_admissible}")

    # CN: **拒绝采样**：从参考顶点出发，接受仍能认证训练方案的扰动。步长自适应——可行就留，不可行就缩小。
    # EN: **rejection sampling** from the reference vertex, keeping only perturbations that still certify the
    #     training plan. The step shrinks on rejection and is nudged up on acceptance, so the walk explores the
    #     cone instead of sticking at one point.
    rng = np.random.default_rng(0)
    # CN: 相对判据——"与已交付参考解在训练对上同样好"。绝对判据在参考点处为空集（见 provenance）。
    # EN: relative criterion -- "as good as the shipped reference on the training pair". The absolute one is
    #     empty at the reference, which would have made the diagnostic report "one point" for the wrong reason.
    tolerance = abs(reference_regret) * (1.0 + float(config.slack_rel))
    log.line(f"admission tolerance (relative): R <= {tolerance:.4g} = R(reference) * {1 + float(config.slack_rel):.2f}")

    admitted: list[Float[np.ndarray, "k"]] = [reference]
    admitted_regret: list[float] = [abs(reference_regret)]
    step = float(config.step)
    proposals = 0
    while proposals < int(config.proposals):
        proposals += 1
        direction = rng.normal(size=k)
        candidate = np.maximum(reference + step * direction, 0.0)
        total = float(candidate.sum())
        if total <= 0.0:
            step *= 0.7
            continue
        candidate /= total
        _ok, value, _wrong = diagnose(candidate, a, b)
        value = abs(value)
        if value <= tolerance:
            admitted.append(candidate)
            admitted_regret.append(value)
            step = min(step * 1.15, 0.5)
        else:
            step *= 0.85
    log.line(f"admitted {len(admitted) - 1}/{proposals} proposals (shell thickness); "
             f"max admission regret {max(admitted_regret):.3g}")
    log.line(f"admitted thetas have {int(np.mean([np.sum(t <= 1e-12) for t in admitted]))} exact zeros on average, "
             f"reference has {int(np.sum(reference <= 1e-12))}")

    # CN: 每对只算一次测地矩阵，然后对所有可行 θ 复用（这是评估里最贵的部分）。
    # EN: one geodesic matrix per pair, reused for every admitted theta -- that is the expensive part.
    per_theta: list[list[float]] = [[] for _ in admitted]
    for pair in test_pairs:
        ta = read_ply(root / f"tr_reg_{pair[0]:03d}.ply")
        tb = read_ply(root / f"tr_reg_{pair[1]:03d}.ply")
        test_index = shared_index(ta, int(config.n_sub), 0)
        da, db, geo, norm, mirror = pair_geometry(ta, tb, test_index)
        for slot, theta in enumerate(admitted):
            per_theta[slot].append(float(map_metrics(da, db, theta, geo, norm, mirror)["miss_5pct"]))
        del geo
        log.line(f"   pair {pair}: scored {len(admitted)} thetas")

    scores = np.array([float(np.mean(values)) for values in per_theta], dtype=np.float64)
    order = np.argsort(scores)
    rank = int(np.where(order == 0)[0][0]) + 1
    log.line(f"\nmiss@5% over admitted thetas: min {scores.min():.4f} | median {np.median(scores):.4f} | "
             f"max {scores.max():.4f} | spread {scores.max() - scores.min():.4f}")
    log.line(f"max-margin (reference) theta: {scores[0]:.4f}  (rank {rank}/{len(scores)})")
    log.line(f"best admitted theta: {scores[order[0]]:.4f}; worst: {scores[order[-1]]:.4f}")

    rows: list[RunRecord] = [
        RunRecord(study=str(config.study), geometry="summary", arm="admitted", k=k, m=int(index.size),
                  draw=0,
                  metrics={"count": float(len(scores)), "min": float(scores.min()),
                           "median": float(np.median(scores)), "max": float(scores.max()),
                           "spread": float(scores.max() - scores.min()),
                           "reference": float(scores[0]), "reference_rank": float(rank),
                           "admitted_fraction": float((len(admitted) - 1) / max(proposals, 1)),
                           "reference_wrong": float(ref_wrong),
                           "reference_regret": float(reference_regret),
                           "slack_rel": float(config.slack_rel),
                           "max_admission_regret": float(max(admitted_regret)),
                           "R_uniform": float(uniform_regret)},
                  config=config.to_dict()),
    ]
    for slot, theta in enumerate(admitted):
        rows.append(RunRecord(study=str(config.study), geometry="theta", arm=f"theta{slot:03d}", k=k,
                              m=int(index.size), draw=0,
                              metrics={"miss_5pct": float(scores[slot]),
                                       "regret": float(admitted_regret[slot]),
                                       "sha": float(abs(theta.sum())), "max": float(theta.max()),
                                       "zeros": float(np.sum(theta <= 1e-12))},
                              config=config.to_dict()))
    log.line("\n" + str(summarise(rows[:1], group=("arm",), metrics=("spread", "reference"))))
    write_records(run_dir / "records.json", rows, provenance=PROVENANCE)
    log.line(f"wrote {run_dir / 'records.json'} ({len(rows)} rows)")
    log.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
