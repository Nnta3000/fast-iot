"""
EN: **Can the fit tell you where it is wrong, without labels?** Risk-coverage of a cost margin against a
    softmax probability margin.

    Every correspondence method emits a map. A pipeline that could *abstain* on the vertices it is unsure
    about would be more useful than one that is equally confident everywhere, even at identical mean error.
    The question here is which confidence score ranks the errors better, and whether the inverse-OT fit
    produces a better one than the supervised fit of the identical features.

    Two scores, both computed from the same `theta` and the same `n x n` cost matrix, neither using labels:

      `cost_margin`     c_i,(second best) - c_i,(best)          -- a certificate-like quantity: the fit is
                                                                 only meaningful where the second-best is
                                                                 genuinely worse than the best
      `softmax_margin`  P_i,pred - max_{j != pred} P_ij          -- the supervised arm's own confidence,
                        with P = softmax(-c / tau)                 which is what it would ship with

    Evaluation is **risk-coverage**: rank the vertices by confidence, keep the most confident fraction `q`,
    and report `miss@5 %` among them. A score that ranks well drives `miss@5 %` down as `q` shrinks; a score
    that is uninformative leaves it flat. AUROC of "vertex is wrong by > 5 %" is reported alongside.

    **Why this is an inverse-OT question at all.** The margin is the slack of the second-best edge under the
    fitted cost -- the same object the master reasons about. If it ranks errors, the fit is not only a fitter
    but a *self-assessment*; if it does not, the certificate is about the training plan, not about new pairs.

    CN: **拟合能否在无标签的情况下指出自己错在哪**：用代价 margin 与 softmax 概率 margin 做风险-覆盖曲线，
        看哪个更能排序错误，以及逆 OT 拟合是否比同特征的监督拟合给出更好的置信度。

    Run:
        .venv/bin/python -u -m iot_experiments.studies.real.faust.abstention --n-sub 4000
"""

from __future__ import annotations

import dataclasses
import sys
import time
from pathlib import Path

import numpy as np
import torch
from jaxtyping import Float, Int

_REPO_ROOT = Path(__file__).resolve().parents[4]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import ot  # noqa: E402

from iot.core.features import SquaredDifferenceFeatures  # noqa: E402
from iot_experiments.harness.config import RunConfig, build_parser  # noqa: E402
from iot_experiments.harness.faust import geodesic_matrix, read_ply, surface_area  # noqa: E402
from iot_experiments.harness.preflight import check as preflight_check  # noqa: E402
from iot_experiments.harness.records import (  # noqa: E402
    RunLog,
    RunRecord,
    format_table,
    summarise,
    write_records,
)
from iot_experiments.studies.real.faust.correspondence import (  # noqa: E402
    _sample_index,
    default_pairs,
    descriptors,
    fit_theta,
    mirror_indices,
    standardise,
)
from iot_experiments.studies.real.faust.finalists import _cost_numpy, fit_softmax  # noqa: E402

DEFAULT_ROOT = "/home/agent/Documents/fast-iot/data/faust_extract/MPI-FAUST/training/registrations"

PROVENANCE: dict[str, object] = {
    "question": "does a label-free confidence score rank correspondence errors, and is the inverse-OT cost "
                "margin a better ranker than the supervised arm's softmax margin?",
    "scores": "cost_margin = c_(2nd) - c_(1st) per source row; softmax_margin = P_ii - max_{j!=i} P_ij with "
              "P = softmax(-c/tau), tau=0.003 (the value selected on n=6890 validation pairs)",
    "thetas": "relaxed_lp (simplex-only regret master) and row_softmax (one-sided CE, 600 Adam steps) -- the "
              "same 13 standardised descriptors, fitted on the same training pair (0,13)",
    "metric": "risk-coverage: rank by confidence, keep the top q fraction, report miss@5% and mean geodesic "
              "error among them; AUROC of the >5% error indicator",
    "no_labels": "neither confidence score uses the ground-truth correspondence; the truth is used only to "
                 "score the ranking",
    "memory": "cost + geodesic matrix + POT plan are three n x n float64 arrays; at n=4000 that is ~384 MB",
}


@dataclasses.dataclass(frozen=True)
class AbstentionConfig(RunConfig):
    """CN: 旋钮。EN: knobs."""

    study: str = "faust_abstention"
    section: str = "faust_abstention"
    root: str = DEFAULT_ROOT
    train_pair: tuple[int, ...] = (0, 13)
    test_subjects: tuple[int, ...] = (5, 6, 7, 8, 9)
    test_pair_count: int = 8
    n_sub: int = 4000
    tau: float = 0.003
    steps: int = 600
    lr: float = 0.05
    max_iterations: int = 30
    coverages: tuple[float, ...] = (0.5, 0.7, 0.9, 1.0)

    @classmethod
    def add_study_arguments(cls, parser) -> None:
        """CN: 只加本研究特有的参数。EN: only this study's flags."""
        parser.add_argument("--root", default=DEFAULT_ROOT)
        parser.add_argument("--train-pair", type=int, nargs=2, default=[0, 13])
        parser.add_argument("--test-subjects", type=int, nargs="+", default=[5, 6, 7, 8, 9])
        parser.add_argument("--test-pair-count", type=int, default=8)
        parser.add_argument("--n-sub", type=int, default=4000)
        parser.add_argument("--tau", type=float, default=0.003)
        parser.add_argument("--steps", type=int, default=600)
        parser.add_argument("--lr", type=float, default=0.05)
        parser.add_argument("--max-iterations", type=int, default=30)
        parser.add_argument("--coverages", type=float, nargs="+", default=[0.5, 0.7, 0.9, 1.0])


def _auc(scores: Float[np.ndarray, "n"], labels: Int[np.ndarray, "n"]) -> float:
    """
    EN: AUROC of `labels` (1 = the class we want to rank first) against `scores` (higher = more confident).
        Rank-based, no sklearn.
    CN: `labels`（1 = 我们想排在前面的一类）对 `scores`（越高越自信）的 AUROC，基于秩，不用 sklearn。
    """
    count = int(scores.size)
    n_pos = int(labels.sum())
    n_neg = count - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(count, dtype=np.float64)
    ranks[order] = np.arange(1, count + 1, dtype=np.float64)
    return float((ranks[labels.astype(bool)].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def _confidences(cost: Float[np.ndarray, "n n"], pred: Int[np.ndarray, "n"], tau: float,
                 dual_v: Float[np.ndarray, "n"]
                 ) -> dict[str, Float[np.ndarray, "n"]]:
    """
    EN: four label-free confidence scores.

        `cost_margin`      c_(2nd) - c_(1st)                       -- naive: scale-dependent, because the
                                                                     magnitude of the cost varies by vertex
        `rel_cost_margin`  (c_(2nd) - c_(1st)) / c_(2nd)            -- scale-free version of the above
        `dual_margin`      max_{j != pred} (c_ij - v_j) - (c_i,pred - v_pred)
                                                                    -- the **OT dual certificate**: the slack
                                                                     of the tightest non-plan edge, after
                                                                     removing each target's own price `v_j`.
                                                                     Zero means the plan is not determined
                                                                     for that row; this is the quantity the
                                                                     optimality proof actually reasons about
        `softmax_margin`   P_i,pred - max_{j != pred} P_ij          -- what the supervised arm would ship

    CN: 四个无标签置信度：原始代价 margin、相对代价 margin、**OT 对偶证书**（去掉目标自身价格 `v_j` 后最紧的
        非方案边的松弛）、softmax 概率 margin。
    """
    as_cost = np.asarray(cost, dtype=np.float64)
    rows = np.arange(pred.size)
    two = np.partition(as_cost, 1, axis=1)[:, :2]
    first, second = two[:, 0], two[:, 1]
    cost_margin = second - first
    rel_cost_margin = cost_margin / np.maximum(second, 1e-12)

    reduced = as_cost - np.asarray(dual_v, dtype=np.float64)[None, :]
    chosen = reduced[rows, pred]
    reduced[rows, pred] = np.inf
    dual_margin = np.maximum(reduced.min(axis=1) - chosen, 0.0)

    logits = -as_cost / float(tau)
    logits -= logits.max(axis=1, keepdims=True)
    prob = np.exp(logits)
    prob /= prob.sum(axis=1, keepdims=True)
    chosen_prob = prob[rows, pred]
    prob[rows, pred] = -np.inf
    return {"cost_margin": np.asarray(cost_margin, dtype=np.float64),
            "rel_cost_margin": np.asarray(rel_cost_margin, dtype=np.float64),
            "dual_margin": np.asarray(dual_margin, dtype=np.float64),
            "softmax_margin": np.asarray(chosen_prob - prob.max(axis=1), dtype=np.float64)}


def main(argv: list[str] | None = None) -> int:
    """CN: 入口。EN: entry point."""
    parser = build_parser(AbstentionConfig)
    args = parser.parse_args(list(argv) if argv is not None else None)
    config = AbstentionConfig.from_namespace(args, parser)
    run_dir = config.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    log = RunLog(run_dir / "run.log")
    pre = preflight_check(run_tiny=False, need_gpu=False)
    pre.write(run_dir / "preflight.json")
    log.line(f"preflight ok={pre.ok} mem={pre.mem_available_gb:.1f} GB "
             f"vram_free={torch.cuda.mem_get_info()[0] / 2**30 if torch.cuda.is_available() else 0:.1f} GB")
    device = "cuda" if torch.cuda.is_available() and torch.cuda.mem_get_info()[0] / 2**30 > 1.5 else "cpu"

    root = Path(config.root)
    train_pair = (int(config.train_pair[0]), int(config.train_pair[1]))
    test_pairs = default_pairs(tuple(int(v) for v in config.test_subjects), int(config.test_pair_count))
    a_tr = read_ply(root / f"tr_reg_{train_pair[0]:03d}.ply")
    b_tr = read_ply(root / f"tr_reg_{train_pair[1]:03d}.ply")
    idx = _sample_index(a_tr, config, int(config.seed))
    a, b = standardise(descriptors(a_tr)[idx], descriptors(b_tr)[idx])
    k = int(a.shape[1])
    log.line(f"train pair {train_pair}, n={idx.size}, K={k}; test {test_pairs}")

    started = time.perf_counter()
    lp_fit = fit_theta(SquaredDifferenceFeatures(a, b), rel=None, max_iterations=int(config.max_iterations))
    thetas: dict[str, Float[np.ndarray, "k"]] = {}
    if lp_fit["theta"] is not None:
        thetas["relaxed_lp"] = np.asarray(lp_fit["theta"], dtype=np.float64)
    log.line(f"relaxed_lp regret={lp_fit['regret']:.4g} converged={lp_fit['converged']} "
             f"({time.perf_counter() - started:.1f}s)")
    started = time.perf_counter()
    thetas["row_softmax"] = fit_softmax(a, b, tau=float(config.tau), two_sided=False,
                                        steps=int(config.steps), lr=float(config.lr),
                                        sinkhorn_iterations=1, device=device)
    log.line(f"row_softmax tau={config.tau:g} ({time.perf_counter() - started:.1f}s, {device})")
    thetas["uniform"] = np.full(k, 1.0 / k)
    for name, theta in thetas.items():
        log.line(f"   {name:<12} theta={np.round(theta, 4).tolist()}")

    rows: list[RunRecord] = []
    for pair in test_pairs:
        ta = read_ply(root / f"tr_reg_{pair[0]:03d}.ply")
        tb = read_ply(root / f"tr_reg_{pair[1]:03d}.ply")
        geometry = f"pair{pair[0]:03d}-{pair[1]:03d}"
        # CN: 评估用的子采样下标（同一 seed，两边同一组下标，双射保持）。
        # EN: the evaluation subsample (same seed, the same index set on both shapes, so the bijection holds).
        test_idx = _sample_index(ta, config, int(config.seed))
        a_test, b_test = standardise(descriptors(ta)[test_idx], descriptors(tb)[test_idx])
        geo = geodesic_matrix(tb.vertices, tb.faces, test_idx)
        norm = float(np.sqrt(surface_area(tb.vertices, tb.faces)))
        mirror, _res = mirror_indices(tb.vertices, test_idx)
        ee = np.arange(int(test_idx.size))
        for name, theta in thetas.items():
            cost = _cost_numpy(a_test, b_test, theta)
            mass = np.full(ee.size, 1.0 / ee.size)
            plan, pot_log = ot.emd(mass, mass, cost, numItermax=10_000_000, log=True)
            pred = np.argmax(plan, axis=1).astype(np.int64)
            error = np.asarray(geo)[ee, pred] / max(norm, 1e-12)
            # CN: POT 返回的 `v` 是对偶势；它是 `dual_margin` 唯一额外需要的东西。
            # EN: POT returns the dual potential `v`; it is the only extra input `dual_margin` needs.
            scores = _confidences(cost, pred, float(config.tau), np.asarray(pot_log["v"], dtype=np.float64))
            # CN: 标签取"正确"（误差 ≤ 5%），于是 AUC > 0.5 表示置信度确实把正确的排在前面。
            # EN: the label is *correctness* (error <= 5 %), so AUC > 0.5 means the score really does put the
            #     correct vertices first.
            correct = (error <= 0.05).astype(np.int64)
            for label, score in scores.items():
                rows.append(RunRecord(study=str(config.study), geometry=geometry,
                                      arm=f"{name}:{label}:auc", k=k, m=int(ee.size), draw=0,
                                      metrics={"auc": _auc(score, correct)}, config=config.to_dict()))
                order = np.argsort(-score)
                for coverage in config.coverages:
                    keep = order[: max(1, int(round(float(coverage) * ee.size)))]
                    rows.append(RunRecord(study=str(config.study), geometry=geometry,
                                          arm=f"{name}:{label}:cov{float(coverage):g}", k=k,
                                          m=int(keep.size), draw=0,
                                          metrics={"mean": float(error[keep].mean()),
                                                   "miss_5pct": float((error[keep] > 0.05).mean())},
                                          config=config.to_dict()))
            del cost, plan
        log.line(f"   {geometry}: done")
        del geo

    table = summarise(rows, group=("arm",), metrics=("mean", "miss_5pct"))
    log.line("\n" + format_table(table, ("arm", "n_draws", "mean_mean", "miss_5pct_mean")))
    # CN: AUROC 单独一张表，因为它的指标名不同。EN: AUROC separately, since its metric name differs.
    auc_rows = [r for r in rows if r.arm.endswith(":auc")]
    auc_table = summarise(auc_rows, group=("arm",), metrics=("auc",))
    log.line("\n" + format_table(auc_table, ("arm", "n_draws", "auc_mean")))
    write_records(run_dir / "records.json", rows, provenance=PROVENANCE)
    log.line(f"wrote {run_dir / 'records.json'} ({len(rows)} rows)")
    log.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
