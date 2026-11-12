"""
EN: **Does the training pair matter, and does pooling pairs help? The cross-decoding table.**

    §4h compared a shared theta against a single-pair theta using *certification* (regret) and one held-out
    score. That left three candidate explanations for the poor transfer, and this module is built to separate
    them:

      1. **the training pair is a lucky/unlucky draw** -- fit theta on each of 8 subject-disjoint pairs and
         score every one on the same held-out pairs. If the spread is small, the pair does not matter.
      2. **regret is misaligned with decoding** -- a theta can certify a plan and still decode badly. Measured
         here as the correlation, across thetas, between mean exact regret on the training pairs and held-out
         `miss@5 %`. Near-zero correlation means the fitting objective does not rank costs for the task.
      3. **pooling the estimates helps** -- average the per-pair thetas over k = 1, 2, 4, 8 pairs and score the
         averages on the same held-out pairs. This is the operation that worked for vertex subsamples (§4e)
         and was never tried across pairs; it is *not* the feasibility-intersection of the block LP.

    Complementing (2), `selection` ranks every candidate (per-pair and pooled) on validation pairs from the
    training subjects and takes the winner to the held-out pairs -- i.e. pooling by selection rather than by
    averaging, which is what §4l showed works inside one pair's near-optimal set.

    Everything here is exact: all regrets are full-grid POT (`ot.emd2`), never a restricted edge set, and all
    scores are geodesic `miss@5 %` on subject-disjoint held-out pairs.

    CN: **训练对到底有多重要、把多对池化是否有用**：8 个受试者不相交的训练对各自拟合 θ，全部在同一批留出对上解码；
        另测 (a) 平均 θ 的学习曲线（k=1,2,4,8），(b) 用验证对挑选候选，(c) "regret 能否排序解码质量"的相关性。
        所有 regret 都是全网格精确 POT，所有分数都是测地 miss@5%。

    Run:
        .venv/bin/python -u -m iot_experiments.studies.real.faust.pairing --n-sub 2000
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
)
from iot_experiments.harness.preflight import check as preflight_check  # noqa: E402
from iot_experiments.harness.records import RunLog, RunRecord, summarise, write_records  # noqa: E402
from iot_experiments.studies.real.faust.correspondence import (  # noqa: E402
    default_pairs,
    fit_theta,
    subject_of,
)

DEFAULT_ROOT = "/home/agent/Documents/fast-iot/data/faust_extract/MPI-FAUST/training/registrations"

# CN: 8 个跨受试者的训练对，全部落在受试者 0–4 内（测试受试者是 5–9）。
# EN: 8 cross-subject training pairs, all inside subjects 0-4 (the test subjects are 5-9).
TRAIN_PAIRS: tuple[tuple[int, int], ...] = (
    (0, 13), (2, 31), (4, 22), (11, 33), (14, 42), (21, 40), (24, 12), (33, 43))

PROVENANCE: dict[str, object] = {
    "question": "is the single training pair a lucky draw, is regret misaligned with decoding, and does "
                "pooling the per-pair thetas help?",
    "training_pairs": "8 cross-subject pairs inside subjects 0-4; SUBJECT-DISJOINT from the test subjects 5-9",
    "regret": "EXACT full-grid POT: R(theta) = <c(theta), pi_hat> - OT(c(theta)); never a restricted edge set, "
              "so the 'certificate was an artefact of F_p' explanation is testable here",
    "pooling": "coordinate-wise mean of the per-pair thetas over the first k pairs (k = 1,2,4,8), renormalised; "
               "plus a validation-ranked selection over every candidate",
    "selection": "6 validation pairs from TRAINING subjects; the test pairs take no part",
    "metric": "geodesic miss@5% on the 8 held-out pairs",
    "memory": "one n x n cost and one n x n geodesic matrix at a time; ~32 MB each at n=2000",
}


@dataclasses.dataclass(frozen=True)
class PairingConfig(RunConfig):
    """CN: 旋钮。EN: knobs."""

    study: str = "faust_pairing"
    section: str = "faust_pairing"
    root: str = DEFAULT_ROOT
    train_subjects: tuple[int, ...] = (0, 1, 2, 3, 4)
    test_subjects: tuple[int, ...] = (5, 6, 7, 8, 9)
    test_pair_count: int = 8
    val_pair_count: int = 6
    n_sub: int = 2000
    max_iterations: int = 30
    backend: str = "cupdlpx"
    curve: tuple[int, ...] = (1, 2, 4, 8)

    @classmethod
    def add_study_arguments(cls, parser) -> None:
        """CN: 只加本研究特有的参数。EN: only this study's flags."""
        parser.add_argument("--root", default=DEFAULT_ROOT)
        parser.add_argument("--train-subjects", type=int, nargs="+", default=[0, 1, 2, 3, 4])
        parser.add_argument("--test-subjects", type=int, nargs="+", default=[5, 6, 7, 8, 9])
        parser.add_argument("--test-pair-count", type=int, default=8)
        parser.add_argument("--val-pair-count", type=int, default=6)
        parser.add_argument("--n-sub", type=int, default=2000)
        parser.add_argument("--max-iterations", type=int, default=30)
        parser.add_argument("--backend", choices=("cupdlpx", "highs"), default="cupdlpx")
        parser.add_argument("--curve", type=int, nargs="+", default=[1, 2, 4, 8])


def _exact_regret(theta: Float[np.ndarray, "k"], a: Float[np.ndarray, "n k"], b: Float[np.ndarray, "n k"]
                  ) -> float:
    """EN: exact full-grid duality gap of the identity plan. CN: 单位方案的全网格精确对偶间隙。"""
    return duality_gap(theta, a, b)[0]


def main(argv: list[str] | None = None) -> int:
    """CN: 入口。EN: entry point."""
    parser = build_parser(PairingConfig)
    args = parser.parse_args(list(argv) if argv is not None else None)
    config = PairingConfig.from_namespace(args, parser)
    run_dir = config.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    log = RunLog(run_dir / "run.log")
    pre = preflight_check(run_tiny=False, need_gpu=False)
    pre.write(run_dir / "preflight.json")
    log.line(f"preflight ok={pre.ok} mem={pre.mem_available_gb:.1f} GB backend={config.backend}")

    root = Path(config.root)
    test_subjects = tuple(int(v) for v in config.test_subjects)
    train_subjects = tuple(int(v) for v in config.train_subjects)
    for pair in TRAIN_PAIRS:
        for vertex_id in pair:
            if subject_of(vertex_id) not in set(train_subjects):
                raise SystemExit(f"training pair {pair} leaves the training subjects")
    test_pairs = default_pairs(test_subjects, int(config.test_pair_count))
    pool = default_pairs(train_subjects, 4 * int(config.val_pair_count) + 16)
    val_pairs = [pair for pair in pool if pair not in TRAIN_PAIRS and pair[::-1] not in TRAIN_PAIRS
                 ][: int(config.val_pair_count)]
    log.line(f"training pairs {TRAIN_PAIRS}")
    log.line(f"validation pairs (training subjects) {val_pairs}")
    log.line(f"test pairs (test subjects) {test_pairs}")

    # CN: **每个训练对单独拟合**，并记录它自己的精确 regret 与训练集解码错误数。
    # EN: **fit every training pair separately**, recording its exact regret and training-decode error count.
    thetas: list[Float[np.ndarray, "k"]] = []
    train_arrays: list[tuple[Float[np.ndarray, "n k"], Float[np.ndarray, "n k"]]] = []
    for pair in TRAIN_PAIRS:
        started = time.perf_counter()
        ma = read_ply(root / f"tr_reg_{pair[0]:03d}.ply")
        mb = read_ply(root / f"tr_reg_{pair[1]:03d}.ply")
        index = shared_index(ma, int(config.n_sub), 0)
        da, db = pair_geometry(ma, mb, index)[:2]
        train_arrays.append((da, db))
        fit = fit_theta(SquaredDifferenceFeatures(da, db), rel=None,
                        max_iterations=int(config.max_iterations), backend=str(config.backend))
        if fit["theta"] is None:
            raise SystemExit(f"fit failed on training pair {pair}")
        theta = np.asarray(fit["theta"], dtype=np.float64)
        thetas.append(theta)
        regret, wrong = duality_gap(theta, da, db)
        log.line(f"   pair {pair}: exact regret {regret:.4g}, training map wrong at {wrong}/{index.size}, "
                 f"{fit['iterations']} rounds, {time.perf_counter() - started:.1f}s")
    k = int(thetas[0].size)

    # CN: 候选 = 8 个单对 θ + 前缀平均 θ（k=1,2,4,8）。平均值未必在单纯形上只有一处——重归一化。
    # EN: candidates = the 8 per-pair thetas + prefix means over k = 1,2,4,8 pairs, renormalised.
    candidates: dict[str, Float[np.ndarray, "k"]] = {f"theta_p{slot}": theta
                                                     for slot, theta in enumerate(thetas)}
    for kk in config.curve:
        stacked = np.stack(thetas[: int(kk)], axis=0).mean(axis=0)
        candidates[f"mean_k{int(kk)}"] = np.asarray(stacked / stacked.sum(), dtype=np.float64)
    candidates["uniform"] = np.full(k, 1.0 / k)
    log.line(f"{len(candidates)} candidates: {sorted(candidates)}")

    # CN: **训练集上的精确 regret 矩阵**（θ_q 在 pair p 上），用于回答"证书是否是 F_p 受限造成的假象"。
    # EN: **exact regret matrix on the training pairs** (theta_q on pair p), which is where the "the
    #     certificate was an artefact of the restricted F_p" explanation would show up.
    regret_matrix = np.zeros((len(TRAIN_PAIRS), len(candidates)), dtype=np.float64)
    for p, (da, db) in enumerate(train_arrays):
        scale = abs(_exact_regret(np.full(k, 1.0 / k), da, db)) or 1.0
        for q, theta in enumerate(candidates.values()):
            regret_matrix[p, q] = _exact_regret(theta, da, db) / scale
    log.line("\n--- exact full-grid regret of each candidate on each TRAINING pair (share of R_uniform) ---")
    names = list(candidates)
    log.line("            " + "".join(f"{name[:9]:>11}" for name in names))
    for p, pair in enumerate(TRAIN_PAIRS):
        log.line(f"pair{str(pair):<10}" + "".join(f"{regret_matrix[p, q]:>11.4f}" for q in range(len(names))))

    # CN: **交叉解码**：每个候选在留出对上打分。这是 §4h 缺失的那张表。
    # EN: **cross-decoding**: every candidate scored on the held-out pairs -- the table §4h was missing.
    log.line("\n--- cross-decoding: candidates x held-out pairs ---")
    held_out = {name: [] for name in candidates}
    per_held: dict[str, dict[str, float]] = {}
    for pair in test_pairs:
        ta = read_ply(root / f"tr_reg_{pair[0]:03d}.ply")
        tb = read_ply(root / f"tr_reg_{pair[1]:03d}.ply")
        geometry = f"pair{pair[0]:03d}-{pair[1]:03d}"
        t_index = shared_index(ta, int(config.n_sub), 0)
        da, db, geo, norm, mirror = pair_geometry(ta, tb, t_index)
        for name, theta in candidates.items():
            score = float(map_metrics(da, db, theta, geo, norm, mirror)["miss_5pct"])
            held_out[name].append(score)
            per_held.setdefault(geometry, {})[name] = score
        del geo
        best = min(candidates, key=lambda name: per_held[geometry][name])
        log.line(f"   {geometry}: best={best} ({per_held[geometry][best]:.3f})  "
                 f"theta_p0={per_held[geometry]['theta_p0']:.3f}  mean_all={per_held[geometry][f'mean_k8']:.3f}")
    held_mean = {name: float(np.mean(values)) for name, values in held_out.items()}
    singles = sorted(((held_mean[f"theta_p{slot}"], slot) for slot in range(len(TRAIN_PAIRS))))
    log.line("\nheld-out miss@5% by candidate (singles sorted):")
    for score, slot in singles:
        log.line(f"   theta_p{slot} (pair {TRAIN_PAIRS[slot]}): {score:.4f}")
    for kk in config.curve:
        log.line(f"   mean_k{int(kk):<2}: {held_mean[f'mean_k{int(kk)}']:.4f}")
    log.line(f"   uniform : {held_mean['uniform']:.4f}")

    single_spread = float(singles[-1][0] - singles[0][0])
    log.line(f"\nspread across the 8 single-pair thetas: {single_spread:.4f} "
             f"(best {singles[0][0]:.4f} pair {TRAIN_PAIRS[singles[0][1]]}, worst {singles[-1][0]:.4f} "
             f"pair {TRAIN_PAIRS[singles[-1][1]]})")

    # CN: **regret 与解码质量的相关性**——机制 2 的定量版本。
    # EN: **regret vs decoding quality** -- the quantitative version of mechanism 2.
    mean_regret = regret_matrix[:, :len(TRAIN_PAIRS)].mean(axis=0)
    decoding = np.array([held_mean[f"theta_p{slot}"] for slot in range(len(TRAIN_PAIRS))])
    spearman = float(np.corrcoef(np.argsort(np.argsort(mean_regret)),
                                 np.argsort(np.argsort(decoding)))[0, 1])
    log.line(f"mean training regret vs held-out miss@5% across the 8 single-pair thetas: "
             f"Spearman {spearman:+.3f} (regret {mean_regret.min():.4f}-{mean_regret.max():.4f}, "
             f"decode {decoding.min():.4f}-{decoding.max():.4f})")

    # CN: **用验证对挑选**：在所有候选里选验证最好的，带去留出对。测试对不参与选择。
    # EN: **select on validation**: rank every candidate on training-subject validation pairs, take the
    #     winner to the held-out pairs; the test pairs take no part.
    log.line("\n--- validation selection over all candidates ---")
    val_scores = {name: [] for name in candidates}
    for pair in val_pairs:
        va = read_ply(root / f"tr_reg_{pair[0]:03d}.ply")
        vb = read_ply(root / f"tr_reg_{pair[1]:03d}.ply")
        v_index = shared_index(va, int(config.n_sub), 0)
        da, db, geo, norm, mirror = pair_geometry(va, vb, v_index)
        for name, theta in candidates.items():
            val_scores[name].append(float(map_metrics(da, db, theta, geo, norm, mirror)["miss_5pct"]))
        del geo
    val_mean = {name: float(np.mean(values)) for name, values in val_scores.items()}
    chosen = min(val_mean, key=lambda name: val_mean[name])
    log.line(f"validation-best candidate: {chosen} (val {val_mean[chosen]:.4f}); "
             f"held-out {held_mean[chosen]:.4f}")
    log.line("validation ranking (top 5): " + ", ".join(
        f"{name}={val_mean[name]:.4f}" for name in sorted(val_mean, key=lambda n: val_mean[n])[:5]))

    rows: list[RunRecord] = []
    for name, theta in candidates.items():
        for geometry, values in per_held.items():
            rows.append(RunRecord(study=str(config.study), geometry=geometry, arm=name, k=k,
                                  m=int(config.n_sub), draw=0,
                                  metrics={"miss_5pct": values[name],
                                           "mean_regret": float(np.mean(
                                               [regret_matrix[p, names.index(name)]
                                                for p in range(len(TRAIN_PAIRS))]))},
                                  config=config.to_dict(), extra={"theta": theta.tolist()}))
    for base in ("theta_p0", "mean_k8"):
        contrasts = bootstrap_contrast(per_held, base=base, others=["uniform", "mean_k1", "mean_k2",
                                                                    "mean_k4", "mean_k8"])
        for other, stats in contrasts.items():
            if other == base:
                continue
            rows.append(RunRecord(study=str(config.study), geometry="contrast", arm=f"{base}_minus_{other}",
                                  k=k, m=int(config.n_sub), draw=-1, metrics=stats,
                                  config=config.to_dict()))
    rows.append(RunRecord(study=str(config.study), geometry="summary", arm="spread", k=k,
                          m=int(config.n_sub), draw=0,
                          metrics={"single_spread": single_spread, "best_single": singles[0][0],
                                   "worst_single": singles[-1][0],
                                   "best_single_pair": float(TRAIN_PAIRS[singles[0][1]][0] * 100
                                                             + TRAIN_PAIRS[singles[0][1]][1]),
                                   "worst_single_pair": float(TRAIN_PAIRS[singles[-1][1]][0] * 100
                                                              + TRAIN_PAIRS[singles[-1][1]][1]),
                                   "spearman_regret_vs_decode": spearman,
                                   "selected_held_out": held_mean[chosen],
                                   "selected_validation": val_mean[chosen],
                                   "uniform": held_mean["uniform"],
                                   "mean_all_pairs": held_mean["mean_k8"]},
                          config=config.to_dict(), extra={"selected": chosen,
                                                          "regret_matrix_names": names,
                                                          "regret_matrix": regret_matrix.tolist()}))
    table = summarise(rows, group=("arm",), metrics=("miss_5pct",))
    log.line("\n" + str(table))
    write_records(run_dir / "records.json", rows, provenance=PROVENANCE)
    log.line(f"wrote {run_dir / 'records.json'} ({len(rows)} rows)")
    log.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
