"""
EN: **Is inverse OT doing any of the work? The ablation against random search.**

    Everything in this study is supervised -- the registered plan is the label at every stage. So the honest
    question is not "is it supervised" but "does the *inverse-OT fit* locate something that a cheaper search
    would miss?". §4m showed that selecting among candidate thetas on held-out evidence is worth more than
    averaging them, which makes that question urgent: if random thetas selected the same way do as well, the
    solver is decoration.

    Four families, **selected identically** on the same 6 validation pairs from the training subjects and
    evaluated on the same 8 held-out pairs:

      | family | what it tests |
      | --- | --- |
      | `iot` (8, one per training pair) | the full method |
      | `gd` (one-sided cross-entropy, `tau` on validation) | a purely supervised learner |
      | `random_support` (60 draws) | "find the right block, then tune" -- weights only where the fit put mass |
      | `random_simplex` (60 draws) | the weak control; mostly measures how large the good region is |

    The *strong* control is `random_support`: a uniform Dirichlet over 13 coordinates puts ~0.23 of its mass
    on the three late-HKS columns, while every fitted theta puts 0.85-0.95 there, so `random_simplex` is
    almost guaranteed to be bad and refutes little. `random_support` asks whether the *weights inside the
    block* carry information the fit found and a random draw would not.

    For each family this reports the validation-selected theta's held-out score, and the whole distribution of
    held-out scores in that family (min/median/max). If the best of 60 random draws is as good as the
    validation-selected iOT theta, the fitting is not needed; if the *distribution* of random draws reaches
    the same level often, the good region is large and selection is doing all the work.

    CN: **逆 OT 到底有没有贡献**：四个候选族用**完全相同**的验证选择与留出评估——iOT 拟合、监督梯度下降、
        在拟合支撑内随机、在单纯形上随机。强对照是"支撑内随机"：它把"找到正确区块"与"区块内权重"分开。

    Run:
        .venv/bin/python -u -m experiments.studies.iot.real.faust.ablation --n-sub 2000 --draws 60
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
from experiments.harness.config import RunConfig, build_parser  # noqa: E402
from experiments.harness.faust import default_registrations_root, pair_geometry, read_ply, shared_index  # noqa: E402
from experiments.harness.faust_match import (  # noqa: E402
    bootstrap_contrast,
    fit_softmax,
    free_vram_gb,
    map_metrics,
)
from experiments.harness.preflight import check as preflight_check  # noqa: E402
from experiments.harness.records import RunLog, RunRecord, summarise, write_records  # noqa: E402
from experiments.studies.iot.real.faust.correspondence import (  # noqa: E402
    default_pairs,
    fit_theta,
    subject_of,
)
from experiments.studies.iot.real.faust.pairing import TRAIN_PAIRS  # noqa: E402

DEFAULT_ROOT = str(default_registrations_root())

PROVENANCE: dict[str, object] = {
    "question": "does the inverse-OT fit locate thetas that a random search over the same support, or a "
                "supervised gradient learner, would miss?",
    "families": "iot (8, one per training pair), gd (one-sided CE, tau on validation), random_support (draws "
                "inside the support of the mean fitted theta), random_simplex (draws on the full simplex)",
    "selection": "identical for every family: mean miss@5% on 6 validation pairs from the TRAINING subjects",
    "evaluation": "8 held-out pairs (subjects 5-9), geodesic miss@5%",
    "why_support": "a uniform Dirichlet puts ~0.23 of its mass on the three late-HKS columns while every fitted "
                   "theta puts 0.85-0.95 there, so random_simplex is a weak control by construction; "
                   "random_support is the informative one",
    "memory": "one n x n cost and one n x n geodesic matrix at a time; ~32 MB each at n=2000",
}


@dataclasses.dataclass(frozen=True)
class AblationConfig(RunConfig):
    """CN: 旋钮。EN: knobs."""

    study: str = "faust_ablation"
    section: str = "faust_ablation"
    root: str = DEFAULT_ROOT
    train_subjects: tuple[int, ...] = (0, 1, 2, 3, 4)
    test_subjects: tuple[int, ...] = (5, 6, 7, 8, 9)
    test_pair_count: int = 8
    val_pair_count: int = 6
    n_sub: int = 2000
    n_draws: int = 60
    steps: int = 600
    lr: float = 0.05
    tau_grid: tuple[float, ...] = (1e-3, 3e-3, 1e-2)
    max_iterations: int = 30
    backend: str = "cupdlpx"

    @classmethod
    def add_study_arguments(cls, parser) -> None:
        """CN: 只加本研究特有的参数。EN: only this study's flags."""
        parser.add_argument("--root", default=DEFAULT_ROOT)
        parser.add_argument("--train-subjects", type=int, nargs="+", default=[0, 1, 2, 3, 4])
        parser.add_argument("--test-subjects", type=int, nargs="+", default=[5, 6, 7, 8, 9])
        parser.add_argument("--test-pair-count", type=int, default=8)
        parser.add_argument("--val-pair-count", type=int, default=6)
        parser.add_argument("--n-sub", type=int, default=2000)
        parser.add_argument("--n-draws", type=int, default=60)
        parser.add_argument("--steps", type=int, default=600)
        parser.add_argument("--lr", type=float, default=0.05)
        parser.add_argument("--tau-grid", type=float, nargs="+", default=[1e-3, 3e-3, 1e-2])
        parser.add_argument("--max-iterations", type=int, default=30)
        parser.add_argument("--backend", choices=("cupdlpx", "highs"), default="cupdlpx")


def _score_all(a: Float[np.ndarray, "n k"], b: Float[np.ndarray, "n k"],
               theta_list: list[Float[np.ndarray, "k"]], geo: Float[np.ndarray, "n n"], norm: float,
               mirror: Int[np.ndarray, "n"] | None) -> list[float]:
    """EN: `miss@5%` of every theta on one pair. CN: 一对上所有 θ 的 miss@5%。"""
    return [float(map_metrics(a, b, theta, geo, norm, mirror)["miss_5pct"]) for theta in theta_list]


def main(argv: list[str] | None = None) -> int:
    """CN: 入口。EN: entry point."""
    parser = build_parser(AblationConfig)
    args = parser.parse_args(list(argv) if argv is not None else None)
    config = AblationConfig.from_namespace(args, parser)
    run_dir = config.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    log = RunLog(run_dir / "run.log")
    pre = preflight_check(run_tiny=False, need_gpu=False)
    pre.write(run_dir / "preflight.json")
    device = "cuda" if free_vram_gb() > 1.5 else "cpu"
    log.line(f"preflight ok={pre.ok} mem={pre.mem_available_gb:.1f} GB; softmax fits on {device}")

    root = Path(config.root)
    train_subjects = tuple(int(v) for v in config.train_subjects)
    test_subjects = tuple(int(v) for v in config.test_subjects)
    for pair in TRAIN_PAIRS:
        for vertex_id in pair:
            if subject_of(vertex_id) not in set(train_subjects):
                raise SystemExit(f"training pair {pair} leaves the training subjects")
    test_pairs = default_pairs(test_subjects, int(config.test_pair_count))
    pool = default_pairs(train_subjects, 4 * int(config.val_pair_count) + 16)
    val_pairs = [pair for pair in pool if pair not in TRAIN_PAIRS and pair[::-1] not in TRAIN_PAIRS
                 ][: int(config.val_pair_count)]
    log.line(f"train pairs {TRAIN_PAIRS}; validation {val_pairs}; test {test_pairs}")

    # CN: **iOT 族**：每个训练对一个 θ。
    # EN: **the iOT family**: one theta per training pair.
    iot: list[Float[np.ndarray, "k"]] = []
    train_arrays: dict[tuple[int, int], tuple] = {}
    for pair in TRAIN_PAIRS:
        started = time.perf_counter()
        ma = read_ply(root / f"tr_reg_{pair[0]:03d}.ply")
        mb = read_ply(root / f"tr_reg_{pair[1]:03d}.ply")
        index = shared_index(ma, int(config.n_sub), 0)
        da, db = pair_geometry(ma, mb, index)[:2]
        train_arrays[pair] = (da, db)
        fit = fit_theta(SquaredDifferenceFeatures(da, db), rel=None,
                        max_iterations=int(config.max_iterations), backend=str(config.backend))
        if fit["theta"] is None:
            raise SystemExit(f"fit failed on {pair}")
        iot.append(np.asarray(fit["theta"], dtype=np.float64))
        log.line(f"   iot {pair}: {time.perf_counter() - started:.1f}s, "
                 f"zeros={int(np.sum(iot[-1] <= 1e-9))}, max={float(iot[-1].max()):.3f}")
    k = int(iot[0].size)

    # CN: **支撑**：平均 θ 的显著坐标（相对阈值，CuPDLPx 会留下 ~1e-7 噪声）。
    # EN: **the support**: significant coordinates of the mean theta (relative threshold; CuPDLPx leaves
    #     ~1e-7 noise that an absolute one would count).
    mean_theta = np.mean(np.stack(iot, axis=0), axis=0)
    support = np.where(mean_theta > 1e-3 * float(mean_theta.max()))[0]
    log.line(f"support of the mean iOT theta: {support.tolist()} ({support.size}/{k}); "
             f"mass on it {float(mean_theta[support].sum()):.3f}")

    # CN: **GD 族**：单边交叉熵，τ 在验证对上选（与其它族相同的选择协议）。
    # EN: **the GD family**: one-sided cross-entropy, tau chosen on the validation pairs.
    ma, mb = read_ply(root / f"tr_reg_{TRAIN_PAIRS[0][0]:03d}.ply"), read_ply(root / f"tr_reg_{TRAIN_PAIRS[0][1]:03d}.ply")
    index0 = shared_index(ma, int(config.n_sub), 0)
    gd_a, gd_b = pair_geometry(ma, mb, index0)[:2]
    gd_candidates: list[Float[np.ndarray, "k"]] = []
    for tau in config.tau_grid:
        started = time.perf_counter()
        gd_candidates.append(fit_softmax(gd_a, gd_b, tau=float(tau), two_sided=False,
                                         steps=int(config.steps), lr=float(config.lr),
                                         sinkhorn_iterations=1, device=device))
        log.line(f"   gd tau={tau:g}: {time.perf_counter() - started:.1f}s")

    rng = np.random.default_rng(0)
    draws_support = rng.dirichlet(np.ones(support.size), size=int(config.n_draws))
    random_support = []
    for row in draws_support:
        theta = np.zeros(k, dtype=np.float64)
        theta[support] = row
        random_support.append(theta)
    random_simplex = [np.asarray(row, dtype=np.float64)
                      for row in rng.dirichlet(np.ones(k), size=int(config.n_draws))]
    log.line(f"families: iot {len(iot)}, gd {len(gd_candidates)}, random_support {len(random_support)}, "
             f"random_simplex {len(random_simplex)}")

    # CN: **验证评分**：所有候选、同一组验证对。每对只算一次测地矩阵。
    # EN: **validation scoring**: every candidate, the same validation pairs, one geodesic matrix per pair.
    families: dict[str, list[Float[np.ndarray, "k"]]] = {
        "iot": iot, "gd": gd_candidates, "random_support": random_support,
        "random_simplex": random_simplex}
    val_scores: dict[str, list[float]] = {name: [] for name in families}
    for pair in val_pairs:
        va = read_ply(root / f"tr_reg_{pair[0]:03d}.ply")
        vb = read_ply(root / f"tr_reg_{pair[1]:03d}.ply")
        v_index = shared_index(va, int(config.n_sub), 0)
        da, db, geo, norm, mirror = pair_geometry(va, vb, v_index)
        for name, thetas in families.items():
            val_scores[name].append(_score_all(da, db, thetas, geo, norm, mirror))
        del geo
        log.line(f"   validation pair {pair}: scored {sum(len(v) for v in families.values())} candidates")
    val_mean = {name: np.mean(np.array(scores), axis=0) for name, scores in val_scores.items()}
    selected: dict[str, int] = {name: int(np.argmin(values)) for name, values in val_mean.items()}
    for name in families:
        log.line(f"   {name:<16} validation min {val_mean[name].min():.4f} "
                 f"median {np.median(val_mean[name]):.4f} max {val_mean[name].max():.4f}; "
                 f"selected #{selected[name]} ({val_mean[name][selected[name]]:.4f})")

    # CN: **留出评估**：每个候选族的前 3 名（按验证）加上该族验证最优，全部在测试对上打分。
    # EN: **held-out evaluation**: every candidate is scored on the test pairs; we keep the per-family
    #     distribution plus the validation-selected one.
    test_scores: dict[str, list[float]] = {name: [] for name in families}
    per_pair_selected: dict[str, dict[str, float]] = {}
    for pair in test_pairs:
        ta = read_ply(root / f"tr_reg_{pair[0]:03d}.ply")
        tb = read_ply(root / f"tr_reg_{pair[1]:03d}.ply")
        geometry = f"pair{pair[0]:03d}-{pair[1]:03d}"
        t_index = shared_index(ta, int(config.n_sub), 0)
        da, db, geo, norm, mirror = pair_geometry(ta, tb, t_index)
        for name, thetas in families.items():
            test_scores[name].append(_score_all(da, db, thetas, geo, norm, mirror))
        for name in families:
            per_pair_selected.setdefault(geometry, {})[name] = test_scores[name][-1][selected[name]]
        del geo
        log.line(f"   {geometry}: " + "  ".join(
            f"{name}={per_pair_selected[geometry][name]:.3f}" for name in families))

    rows: list[RunRecord] = []
    log.line("\nfamily            val-selected held-out   | held-out distribution (min / median / max)")
    for name, thetas in families.items():
        held = np.array(test_scores[name], dtype=np.float64)  # (pairs, candidates)
        chosen = held[:, selected[name]]
        best_per_pair = held.min(axis=1)
        log.line(f"{name:<16} {chosen.mean():.4f}            | "
                 f"{held.min():.4f} / {np.median(held):.4f} / {held.max():.4f}   "
                 f"(oracle-per-pair {best_per_pair.mean():.4f})")
        rows.append(RunRecord(study=str(config.study), geometry="summary", arm=name, k=k,
                              m=int(config.n_sub), draw=0,
                              metrics={"selected_held_out": float(chosen.mean()),
                                       "family_min": float(held.min()),
                                       "family_median": float(np.median(held)),
                                       "family_max": float(held.max()),
                                       "per_pair_oracle": float(best_per_pair.mean()),
                                       "validation_selected": float(val_mean[name][selected[name]]),
                                       "validation_min": float(val_mean[name].min()),
                                       "family_size": float(len(thetas))},
                              config=config.to_dict(),
                              extra={"selected_index": selected[name],
                                     "selected_theta": thetas[selected[name]].tolist(),
                                     "test_scores": held.tolist(),
                                     "val_scores": val_mean[name].tolist()}))

    # CN: 配对 bootstrap：iOT 的验证最优 vs 其它族的验证最优，按测试对配对。
    contrasts = bootstrap_contrast(per_pair_selected, base="iot",
                                   others=[n for n in families if n != "iot"])
    for other, stats in contrasts.items():
        log.line(f"   iot - {other:<16} {stats['mean']:+.4f}  "
                 f"CI [{stats['ci_low']:+.4f}, {stats['ci_high']:+.4f}]  wins {int(stats['wins'])}/{int(stats['pairs'])}")
        rows.append(RunRecord(study=str(config.study), geometry="contrast", arm=f"iot_minus_{other}", k=k,
                              m=int(config.n_sub), draw=-1, metrics=stats, config=config.to_dict()))

    # CN: 验证分数能否预测测试分数（在随机族内部）——选择器本身是否有效。
    # EN: does validation predict test inside the random families? i.e. does the selector work at all.
    for name in ("random_support", "random_simplex"):
        val_flat = np.asarray(val_mean[name], dtype=np.float64)
        test_flat = np.asarray(test_scores[name], dtype=np.float64).mean(axis=0)
        spearman = float(np.corrcoef(np.argsort(np.argsort(val_flat)),
                                     np.argsort(np.argsort(test_flat)))[0, 1])
        log.line(f"   validation vs test inside {name}: Spearman {spearman:+.3f}")
        rows.append(RunRecord(study=str(config.study), geometry="selector", arm=name, k=k,
                              m=int(config.n_sub), draw=0, metrics={"spearman": spearman},
                              config=config.to_dict()))

    table = summarise(rows, group=("arm",), metrics=("selected_held_out",))
    log.line("\n" + str(table))
    write_records(run_dir / "records.json", rows, provenance=PROVENANCE)
    log.line(f"wrote {run_dir / 'records.json'} ({len(rows)} rows)")
    log.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
