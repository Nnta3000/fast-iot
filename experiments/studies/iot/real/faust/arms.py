"""
EN: **Four objectives, one hypothesis class -- which part of the relaxed LP actually earns its keep?**

    Every arm here fits the *same* object: `c_ij = sum_k theta_k (a_ik - b_jk)^2`, `theta >= 0`, on the same
    standardised descriptors and the same training pair. Only the **objective** differs:

      | arm | objective | marginals it sees |
      | --- | --- | --- |
      | `row_softmax` | `mean_i CE(-C_ij/tau, i)` | source only (one-sided) |
      | `rowcol_softmax` | `mean_i CE(log P_ii, i)`, `P = Sinkhorn(exp(-C/tau))` | source **and** target |
      | `entropic_iot` | minimise the entropic joint master `Psi(theta, f, g)` | source and target, exact |
      | `relaxed_lp` | minimise the regret of the relaxed master | source and target, exact, LP |

    plus `uniform` as the label-free reference.

    **Why these four.** The pairing study (`SANITY_CHECKS_REPORT.md`) never separated "inverse OT" from "a
    supervised reweighting of the same features". Here the features are identical, so a difference between the
    columns is attributable to the objective alone. In particular `row_softmax` is the classic InfoNCE-style
    arm: if it matches the LP, the LP's structure is not what is doing the work.

    **Fairness of comparison.** Each arm's hyperparameter (`tau` for the two softmax arms, `rel_eps` for the
    entropic arm) is chosen on **validation pairs drawn from the training subjects**, never on the test pairs;
    the relaxed LP has no hyperparameter once the box is dropped (simplex-only, as selected in the main study).
    All arms are then evaluated on the same held-out test pairs with the same geodesic metric.

    CN: **四个目标、同一个假设类**：代价形式、特征标准化、训练对完全相同，只有**目标函数**不同。`tau` 与
        `rel_eps` 只在**训练受试者**的验证对上选，测试对从不参与选择。

    Run:
        .venv/bin/python -u -m experiments.studies.iot.real.faust.arms --n-sub 2000
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
from experiments.harness.config import RunConfig, build_parser  # noqa: E402
from experiments.harness.faust import default_registrations_root, read_ply  # noqa: E402
from experiments.harness.preflight import check as preflight_check  # noqa: E402
from experiments.harness.records import (  # noqa: E402
    RunLog,
    RunRecord,
    format_table,
    summarise,
    write_records,
)
from experiments.harness.faust_match import fit_softmax  # noqa: E402
from experiments.studies.iot.real.faust.correspondence import (  # noqa: E402
    _evaluate_pair,
    _PairCache,
    _sample_index,
    default_pairs,
    descriptors,
    fit_theta,
    standardise,
)
from experiments.studies.iot.real.t1t2.entropic_metric.run import fit_entropic  # noqa: E402

DEFAULT_ROOT = str(default_registrations_root())

PROVENANCE: dict[str, object] = {
    "question": "with the hypothesis class held fixed, does the relaxed LP's structure beat a plain supervised "
                "reweighting?",
    "hypothesis_class": "c_ij = sum_k theta_k (a_ik - b_jk)^2, theta on the simplex, same standardised "
                        "descriptors (K=13) in every arm",
    "arms": "row_softmax (one-sided CE), rowcol_softmax (Sinkhorn CE, both marginals), entropic_iot (entropic "
            "joint master), relaxed_lp (regret master), uniform (label-free reference)",
    "selection": "tau in {0.01,0.05,0.1,0.5} and rel_eps in {0.05,0.1,0.25,0.5} chosen on VALIDATION pairs from "
                 "the TRAINING subjects; the relaxed LP is simplex-only and has no hyperparameter",
    "solver": "row_softmax / rowcol_softmax: torch autograd on a softmax parameterisation, Adam; entropic_iot: "
              "SLSQP from iot.entropic.joint_reference; relaxed_lp: the regret loop (CuPDLPx)",
    "metric": "geodesic error on the target, normalised by sqrt(surface area) -- identical to the main study",
    "memory": "n=2000 keeps the entropic arm's (m*m*K) feature tensor at 416 MB; at n=6890 it would be 4.9 GB",
}


@dataclasses.dataclass(frozen=True)
class ArmsConfig(RunConfig):
    """CN: 旋钮。EN: knobs."""

    study: str = "faust_arms"
    section: str = "faust_arms"
    root: str = DEFAULT_ROOT
    train_pair: tuple[int, ...] = (0, 13)
    train_subjects: tuple[int, ...] = (0, 1, 2, 3, 4)
    test_subjects: tuple[int, ...] = (5, 6, 7, 8, 9)
    test_pair_count: int = 8
    val_pair_count: int = 3
    n_sub: int = 2000
    max_iterations: int = 30
    steps: int = 600
    lr: float = 0.05
    tau_grid: tuple[float, ...] = (0.01, 0.05, 0.1, 0.5)
    rel_eps_grid: tuple[float, ...] = (0.05, 0.1, 0.25, 0.5)
    sinkhorn_iterations: int = 50
    master_backend: str = "auto"

    @classmethod
    def add_study_arguments(cls, parser) -> None:
        """CN: 只加本研究特有的参数。EN: only this study's flags."""
        parser.add_argument("--root", default=DEFAULT_ROOT)
        parser.add_argument("--train-pair", type=int, nargs=2, default=[0, 13])
        parser.add_argument("--train-subjects", type=int, nargs="+", default=[0, 1, 2, 3, 4])
        parser.add_argument("--test-subjects", type=int, nargs="+", default=[5, 6, 7, 8, 9])
        parser.add_argument("--test-pair-count", type=int, default=8)
        parser.add_argument("--val-pair-count", type=int, default=3)
        parser.add_argument("--n-sub", type=int, default=2000)
        parser.add_argument("--max-iterations", type=int, default=30)
        parser.add_argument("--steps", type=int, default=600)
        parser.add_argument("--lr", type=float, default=0.05)
        parser.add_argument("--tau-grid", type=float, nargs="+", default=[0.01, 0.05, 0.1, 0.5])
        parser.add_argument("--rel-eps-grid", type=float, nargs="+", default=[0.05, 0.1, 0.25, 0.5])
        parser.add_argument("--sinkhorn-iterations", type=int, default=50)
        parser.add_argument("--master-backend", choices=("auto", "cupdlpx", "highs"), default="auto")


def _resolve_master_backend(requested: str) -> str:
    """
    EN: `auto` -> `cupdlpx` only when a CUDA device is actually usable, else `highs`.

        The naive check is not enough: `cupdlpx_available()` (`iot/backends/lp_common.py`) only tests whether
        the `pycupdlpx` extension **imports**, which it does on a machine whose driver is too old for its torch
        build. `RegretMasterSolver` then constructs happily and the failure happens inside `solve()`, after the
        whole hyperparameter selection has been paid for (measured on a driver-11.4 host: `torch.cuda.is_available()`
        False, `import pycupdlpx` fine). So gate on the device, not the import.

        `highs` is the exact CPU simplex. For this study it is not a downgrade: the question is about the *set*
        of optimal thetas, which is what `fit_theta`'s own docstring recommends `highs` for. It is recorded in the
        run log and in the `chosen` record, because the shipped numbers used the first-order GPU path.

    CN: `auto` 只在 CUDA 设备**真的可用**时选 `cupdlpx`，否则用 `highs`。`cupdlpx_available()` 只检查扩展能否
        **导入**，在驱动过旧的机器上也会返回 True，于是构造成功、到 `solve()` 才炸（实测驱动 11.4：torch 无 CUDA、
        `import pycupdlpx` 正常）。所以要按设备而不是按导入判断。`highs` 是精确 CPU 单纯形，对本研究不是降级。
    """
    choice = str(requested)
    if choice in {"cupdlpx", "highs"}:
        return choice
    try:
        import torch  # noqa: PLC0415  # only to ask whether a device is usable
    except ImportError:
        return "highs"
    return "cupdlpx" if torch.cuda.is_available() else "highs"


def _fit_relaxed(a: Float[np.ndarray, "n k"], b: Float[np.ndarray, "n k"], *, max_iterations: int,
                 backend: str) -> Float[np.ndarray, "k"] | None:
    """CN: 我们的松弛 LP（simplex-only）。EN: our relaxed LP (simplex-only)."""
    fit = fit_theta(SquaredDifferenceFeatures(a, b), rel=None, max_iterations=int(max_iterations),
                    backend=str(backend))
    return None if fit["theta"] is None else np.asarray(fit["theta"], dtype=np.float64)


def main(argv: list[str] | None = None) -> int:
    """CN: 入口。EN: entry point."""
    parser = build_parser(ArmsConfig)
    args = parser.parse_args(list(argv) if argv is not None else None)
    config = ArmsConfig.from_namespace(args, parser)
    run_dir = config.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    log = RunLog(run_dir / "run.log")
    pre = preflight_check(run_tiny=False, need_gpu=False)
    pre.write(run_dir / "preflight.json")
    log.line(f"preflight ok={pre.ok} mem={pre.mem_available_gb:.1f} GB")
    master_backend = _resolve_master_backend(str(config.master_backend))
    log.line(f"master backend: requested={config.master_backend} resolved={master_backend}")
    if master_backend == "highs" and str(config.master_backend) == "auto":
        log.line("   note: no usable CUDA device, so the relaxed-LP master runs on the exact CPU simplex; "
                 "the arms comparison is unaffected but this differs from the shipped first-order GPU runs")

    root = Path(config.root)
    train_pair = (int(config.train_pair[0]), int(config.train_pair[1]))
    test_pairs = default_pairs(tuple(int(v) for v in config.test_subjects), int(config.test_pair_count))
    pool = default_pairs(tuple(int(v) for v in config.train_subjects), 4 * int(config.val_pair_count) + 16)
    val_pairs = [p for p in pool if p != train_pair][: int(config.val_pair_count)]
    log.line(f"train pair {train_pair}; validation {val_pairs}; test {test_pairs}")

    a_mesh = read_ply(root / f"tr_reg_{train_pair[0]:03d}.ply")
    b_mesh = read_ply(root / f"tr_reg_{train_pair[1]:03d}.ply")
    idx = _sample_index(a_mesh, config, 0)
    a, b = standardise(descriptors(a_mesh)[idx], descriptors(b_mesh)[idx])
    k = int(a.shape[1])
    log.line(f"train fit: n={idx.size}, K={k}")

    # EN: read the validation meshes ONCE (they were re-read on every score call) and share one preparation cache,
    #     so the tau / rel_eps sweep pays the descriptors + geodesics once per pair instead of once per candidate.
    # CN: 验证网格只读一次，并共用一个准备缓存；超参数扫描因此每对只付出一次描述子与测地开销。
    val_meshes = [
        (read_ply(root / f"tr_reg_{first:03d}.ply"), read_ply(root / f"tr_reg_{second:03d}.ply"))
        for first, second in val_pairs
    ]
    pair_cache: dict[tuple[str, str, int], _PairCache] = {}

    def validation_score(theta: Float[np.ndarray, "k"]) -> float:
        """CN: 在训练受试者的验证对上算 miss@5%。EN: miss@5 % on the training-subject validation pairs."""
        scores = []
        for pair, (va, vb) in zip(val_pairs, val_meshes, strict=True):
            rows = _evaluate_pair(va, vb, config=config, thetas={"cand": theta}, seed=0, baselines=False,
                                  cache=pair_cache)
            scores.append(rows[0].metrics["miss_5pct"])
        return float(np.mean(scores))

    log.line("\n--- hyperparameter selection on training-subject validation pairs ---")
    best_tau: dict[str, float] = {}
    for name, two_sided in (("row_softmax", False), ("rowcol_softmax", True)):
        scored: list[tuple[float, float]] = []
        for tau in config.tau_grid:
            theta = fit_softmax(a, b, tau=float(tau), two_sided=two_sided, steps=int(config.steps),
                                lr=float(config.lr), sinkhorn_iterations=int(config.sinkhorn_iterations))
            score = validation_score(theta)
            scored.append((float(tau), score))
            log.line(f"   {name:<16} tau={tau:g}  val miss@5%={score:.4f}")
        best_tau[name] = min(scored, key=lambda t: t[1])[0]

    best_eps = float(config.rel_eps_grid[0])
    entropic_scored: list[tuple[float, float]] = []
    for rel_eps in config.rel_eps_grid:
        theta, _diag = fit_entropic(a, b, rel_eps=float(rel_eps),
                                    max_iterations=int(config.max_iterations))
        score = validation_score(np.asarray(theta, dtype=np.float64))
        entropic_scored.append((float(rel_eps), score))
        log.line(f"   {'entropic_iot':<16} rel_eps={rel_eps:g}  val miss@5%={score:.4f}")
    best_eps = min(entropic_scored, key=lambda t: t[1])[0]
    log.line(f"selected: tau(row)={best_tau['row_softmax']:g}, tau(rowcol)={best_tau['rowcol_softmax']:g}, "
             f"rel_eps={best_eps:g}")

    log.line("\n--- refit on the training pair ---")
    thetas: dict[str, Float[np.ndarray, "k"]] = {
        "uniform": np.full(k, 1.0 / k),
        "row_softmax": fit_softmax(a, b, tau=best_tau["row_softmax"], two_sided=False,
                                   steps=int(config.steps), lr=float(config.lr),
                                   sinkhorn_iterations=int(config.sinkhorn_iterations)),
        "rowcol_softmax": fit_softmax(a, b, tau=best_tau["rowcol_softmax"], two_sided=True,
                                      steps=int(config.steps), lr=float(config.lr),
                                      sinkhorn_iterations=int(config.sinkhorn_iterations)),
    }
    theta_e, _diag = fit_entropic(a, b, rel_eps=best_eps, max_iterations=int(config.max_iterations))
    thetas["entropic_iot"] = np.asarray(theta_e, dtype=np.float64)
    theta_lp = _fit_relaxed(a, b, max_iterations=int(config.max_iterations), backend=master_backend)
    if theta_lp is None:
        log.line("relaxed_lp FAILED; dropping the arm")
    else:
        thetas["relaxed_lp"] = theta_lp
    for name, theta in thetas.items():
        log.line(f"   {name:<16} theta={np.round(theta, 4).tolist()}  "
                 f"L1-to-uniform={float(np.abs(theta - 1.0 / k).sum()):.3f}")

    log.line("\n--- evaluation on held-out test pairs ---")
    rows: list[RunRecord] = []
    for pair in test_pairs:
        va = read_ply(root / f"tr_reg_{pair[0]:03d}.ply")
        vb = read_ply(root / f"tr_reg_{pair[1]:03d}.ply")
        rows.extend(_evaluate_pair(va, vb, config=config, thetas=thetas, seed=0, baselines=False,
                                   cache=pair_cache))
    rows.append(RunRecord(study=str(config.study), geometry="selection", arm="chosen",
                          k=k, m=int(idx.size), draw=0,
                          metrics={"tau_row": best_tau["row_softmax"],
                                   "tau_rowcol": best_tau["rowcol_softmax"],
                                   "rel_eps": best_eps},
                          # EN: the master backend is part of the config of any relaxed_lp number, so it is
                          #     recorded on the selection row as well as in the log.
                          # CN: master 后端是 relaxed_lp 任何数字的配置之一，因此在选择行与日志里都记录。
                          extra={"master_backend": master_backend,
                                 "master_backend_requested": str(config.master_backend)},
                          config=config.to_dict()))

    table = summarise(rows, group=("arm",), metrics=("mean", "miss_1pct", "miss_5pct"))
    log.line("\n" + format_table(table, ("arm", "n_draws", "mean_mean", "miss_1pct_mean", "miss_5pct_mean")))
    write_records(run_dir / "records.json", rows, provenance=PROVENANCE)
    log.line(f"wrote {run_dir / 'records.json'} ({len(rows)} rows)")
    log.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
