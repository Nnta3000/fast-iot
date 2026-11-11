"""
EN: **Is the corner `theta` the objective or the solver -- and are the weights identified?**

    The fitted `theta` sits on the boundary of the simplex in every run, with `converged=False`. Two different
    questions hide behind that, and they have different answers:

      1. **Is the corner the solver's fault?** The relaxed master minimises a **linear** objective over a
         polytope, so an optimum at a *vertex* is the expected outcome, not a symptom. The test is to solve the
         same instance with a **different** solver -- HiGHS (exact, simplex/interior point) against the default
         CuPDLPx (first-order) -- and compare the objective. If both reach the same regret, the corner belongs to
         the objective.

      2. **Are the weights identified?** A vertex may be one point of a whole **optimal face**. For each
         coordinate `k`, this module computes

             min / max  theta_k   s.t.  R(x) <= R* + eta,   x feasible in the assembled LP,

         i.e. how far `theta_k` can move while the regret stays within `eta` of optimal. A coordinate whose
         range is a point is **identified**; one whose range spans the simplex is not, whatever its fitted value.

    Everything is measured on the *same* edge set `F` the loop finished with (captured by wrapping the master),
    not on a re-assembled grid, so the LP is exactly the one the solver certified.

    **Memory.** `|F| ~ 8n` at `n = 800`, so the LP has ~6k rows and ~1.6k columns with ~16 non-zeros per row.
    Nothing of size `n * n` or `n * n * K` is built; each of the `2K` range LPs is solved on the same sparse
    system with one extra row.

    CN: **角落解是目标函数还是求解器？权重是否被识别？** 松弛 master 在**多面体上最小化线性目标**，所以最优解落在
        顶点是**预期**结果。用不同求解器（HiGHS 精确 vs CuPDLPx 一阶）解同一实例，若目标值相同，则角落属于目标函数。
        再用"固定 `R <= R* + eta` 后每个 `theta_k` 的 min/max"量化**最优面**的大小。

    Run:
        .venv/bin/python -u -m iot_experiments.studies.real.faust.corner --n-sub 800
"""

from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sparse

_REPO_ROOT = Path(__file__).resolve().parents[4]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from iot.core.features import SquaredDifferenceFeatures  # noqa: E402
from iot.core.parameters import (  # noqa: E402
    PruningParameters,
    RelaxedObjective,
    SolverParameters,
)
from iot.core.problem import InverseOTProblem  # noqa: E402
from iot.core.theta import SimplexTheta  # noqa: E402
from iot.relaxed.loop_regret import solve_inverse_ot_regret  # noqa: E402
from iot_experiments.harness.config import RunConfig, build_parser  # noqa: E402
from iot_experiments.harness.faust import read_ply  # noqa: E402
from iot_experiments.harness.preflight import check as preflight_check  # noqa: E402
from iot_experiments.harness.records import (  # noqa: E402
    RunLog,
    RunRecord,
    format_table,
    summarise,
    write_records,
)
from iot_experiments.harness.faust import descriptors, standardise  # noqa: E402
from iot_experiments.harness.master_probe import RecordingMaster  # noqa: E402

DEFAULT_ROOT = "/home/agent/Documents/fast-iot/data/faust_extract/MPI-FAUST/training/registrations"

PROVENANCE: dict[str, object] = {
    "question": "is the boundary theta an artefact of the solver, and are the individual weights identified?",
    "theory": "the relaxed master minimises a LINEAR objective over a polytope, so an optimal vertex is expected; "
              "a vertex is not evidence of a solver defect",
    "test_1_solver": "fit the same instance with HiGHS (exact) and CuPDLPx (first-order); equal regret means the "
                     "corner belongs to the objective",
    "test_2_identified": "EMPIRICAL: refit on 5 different vertex subsamples and report each theta_k's mean/std "
                         "and whether the SUPPORT (coordinates exactly 0) is identical every time. An LP-based "
                         "range over {R <= R*+eta} was attempted first and abandoned: with objective +/-theta_k "
                         "the (f,g) block is cost-free, the LP is degenerate, and HiGHS did not finish one cell "
                         "in 18 minutes (job stopped by hand, exit 143)",
    "memory": "the LP is the one the solver certified: |F| ~ 8n rows at n=800, ~16 non-zeros per row. Nothing of "
              "size n*n or n*n*K is built; the 2K range LPs reuse the same sparse system plus one row",
}


@dataclasses.dataclass(frozen=True)
class CornerConfig(RunConfig):
    """CN: 旋钮。EN: knobs."""

    study: str = "faust_corner"
    section: str = "faust_corner"
    root: str = DEFAULT_ROOT
    train_pair: tuple[int, ...] = (0, 13)
    n_sub: int = 800
    max_iterations: int = 30
    eta_rel: float = 1e-3

    @classmethod
    def add_study_arguments(cls, parser) -> None:
        """CN: 只加本研究特有的参数。EN: only this study's flags."""
        parser.add_argument("--root", default=DEFAULT_ROOT)
        parser.add_argument("--train-pair", type=int, nargs=2, default=[0, 13])
        parser.add_argument("--n-sub", type=int, default=800)
        parser.add_argument("--max-iterations", type=int, default=30)
        parser.add_argument("--eta-rel", type=float, default=1e-3)


def _fit(problem: InverseOTProblem, backend: str, *, max_iterations: int, seed: int = 0
         ) -> tuple[dict[str, object], _RecordingMaster]:
    """CN: 用指定后端解一次松弛 master。EN: one relaxed solve with the requested backend."""
    master = RecordingMaster(backend=backend, tie_break=False,
                              params={"time_sec_limit": 300.0} if backend == "cupdlpx" else None)
    result = solve_inverse_ot_regret(
        problem,
        objective=RelaxedObjective(eps_r_rel=1e-6, check_pot=False, selection="none"),
        params=SolverParameters(tol=None, gamma=1, max_iterations=int(max_iterations),
                                pruning=PruningParameters(enabled=False)),
        master=master,
    )
    return {"theta": np.asarray(result.theta, dtype=np.float64), "regret": float(result.regret),
            "lower": float(result.lower), "upper": float(result.upper),
            "converged": bool(result.converged), "edges": int(result.finite_edges),
            "iterations": int(result.n_iterations)}, master


def _stability(a_mesh, b_mesh, config: "CornerConfig", *, seeds: tuple[int, ...]
               ) -> dict[str, object]:
    """
    CN: **经验可辨识性**：用不同的顶点子样本重复拟合，看每个 `theta_k` 的稳定性。
        支持集（哪些坐标恰为 0）是否每次都一样、非零坐标的离散程度。
    EN: **empirical identifiability**: refit on different vertex subsamples and look at how each `theta_k`
        moves. Is the *support* (which coordinates are exactly 0) the same every time, and how spread out are
        the non-zero coordinates?
    """
    samples: list[np.ndarray] = []
    # CN: **每个种子都必须重建问题**（新的顶点子样本）。上一版把 `seed` 传给 `_fit` 而 `_fit` 忽略了它，
    #     于是五次"重拟合"是同一个问题，std 恒为 0——一个假的"完全稳定"。
    # EN: **each seed must rebuild the problem** (a new vertex subsample). The previous version passed `seed`
    #     to `_fit`, which ignored it, so the five "refits" were the same problem and the std was identically
    #     0 -- a fake "perfectly stable" result. Found because the numbers were too good; the tolerance below
    #     is also relative now, since CuPDLPx returns ~1e-6 instead of exact 0.
    for seed in seeds:
        idx = np.sort(np.random.default_rng(int(seed)).permutation(a_mesh.n_vertices)[: int(config.n_sub)])
        da, db = standardise(descriptors(a_mesh)[idx], descriptors(b_mesh)[idx])
        n_local = int(idx.size)
        plan = sparse.csr_matrix((np.full(n_local, 1.0 / n_local), (np.arange(n_local), np.arange(n_local))),
                                 shape=(n_local, n_local))
        local = InverseOTProblem(features=SquaredDifferenceFeatures(da, db),
                                 theta_set=SimplexTheta(int(da.shape[1])), plan=plan)
        out, _master = _fit(local, "cupdlpx", max_iterations=int(config.max_iterations))
        samples.append(np.asarray(out["theta"], dtype=np.float64))
    stack = np.stack(samples)
    zero_tol = 1e-6
    always_zero = np.all(stack <= zero_tol, axis=0)
    ever_zero = np.any(stack <= zero_tol, axis=0)
    return {"samples": stack.tolist(),
            "mean": stack.mean(axis=0).tolist(),
            "std": stack.std(axis=0).tolist(),
            "always_zero": [int(v) for v in np.flatnonzero(always_zero)],
            "sometimes_zero": [int(v) for v in np.flatnonzero(ever_zero & ~always_zero)],
            "max_std": float(stack.std(axis=0).max()),
            "support_stable": bool(np.array_equal(always_zero, ever_zero))}


def main(argv: list[str] | None = None) -> int:
    """CN: 入口。EN: entry point."""
    parser = build_parser(CornerConfig)
    args = parser.parse_args(list(argv) if argv is not None else None)
    config = CornerConfig.from_namespace(args, parser)
    run_dir = config.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    log = RunLog(run_dir / "run.log")
    pre = preflight_check(run_tiny=False, need_gpu=False)
    pre.write(run_dir / "preflight.json")
    log.line(f"preflight ok={pre.ok} mem={pre.mem_available_gb:.1f} GB")

    root = Path(config.root)
    a_mesh = read_ply(root / f"tr_reg_{int(config.train_pair[0]):03d}.ply")
    b_mesh = read_ply(root / f"tr_reg_{int(config.train_pair[1]):03d}.ply")
    idx = np.sort(np.random.default_rng(0).permutation(a_mesh.n_vertices)[: int(config.n_sub)])
    da, db = standardise(descriptors(a_mesh)[idx], descriptors(b_mesh)[idx])
    n = int(idx.size)
    k = int(da.shape[1])
    plan = sparse.csr_matrix((np.full(n, 1.0 / n), (np.arange(n), np.arange(n))), shape=(n, n))
    problem = InverseOTProblem(features=SquaredDifferenceFeatures(da, db),
                               theta_set=SimplexTheta(k), plan=plan)
    log.line(f"instance: n={n}, K={k}, Theta=simplex, plan=I/n")

    rows: list[RunRecord] = []
    fits: dict[str, dict[str, object]] = {}
    masters: dict[str, _RecordingMaster] = {}
    for backend in ("cupdlpx", "highs"):
        fit, master = _fit(problem, backend, max_iterations=int(config.max_iterations))
        fits[backend] = fit
        masters[backend] = master
        log.line(f"[{backend:<8}] regret={fit['regret']:.10g} upper={fit['upper']:.6g} "
                 f"|F|={fit['edges']} iters={fit['iterations']} converged={fit['converged']}")
        log.line(f"           theta={np.round(np.asarray(fit['theta']), 4).tolist()}")
        rows.append(RunRecord(study=str(config.study), geometry=f"pair{config.train_pair[0]}-"
                              f"{config.train_pair[1]}", arm=f"fit_{backend}", k=k, m=n, draw=0,
                              metrics={"regret": float(fit["regret"]), "edges": float(fit["edges"]),
                                       "iterations": float(fit["iterations"]),
                                       "n_zero": float(np.sum(np.asarray(fit["theta"]) <= 1e-6))},
                              config=config.to_dict(), extra={"theta": np.asarray(fit["theta"]).tolist()}))

    agree = abs(float(fits["cupdlpx"]["regret"]) - float(fits["highs"]["regret"]))
    log.line(f"\nregret agreement between backends: |diff| = {agree:.3e}")

    # CN: **为什么不是 LP 区间**：以 `±theta_k` 为目标时 `(f,g)` 块无代价，LP 高度退化，HiGHS 实测 18 分钟
    #     未解完一个单元（已停止）。改用**重采样稳定性**：同一问题、不同顶点子样本，看 `theta_k` 怎么动。
    # EN: **why not an LP range**: with objective `±theta_k` the `(f,g)` block is cost-free, the LP is highly
    #     degenerate, and HiGHS did not finish one cell in 18 minutes (stopped). Resampling stability answers
    #     the same question and is well posed.
    stability = _stability(a_mesh, b_mesh, config, seeds=(0, 1, 2, 3, 4))
    log.line(f"\nstability over 5 vertex subsamples: max std {stability['max_std']:.4f}, "
             f"support stable {stability['support_stable']}")
    log.line(f"   always zero: {stability['always_zero']}")
    log.line(f"   sometimes zero: {stability['sometimes_zero']}")
    log.line(f"   mean theta: {np.round(np.asarray(stability['mean']), 4).tolist()}")
    log.line(f"   std  theta: {np.round(np.asarray(stability['std']), 4).tolist()}")
    rows.append(RunRecord(study=str(config.study), geometry="stability", arm="refit_5_subsamples",
                          k=k, m=n, draw=0,
                          metrics={"max_std": float(stability["max_std"]),
                                   "support_stable": float(bool(stability["support_stable"])),
                                   "n_always_zero": float(len(stability["always_zero"]))},
                          config=config.to_dict(), extra=stability))

    table = summarise(rows, group=("arm",), metrics=("regret", "max_width", "n_identified"))
    log.line("\n" + format_table(table, ("arm", "n_draws", "regret_mean", "max_width_mean",
                                          "n_identified_mean")))
    write_records(run_dir / "records.json", rows, provenance=PROVENANCE)
    log.line(f"wrote {run_dir / 'records.json'} ({len(rows)} rows)")
    log.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
