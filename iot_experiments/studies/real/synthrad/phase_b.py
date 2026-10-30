"""
CN: **Phase B —— 密集 LP 与松弛循环的定点对比**（m=300，真实 SynthRAD 特征，K=64，G3，soft 窗，1 draw）。

    CN: 目的：在**同一个问题**上比较两条路径：`fit_full_lp`（全边）与 `solve_inverse_ot_regret`（松弛循环）。
        报告：证书上下界 vs 密集 `R'`、`||theta_loop - theta_dense||_1`、留出集上的 exact 恢复。
        **若二者超出求解器噪声地不一致，就停下并报告**（这是本阶段的退出条件）。

    CN: 求解器状态必须如实报：`converged` / `stall_status` / `solver_floor` / `U - L`。
        Phase B 的验收规则与 Phase C 相同：`converged 且 U - L <= 1e-2 * R_uniform`，否则标记为 solver-limited。

EN: **Phase B -- dense LP vs relaxed loop, one spot check** (m=300, real SynthRAD features, K=64, G3, soft
    window, 1 draw).

    Compare `fit_full_lp` (all edges) against `solve_inverse_ot_regret` (relaxed loop) on the **same problem**:
    bracket vs dense `R'`, `||theta_loop - theta_dense||_1`, and held-out exact recovery. **If they disagree
    beyond solver noise, stop and report** -- that is this phase's exit condition.
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import scipy.sparse as sparse
from jaxtyping import Float, Int

_REPO_ROOT = Path(__file__).resolve().parents[4]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from iot import EdgeSet, InverseOTProblem  # noqa: E402
from iot.core.features import DiagonalWeightFeatures  # noqa: E402
from iot.relaxed.loop_regret import solve_inverse_ot_regret  # noqa: E402
from iot.relaxed.master_regret import RegretMasterSolver, regret_objective  # noqa: E402
from iot_experiments.harness.config import RunConfig, build_parser  # noqa: E402
from iot_experiments.harness.records import RunLog  # noqa: E402
from iot_experiments.harness.solver import box_simplex, diagonal_cost  # noqa: E402
from iot_experiments.harness.synthrad import discover_patients  # noqa: E402
from iot_experiments.studies.real.synthrad.recovery import (  # noqa: E402
    exact_ot,
    score_permutation,
)
from iot_experiments.harness.datasets import build_table, reduce_features, stream_features  # noqa: E402

PROVENANCE: dict[str, object] = {
    "purpose": "dense LP vs relaxed loop on the same inverse-OT problem",
    "config": "m=300, K=64, G3, soft window, 1 draw, rel=0.25",
    "acceptance": "converged and U - L <= 1e-2 * R_uniform, else solver-limited",
    "exit_rule": "if the two disagree beyond solver noise, stop and report",
}


def _dense(x: Float[np.ndarray, "m k"], y: Float[np.ndarray, "m k"], *, rel: float, solver: str,
           time_limit_s: float) -> tuple[Float[np.ndarray, "k"], float, float, str, int]:
    """
    CN: 全边 LP：返回 `(theta, R', R_uniform, status, iterations)`。
    EN: all-edges LP: returns `(theta, R', R_uniform, status, iterations)`.
    """
    m, k = int(x.shape[0]), int(x.shape[1])
    plan = sparse.csr_matrix((np.full(m, 1.0 / m), (np.arange(m), np.arange(m))), shape=(m, m))
    problem = InverseOTProblem(features=DiagonalWeightFeatures(x, y),
                               theta_set=box_simplex(k, box_relative=float(rel)), plan=plan)
    r: Int[np.ndarray, "m m"]
    c: Int[np.ndarray, "m m"]
    r, c = np.meshgrid(np.arange(m), np.arange(m), indexing="ij")
    edges = EdgeSet.from_arrays(r.reshape(-1), c.reshape(-1), n_target=m)
    uni = np.full(k, 1.0 / k, dtype=np.float64)
    cost_l = np.asarray(problem.features.costs(uni, r.reshape(-1), c.reshape(-1)),
                        dtype=np.float64).reshape(m, m)
    f_uni = cost_l.min(axis=1)
    g_uni = np.zeros(m, dtype=np.float64)
    r_uniform = float(regret_objective(problem) @ np.concatenate([uni, f_uni, g_uni]))
    master = RegretMasterSolver(backend=str(solver), tie_break=False,
                                params={"time_sec_limit": float(time_limit_s)})
    sol = master.solve(problem, edges)
    return (np.asarray(sol.theta, dtype=np.float64), float(sol.regret), float(r_uniform), "ok",
            int(getattr(sol, "iterations", -1)))


@dataclass(frozen=True)
class PhaseBConfig(RunConfig):
    """CN: Phase B 的旋钮。EN: Phase B's knobs."""

    section = "synthrad_phase_b"
    study: str = "synthrad_phase_b"
    m: int = 300
    k: int = 64
    geometry: str = "G3"
    rel: float = 0.25
    draw: int = 0

    @classmethod
    def add_study_arguments(cls, parser) -> None:
        """CN: 只加本阶段特有的参数。EN: only this phase's flags."""
        parser.add_argument("--m", type=int, default=300)


def main(argv: list[str] | None = None) -> int:
    """CN: 入口。EN: entry point."""
    parser = build_parser(PhaseBConfig)
    args = parser.parse_args(list(argv) if argv is not None else None)
    config = PhaseBConfig.from_namespace(args, parser)
    log = RunLog(config.run_dir / "run.log")
    log.line(f"config: m={config.m} k={config.k} geometry={config.geometry} rel={config.rel} "
             f"draw={config.draw}")

    split = json.loads((config.run_dir.parent / "synthrad_v2" / "split_v2.json").read_text())
    patients = discover_patients(_REPO_ROOT / "data" / "Task1", anatomy="brain")
    fit_pats = [p for p in patients if p.patient in set(split["fit"])]
    dev_pats = [p for p in patients if p.patient in set(split["dev"])]
    rng = np.random.default_rng(1000 + int(config.draw))
    chosen = [fit_pats[i] for i in rng.permutation(len(fit_pats))[:60]]
    log.line(f"streaming features for {len(chosen)} fit patients (K={config.k}, {config.geometry})")

    pooled_mr, pooled_ct, _pid, _z, _ctr = stream_features(chosen, window=(-200.0, 300.0),
                                                           max_per_patient=None, min_gap=0,
                                                           seed=int(config.draw))
    ftr_full, ftc_full = reduce_features(pooled_mr, pooled_ct, k=int(config.k),
                                         geometry_name=str(config.geometry), whiten=False)
    # CN: 拟合特征已约简，释放原始 6144 维池化特征（每侧可达 0.28 GB）以腾出内存给 dev 流。
    # EN: the fit features are reduced; release the raw 6144-dim pooled features (up to 0.28 GB per side) to
    #     leave room for the dev stream.
    del pooled_mr, pooled_ct
    import gc as _gc  # noqa: PLC0415
    _gc.collect()
    sub = np.sort(np.random.default_rng(777).permutation(ftr_full.shape[0])[: int(config.m)])
    x, y = ftr_full[sub], ftc_full[sub]
    log.line(f"fit problem: {x.shape[0]} slices x {x.shape[1]} features (sampled from "
             f"{ftr_full.shape[0]})")

    # CN: 两条路径解**同一个问题**。
    # EN: both paths solve the **same problem**.
    t0 = time.perf_counter()
    theta_d, r_dense, r_uniform, status_d, iters_d = _dense(x, y, rel=float(config.rel),
                                                           solver=str(config.solver),
                                                           time_limit_s=float(config.solve_limit_s))
    dense_s = time.perf_counter() - t0

    m = int(x.shape[0]); k = int(x.shape[1])
    plan = sparse.csr_matrix((np.full(m, 1.0 / m), (np.arange(m), np.arange(m))), shape=(m, m))
    problem = InverseOTProblem(features=DiagonalWeightFeatures(x, y),
                               theta_set=box_simplex(k, box_relative=float(config.rel)), plan=plan)
    t0 = time.perf_counter()
    loop = solve_inverse_ot_regret(problem, tol=None, gamma=1, max_iterations=200,
                                   master=RegretMasterSolver(backend=str(config.solver),
                                                             tie_break=False,
                                                             params={"time_sec_limit":
                                                                     float(config.solve_limit_s)}),
                                   eps_r_rel=1e-6, prune=False, check_pot=False)
    loop_s = time.perf_counter() - t0

    theta_l = np.asarray(loop.theta, dtype=np.float64)
    l1 = float(np.abs(theta_l - theta_d).sum())
    accepted = bool(loop.converged and (loop.upper - loop.lower) <= 1e-2 * max(r_uniform, 1e-30))
    log.line(f"dense : R'={r_dense:.6e} R_uni={r_uniform:.6e} iters={iters_d} time={dense_s:.2f}s")
    log.line(f"loop  : L={loop.lower:.6e} U={loop.upper:.6e} gap={loop.gap:.3e} rounds={loop.n_iterations} "
             f"converged={loop.converged} stall={loop.stall_status} floor={loop.solver_floor:.3e} "
             f"|F|={loop.finite_edges} time={loop_s:.2f}s accepted={accepted}")
    log.line(f"||theta_loop - theta_dense||_1 = {l1:.6e}")
    log.line("per-round: " + " | ".join(
        f"{r.index}: U={r.upper:.3e} gap={r.gap:.2e} |F|={r.finite_edges} {r.wall_time:.2f}s"
        for r in loop.iterations[:12]))

    # CN: 留出集上的 exact 恢复：同一个 dev 池，两种 θ 各解一次 OT。
    # EN: held-out exact recovery: the same dev pool, one OT solve per theta.
    pool = build_table(dev_pats, window=(-200.0, 300.0), max_per_patient=14, min_gap=8, seed=10_000)
    _pm, _pc = stream_features(dev_pats, window=(-200.0, 300.0), max_per_patient=14, min_gap=8,
                               seed=10_000)[:2]
    p_mr, p_ct = reduce_features(_pm, _pc, k=int(config.k), geometry_name=str(config.geometry),
                                 whiten=False)
    del _pm, _pc
    order = np.random.default_rng(50_000).permutation(len(pool))
    pat_cols = [pool.patient[i] for i in order]
    z_cols = pool.z_index[order]
    ftc_s = p_ct[order]
    for name, theta in (("dense", theta_d), ("loop", theta_l)):
        perm = exact_ot(diagonal_cost(p_mr, ftc_s, theta))
        met = score_permutation(perm, patient_rows=pool.patient, patient_cols=pat_cols,
                                z_rows=pool.z_index, z_cols=z_cols)
        log.line(f"held-out exact recovery [{name}]: exact_hit={met['exact_hit']:.4f} "
                 f"patient_hit={met['patient_hit']:.4f}")
    log.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
