"""
CN: **Phase C —— 松弛循环的规模行为**（GPU master + GPU scan，无 HiGHS）。

    CN: 对 m ∈ {600, 2000, 5000, 11263} 各跑一次，记录：轮数、`|F|`、`|F|/(m·n)`、每轮时间、
        峰值内存、最终证书上下界与状态。**每 个 fit 都报**，包括未达到 `U-L <= 1e-2*R_uniform` 的。

    CN: 特征来自 FIT 患者（全部切片，流式编码），K=64、G3、soft 窗。池化特征不落地 `m×n`：
        扫描由 GPU 后端完成，master 只作用于 `F`。

EN: **Phase C -- scaling of the relaxed loop** (GPU master + GPU scan, no HiGHS).

    For m in {600, 2000, 5000, 11263}: rounds, `|F|`, `|F|/(m*n)`, time per round, peak memory, final
    certificate bounds and status. **Every fit is reported**, including those missing
    `U - L <= 1e-2 * R_uniform`.
"""

from __future__ import annotations

import gc
import json
import resource
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import scipy.sparse as sparse

_REPO_ROOT = Path(__file__).resolve().parents[4]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from iot import InverseOTProblem  # noqa: E402
from iot.core.features import DiagonalWeightFeatures  # noqa: E402
from iot.relaxed.loop_regret import solve_inverse_ot_regret  # noqa: E402
from iot.relaxed.master_regret import RegretMasterSolver  # noqa: E402
from iot_experiments.harness.config import RunConfig, build_parser  # noqa: E402
from iot_experiments.harness.records import RunLog, format_table  # noqa: E402
from iot_experiments.harness.solver import box_simplex  # noqa: E402
from iot_experiments.harness.synthrad import discover_patients  # noqa: E402
from iot_experiments.harness.datasets import reduce_features, stream_features  # noqa: E402

PROVENANCE: dict[str, object] = {
    "purpose": "loop scaling in m with the GPU master and the GPU scan",
    "solver": "CuPDLPx for the master; GPU scan for violations; no HiGHS anywhere",
    "acceptance": "U - L <= 1e-2 * R_uniform",
    "features": "FIT patients, all slices, streamed; K=64, G3, soft CT window",
    "policy": "every fit is reported with its bracket width; misses are flagged, never dropped silently",
}


@dataclass(frozen=True)
class PhaseCConfig(RunConfig):
    """CN: Phase C 的旋钮。EN: Phase C's knobs."""

    section = "synthrad_phase_c"
    study: str = "synthrad_phase_c"
    ms: tuple[int, ...] = (600, 2000, 5000, 11263)
    k: int = 64

    @classmethod
    def add_study_arguments(cls, parser) -> None:
        """CN: 只加本阶段特有的参数。EN: only this phase's flags."""
        parser.add_argument("--ms", type=int, nargs="+", default=[600, 2000, 5000, 11263])
        parser.add_argument("--k", type=int, default=64)


def main(argv: list[str] | None = None) -> int:
    """CN: 入口。EN: entry point."""
    parser = build_parser(PhaseCConfig)
    args = parser.parse_args(list(argv) if argv is not None else None)
    config = PhaseCConfig.from_namespace(args, parser)
    log = RunLog(config.run_dir / "run.log")
    log.line(f"config: ms={config.ms} k={config.k} rel=0.25 master=cupdlpx scan=torch")

    split = json.loads((config.run_dir.parent / "synthrad_v2" / "split_v2.json").read_text())
    patients = discover_patients(_REPO_ROOT / "data" / "Task1", anatomy="brain")
    fit_pats = [p for p in patients if p.patient in set(split["fit"])]
    rng = np.random.default_rng(1000)
    chosen = [fit_pats[i] for i in rng.permutation(len(fit_pats))[:60]]
    log.line(f"streaming features for {len(chosen)} FIT patients (K={config.k}, G3, soft)")

    t0 = time.perf_counter()
    pm, pc, _pid, _z, _ctr = stream_features(chosen, window=(-200.0, 300.0), max_per_patient=None,
                                             min_gap=0, seed=0)
    ftr, ftc = reduce_features(pm, pc, k=int(config.k), geometry_name="G3", whiten=False)
    del pm, pc
    gc.collect()
    log.line(f"features ready: {ftr.shape} in {time.perf_counter() - t0:.1f}s")

    rows: list[dict[str, object]] = []
    misses = 0
    for m_target in config.ms:
        n_avail = int(ftr.shape[0])
        m = int(min(int(m_target), n_avail))
        sub = np.sort(np.random.default_rng(777).permutation(n_avail)[:m])
        x, y = ftr[sub], ftc[sub]
        k = int(x.shape[1])
        plan = sparse.csr_matrix((np.full(m, 1.0 / m), (np.arange(m), np.arange(m))), shape=(m, m))
        problem = InverseOTProblem(features=DiagonalWeightFeatures(x, y),
                                   theta_set=box_simplex(k, box_relative=0.25), plan=plan)
        rss0 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
        t_start = time.perf_counter()
        try:
            res = solve_inverse_ot_regret(
                problem, tol=None, gamma=1, max_iterations=200,
                master=RegretMasterSolver(backend="cupdlpx", tie_break=False,
                                          params={"time_sec_limit": float(config.solve_limit_s)}),
                eps_r_rel=1e-6, prune=False, check_pot=False, selection="none")
            elapsed = time.perf_counter() - t_start
            rss1 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
            width = float(res.upper - res.lower)
            ratio = width / max(float(res.upper), 1e-30)
            accepted = ratio <= 1e-2
            misses += 0 if accepted else 1
            per_round = [round(float(it.wall_time), 3) for it in res.iterations]
            rows.append({"m": m, "requested": int(m_target), "rounds": int(res.n_iterations),
                         "F": int(res.finite_edges), "F_frac": round(res.finite_edges / (m * m), 5),
                         "secs": round(elapsed, 1), "rss_gb": round(max(rss1, rss0), 2),
                         "L": f"{res.lower:.3e}", "U": f"{res.upper:.3e}",
                         "width_rel": f"{ratio:.2e}", "status": res.stall_status,
                         "accepted": accepted})
            log.line(f"m={m:>6} rounds={res.n_iterations:>3} |F|={res.finite_edges:>7} "
                     f"|F|/(mn)={res.finite_edges / (m * m):.4f} time={elapsed:6.1f}s "
                     f"per-round={per_round[:6]} rss={max(rss1, rss0):.1f}GB "
                     f"L={res.lower:.3e} U={res.upper:.3e} rel-width={ratio:.2e} "
                     f"status={res.stall_status} accepted={accepted}")
        except Exception as error:  # noqa: BLE001 - reported, never substituted
            elapsed = time.perf_counter() - t_start
            misses += 1
            rows.append({"m": m, "requested": int(m_target), "rounds": -1, "F": -1, "F_frac": -1.0,
                         "secs": round(elapsed, 1), "rss_gb": -1.0, "L": "-", "U": "-",
                         "width_rel": "-", "status": f"FAILED: {type(error).__name__}",
                         "accepted": False})
            log.line(f"m={m:>6} FAILED after {elapsed:.1f}s: {type(error).__name__}: {str(error)[:90]}")
        del x, y
        gc.collect()

    log.line("\n" + format_table(rows, ("requested", "m", "rounds", "F", "F_frac", "secs", "rss_gb",
                                        "L", "U", "width_rel", "accepted")))
    log.line(f"fits missing U-L <= 1e-2*R_uniform: {misses} of {len(rows)}")
    (config.run_dir / "phase_c.json").write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    log.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
