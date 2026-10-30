"""
CN: **Stage 0 —— 层级有没有杠杆？** 只测量，不实现层级。

    CN: 问题：每一轮的时间里，是 **master LP** 占大头，还是 **分离扫描** 占大头？
          - 扫描占大头 ⇒ 减少轮数不是唯一手段，做更快的扫描比做层级更划算；
          - LP 占大头   ⇒ 轮数是唯一杠杆，层级值得一试。
        这是**枢轴问题**，不是决策：决策在 Stage 1（预注册的 0.70× 时间判据）。

    CN: 实例：SynthRAD，δ* ≈ 0 的真实情形（这正是决策所依据的 regime）。
        每个 m 记录：轮数、|F|、每轮的 master/scan 时间、LP 占比、RSS。

EN: **Stage 0 -- does the hierarchy have a lever?** Measurement only; no hierarchy is implemented.

    The question: within one round, is time dominated by the **master LP** or by the **separation scan**?
      - scan-dominated => fewer rounds is not the only lever; a faster scan beats a hierarchy;
      - LP-dominated   => rounds are the only lever, and the hierarchy is worth trying.
    This is the **pivot** question, not the decision: the decision is Stage 1's pre-registered 0.70x
    wall-time criterion.

    Instances: SynthRAD, the real delta* ~ 0 regime -- which is the regime the decision is made on.
    For each m: rounds, |F|, per-round master/scan time, LP share, RSS.
"""

from __future__ import annotations

import dataclasses
import json
import resource
import sys
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sparse
from jaxtyping import Float

_REPO_ROOT = Path(__file__).resolve().parents[4]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from iot import InverseOTProblem  # noqa: E402
from iot.core.features import DiagonalWeightFeatures  # noqa: E402
from iot.relaxed.loop_regret import solve_inverse_ot_regret  # noqa: E402
from iot.relaxed.master_regret import RegretMasterSolver  # noqa: E402
from iot_experiments.harness.config import RunConfig, build_parser  # noqa: E402
from iot_experiments.harness.datasets import reduce_features, stream_features  # noqa: E402
from iot_experiments.harness.records import RunLog  # noqa: E402
from iot_experiments.harness.solver import box_simplex  # noqa: E402
from iot_experiments.harness.synthrad import discover_patients  # noqa: E402

PROVENANCE: dict[str, object] = {
    "stage": "0 — pivot only: master LP vs separation scan share of round time",
    "instance": "SynthRAD brain, delta* ~ 0 regime (the regime the Stage 1 decision uses)",
    "engine": "solve_inverse_ot_regret, prune=False, cupdlpx, gamma=1",
    "not_a_decision": "Stage 1's pre-registered 0.70x wall-time criterion decides, not this",
}


def _rss_gb() -> float:
    """CN: 峰值 RSS（GB）。EN: peak RSS in GB."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6


def _mem_available_gb() -> float:
    """CN: 系统可用内存（GB）。EN: system MemAvailable in GB."""
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return float(line.split()[1]) / 1024.0 / 1024.0
    return float("nan")


@dataclasses.dataclass(frozen=True)
class Stage0Config(RunConfig):
    """CN: 旋钮。EN: knobs."""

    section = "synthrad_stage0"
    study: str = "synthrad_stage0"
    ms: tuple[int, ...] = (2000, 5000)
    k: int = 64
    draw: int = 0
    gammas: tuple[int, ...] = (1,)
    warm: bool = False

    @classmethod
    def add_study_arguments(cls, parser) -> None:
        """CN: 只加本脚本特有的参数。EN: only this script's flags."""
        parser.add_argument("--ms", type=int, nargs="+", default=[2000, 5000])
        parser.add_argument("--k", type=int, default=64)
        parser.add_argument("--draw", type=int, default=0)
        parser.add_argument("--warm", action="store_true", help="warm-start the master between rounds")
        parser.add_argument("--gammas", type=int, nargs="+", default=[1],
                            help="separation top-gamma values; more than one gives a baseline sweep")


def main(argv: list[str] | None = None) -> int:
    """CN: 入口。EN: entry point."""
    parser = build_parser(Stage0Config)
    args = parser.parse_args(list(argv) if argv is not None else None)
    config = Stage0Config.from_namespace(args, parser)
    run_dir = config.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    log = RunLog(run_dir / "run.log")
    log.line(f"stage 0 pivot: master LP vs scan. ms={config.ms} k={config.k}")
    log.line(f"MemAvailable at start: {_mem_available_gb():.2f} GB")

    split = json.loads((config.out / "synthrad_v2" / "split_v2.json").read_text())
    patients = discover_patients(_REPO_ROOT / "data" / "Task1", anatomy="brain")
    fit_pats = [p for p in patients if p.patient in set(split["fit"])]
    rng = np.random.default_rng(1000 + int(config.draw))
    chosen = [fit_pats[i] for i in rng.permutation(len(fit_pats))[:60]]
    log.line(f"streaming features for {len(chosen)} FIT patients (k={config.k})")
    pm, pc, _pid, _z, _ctr = stream_features(chosen, window=(-200.0, 300.0), max_per_patient=None,
                                             min_gap=0, seed=int(config.draw))
    ftr, ftc = reduce_features(pm, pc, k=int(config.k), geometry_name="G3", whiten=False)
    del pm, pc
    log.line(f"features ready {ftr.shape}, peak RSS {_rss_gb():.2f} GB")

    rows: list[dict[str, object]] = []
    for m_target in [int(v) for v in config.ms]:
      for gamma in [int(g) for g in config.gammas]:
          m = int(min(int(m_target), int(ftr.shape[0])))
          sub = np.sort(np.random.default_rng(777).permutation(int(ftr.shape[0]))[:m])
          x: Float[np.ndarray, "m d"] = np.ascontiguousarray(ftr[sub])
          y: Float[np.ndarray, "m d"] = np.ascontiguousarray(ftc[sub])
          k = int(x.shape[1])
          plan = sparse.csr_matrix((np.full(m, 1.0 / m), (np.arange(m), np.arange(m))), shape=(m, m))
          problem = InverseOTProblem(features=DiagonalWeightFeatures(x, y),
                                     theta_set=box_simplex(k, box_relative=0.25), plan=plan)
          log.line(f"--- m={m} k={k}, MemAvailable {_mem_available_gb():.2f} GB, "
                   f"peak RSS {_rss_gb():.2f} GB ---")
          t0 = time.perf_counter()
          status = "ok"
          try:
              res = solve_inverse_ot_regret(
                  problem, tol=None, gamma=int(gamma), max_iterations=200, warm_start=bool(config.warm),
                  master=RegretMasterSolver(backend=str(config.solver), tie_break=False,
                                            params={"time_sec_limit": float(config.solve_limit_s)}),
                  eps_r_rel=1e-6, prune=False, check_pot=False, selection="none")
              elapsed = time.perf_counter() - t0
              per_round = [{"round": int(it.index), "master_s": round(float(it.master_time), 3),
                            "scan_s": round(float(it.scan_time), 3),
                            "wall_s": round(float(it.wall_time), 3),
                            "F": int(it.finite_edges)} for it in res.iterations]
              master_total = float(sum(r["master_s"] for r in per_round))
              scan_total = float(sum(r["scan_s"] for r in per_round))
              share = scan_total / max(master_total + scan_total, 1e-30)
              row = {"m": m, "k": k, "gamma": int(gamma), "warm": bool(config.warm), "rounds": int(res.n_iterations),
                     "F_final": int(res.finite_edges), "F_frac": int(res.finite_edges) / (m * m),
                     "secs": round(elapsed, 1), "peak_gb": round(_rss_gb(), 2),
                     "master_total_s": round(master_total, 2), "scan_total_s": round(scan_total, 2),
                     "scan_share": round(share, 4), "per_round": per_round,
                     "lower": float(res.lower), "upper": float(res.upper),
                     "stall_status": str(res.stall_status)}
              rows.append(row)
              log.line(f"m={m:>5} gamma={gamma} warm={config.warm} rounds={row['rounds']:>3} |F|={row['F_final']:>7} "
                       f"({100 * row['F_frac']:.2f}%) master={row['master_total_s']:.1f}s "
                       f"scan={row['scan_total_s']:.1f}s scan_share={share:.3f} "
                       f"total={row['secs']:.0f}s peak={row['peak_gb']:.1f}GB "
                       f"stall={row['stall_status']}")
              log.line("  per-round (master_s, scan_s): " + ", ".join(
                  f"{r['round']}:({r['master_s']},{r['scan_s']})" for r in per_round[:12]))
          except Exception as error:  # noqa: BLE001 - reported, never substituted
              status = f"FAILED: {type(error).__name__}: {str(error)[:160]}"
              rows.append({"m": m, "k": k, "status": status,
                           "secs": round(time.perf_counter() - t0, 1),
                           "peak_gb": round(_rss_gb(), 2)})
              log.line(f"m={m:>5} {status}")

    (run_dir / "stage0.json").write_text(
        json.dumps({"rows": rows, "provenance": PROVENANCE,
                    "config": {k2: (list(v) if isinstance(v, tuple) else str(v) if isinstance(v, Path) else v)
                               for k2, v in dataclasses.asdict(config).items()}}, indent=2) + "\n",
        encoding="utf-8")
    log.line(f"wrote {run_dir / 'stage0.json'}")
    log.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
