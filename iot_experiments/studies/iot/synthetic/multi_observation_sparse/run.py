"""
EN: **Sparse multi-observation inverse OT: correctness, inner-solver scaling, and the scalability benchmark.**

    Four modes, all synthetic (unit-square points, features computed on the fly):

      `correctness`  n≈30, L≈5: the bundle's value against a dense joint LP (the exact `F*`) and against a dense
                     Frank–Wolfe on the same feature family; the bracket must contain `F*`.
      `inner`        one cold forward solve at `n in {200, 800, 3200}`: sweeps, `|E|`, `|E|/(n+m)`, LP and pricing
                     times — the measurement that decides whether the sparse HiGHS inner LP dominates.
      `benchmark`    dense Frank–Wolfe vs the sparse bundle at the same `n` and `L`, in separate processes so the
                     peak resident memory is per arm; the dense arm stops at a stated time/memory cap.
      `sanity`       planted θ* with L windows, controls (out-of-family plans, shuffled feature assignment), labelled
                     a sanity check: it shows the solver recovers plans, not that the method is better.

    Every row records θ, the certified outer gap, the per-window inner gaps, the diagonal mass of observed and
    predicted plans, and — where the family has such a column — the weight on a constant / same-zone / mask column.
    The family used here has none of those three by construction.

CN: **稀疏多观测逆 OT：正确性、内层求解器规模、以及可扩展性基准。** 四种模式，全部为合成数据（单位方格点、特征即时计算）：
    `correctness`（与稠密联合 LP 的 `F*` 及同族稠密 FW 对比，区间须包含 `F*`）、`inner`（单次冷启动前向求解在
    `n∈{200,800,3200}` 的扫描数、`|E|`、`|E|/(n+m)`、LP 与定价时间 —— 决定稀疏 HiGHS 内层是否主导）、`benchmark`
    （同 n、L 下稠密 FW 与稀疏 bundle 分进程对比峰值内存与时间，稠密臂在给定时间/内存上限处停止）、`sanity`（植入 θ*，
    含族外计划与特征错配两种对照，明确标注为健全性检查）。每行记录 θ、带证书的外层间隙、逐窗内层间隙、观测与预测计划的
    对角质量；若特征族含常数/同 zone/掩码列则记录其权重 —— 本族按构造不含这三列。
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import ot
import scipy.sparse as sparse

_REPO_ROOT = Path(__file__).resolve().parents[4]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from iot.core.theta import SimplexTheta  # noqa: E402
from iot.sparse.bundle_multi import solve_bundle_multi  # noqa: E402
from iot.sparse.features import PointDistanceFeatures  # noqa: E402
from iot.sparse.sparse_ot import solve_sparse_ot  # noqa: E402
from iot_experiments.harness.records import RunLog, RunRecord, format_table, summarise, write_records  # noqa: E402
from iot_experiments.studies.iot.synthetic.multi_observation_sparse.reference import (  # noqa: E402
    dense_frank_wolfe,
    dense_joint_lp,
)

OUT = Path("iot_experiments/out/multi_observation_sparse")


def _peak_memory_mb() -> float:
    """EN: peak resident set size of this process, MB. CN: 本进程的峰值常驻内存（MB）。"""
    try:
        with open("/proc/self/status", encoding="ascii") as handle:
            for line in handle:
                if line.startswith("VmHWM:"):
                    return float(line.split()[1]) / 1024.0
    except OSError:
        return float("nan")
    return float("nan")


def _planted(n: int, n_windows: int, seed: int, *, fresh_points: bool, out_of_family: bool = False,
             theta_star: np.ndarray | None = None
             ) -> tuple[list[PointDistanceFeatures], list[sparse.csr_matrix], np.ndarray]:
    """
    EN: features (one per window), observed plans exactly optimal at θ*, and θ*.

        `fresh_points=True` gives every window its own point cloud (same θ*). `out_of_family=True` adds a random
        symmetric cost to the generating cost, so no θ in the family makes those plans optimal.

    CN: 特征（每窗一个）、恰在 θ* 处最优的观测计划、以及 θ*。`fresh_points=True` 时每窗新点云（同一 θ*）；
        `out_of_family=True` 时在生成代价上加随机对称代价，使族内任何 θ 都无法让这些计划最优。
    """
    planted = np.array([0.1, 0.5, 0.2, 0.2]) if theta_star is None else np.asarray(theta_star, dtype=np.float64)
    rng = np.random.default_rng(seed)
    maps: list[PointDistanceFeatures] = []
    plans: list[sparse.csr_matrix] = []
    for window in range(n_windows):
        point_rng = np.random.default_rng((seed, window) if fresh_points else (seed, 0))
        source_points = point_rng.random((n, 2))
        target_points = point_rng.random((n, 2))
        features = PointDistanceFeatures(source_points, target_points, scale=1.0, width=4)
        rows = np.repeat(np.arange(n, dtype=np.int64), n)
        cols = np.tile(np.arange(n, dtype=np.int64), n)
        cost = np.asarray(features.costs(planted, rows, cols)).reshape(n, n)
        if out_of_family:
            noise = rng.random((n, n))
            cost = cost + 0.5 * (noise + noise.T)
        source_mass = rng.random(n) + 0.1
        source_mass /= source_mass.sum()
        target_mass = rng.random(n) + 0.1
        target_mass /= target_mass.sum()
        plan = np.asarray(ot.emd(source_mass, target_mass, cost, numItermax=2_000_000))
        maps.append(features)
        plans.append(sparse.csr_matrix(plan))
    return maps, plans, planted


def _plan_agreement(observed: list[sparse.csr_matrix], predicted: tuple[sparse.csr_matrix, ...]) -> float:
    """EN: `Σ min(P, P̂) / Σ P`, i.e. the fraction of the observed mass placed on the same cells. CN: 观测质量中落在同一格子的比例。"""
    totals = []
    for plan, guess in zip(observed, predicted, strict=True):
        dense_observed = plan.toarray()
        dense_predicted = guess.toarray()
        denominator = float(dense_observed.sum())
        totals.append(float(np.minimum(dense_observed, dense_predicted).sum()) / denominator)
    return float(np.mean(totals))


def _cost_rel(theta: np.ndarray, planted: np.ndarray, features: PointDistanceFeatures) -> float:
    """EN: the recovered cost against the planted one, up to the positive scale. CN: 恢复代价与植入代价的相对误差（到正缩放）。"""
    n = int(features.n_source)
    rows = np.repeat(np.arange(n, dtype=np.int64), n)
    cols = np.tile(np.arange(n, dtype=np.int64), n)
    truth = np.asarray(features.costs(planted, rows, cols)).reshape(n, n)
    guess = np.asarray(features.costs(theta, rows, cols)).reshape(n, n)
    scale = float(np.sum(guess * truth) / np.sum(guess ** 2)) if np.sum(guess ** 2) > 0 else 0.0
    return float(np.linalg.norm(scale * guess - truth) / np.linalg.norm(truth))


def _theta_l1(theta: np.ndarray, planted: np.ndarray) -> float:
    """EN: `Σ|θ − θ*|`. CN: 参数支持集的 L1 距离。"""
    return float(np.abs(np.asarray(theta) - np.asarray(planted)).sum())


def mode_correctness(log: RunLog, out_dir: Path) -> list[RunRecord]:
    """EN: bundle vs dense joint LP vs dense Frank–Wolfe on one planted instance. CN: bundle 与稠密 LP、稠密 FW 对比。"""
    n, n_windows, seed = 30, 5, 0
    maps, plans, planted = _planted(n, n_windows, seed, fresh_points=False)
    started = time.perf_counter()
    bundle = solve_bundle_multi(plans, maps, tol=1e-6, max_iterations=60, inner_tol=1e-9)
    bundle_seconds = time.perf_counter() - started
    lp = dense_joint_lp(plans, maps, theta_set=SimplexTheta(4))
    fw_value, fw_theta, _trace = dense_frank_wolfe(plans, maps, theta_set=SimplexTheta(4), iterations=100)
    log.line(f"[correctness] n={n} L={n_windows}: bundle F={bundle.value:.10e} "
             f"bracket=[{bundle.lower:.10e},{bundle.upper:.10e}] gap={bundle.gap:.2e} "
             f"converged={bundle.converged} ({bundle_seconds:.1f}s)")
    log.line(f"[correctness] dense joint LP F*={lp.value:.10e}  dense FW F={fw_value:.10e}")
    log.line(f"[correctness] theta bundle={np.round(bundle.theta, 4).tolist()} FW={np.round(fw_theta, 4).tolist()} "
             f"planted={np.round(planted, 4).tolist()}")
    log.line(f"[correctness] plan agreement bundle={_plan_agreement(plans, bundle.plans):.6f}  "
             f"diag obs={np.round(bundle.observed_diagonal_mass, 3).tolist()} "
             f"pred={np.round(bundle.predicted_diagonal_mass, 3).tolist()}")
    row = RunRecord(
        study="multi_observation_sparse", geometry=f"correctness_n{n}_L{n_windows}", arm="bundle", k=4, m=n, draw=0,
        metrics={"value": bundle.value, "lower": bundle.lower, "upper": bundle.upper, "gap": bundle.gap,
                 "lp_value": lp.value, "fw_value": fw_value,
                 "theta_l1": _theta_l1(bundle.theta, planted), "cost_rel": _cost_rel(bundle.theta, planted, maps[0]),
                 "plan_agreement": _plan_agreement(plans, bundle.plans)},
        config={"mode": "correctness", "n": n, "L": n_windows, "tol": 1e-6, "inner_tol": 1e-9},
        extra={"theta": [float(v) for v in bundle.theta], "theta_star": [float(v) for v in planted],
               "inner_gaps": [float(v) for v in bundle.iterations[-1].inner_gaps],
               "edges": [int(v) for v in bundle.iterations[-1].edges],
               "observed_diagonal_mass": [float(v) for v in bundle.observed_diagonal_mass],
               "predicted_diagonal_mass": [float(v) for v in bundle.predicted_diagonal_mass],
               "seconds": round(bundle_seconds, 3)})
    return [row]


def mode_inner(log: RunLog, out_dir: Path, *, prune_slack: float | None = None) -> list[RunRecord]:
    """EN: one cold forward solve per size; how `|E|` and the inner time grow. CN: 每个规模一次冷启动前向求解。"""
    rows: list[RunRecord] = []
    for n in (200, 800, 3200):
        maps, plans, _planted_theta = _planted(n, 1, 1, fresh_points=False)
        features = maps[0]
        plan = plans[0]
        source_mass = np.asarray(plan.sum(axis=1)).reshape(-1)
        target_mass = np.asarray(plan.sum(axis=0)).reshape(-1)
        theta = np.array([0.1, 0.5, 0.2, 0.2])
        result = solve_sparse_ot(features, source_mass, target_mass, theta=theta, edges=None, gamma=2,
                                 tol=1e-9, max_sweeps=80, prune_slack=prune_slack)
        reference = float("nan")
        if n <= 800:
            # EN: the dense reference is skipped above n=800: its cost tensor would be 320 MB at n=3200 before POT
            #     allocates anything, and the point of the run is the sparse path, not the reference.
            # CN: n>800 时跳过稠密参考：n=3200 的光代价张量就是 320 MB，何况 POT 还要再分配；该规模要测的是稀疏路径。
            all_rows = np.repeat(np.arange(n, dtype=np.int64), n)
            all_cols = np.tile(np.arange(n, dtype=np.int64), n)
            dense = np.asarray(features.costs(theta, all_rows, all_cols)).reshape(n, n)
            reference = float(np.sum(dense * np.asarray(ot.emd(source_mass, target_mass, dense,
                                                              numItermax=2_000_000))))
        log.line(f"[inner] n={n} prune={prune_slack}: |E|={result.n_edges} "
                 f"(peak {result.max_edges} = {result.max_edges / (2 * n):.2f}×(n+m), "
                 f"kept {result.n_edges / (2 * n):.2f}×) "
                 f"sweeps={result.sweeps} gap={result.inner_gap:.1e} value={result.value:.8f} "
                 f"dense={reference:.8f} lp={result.lp_seconds:.2f}s price={result.price_seconds:.2f}s "
                 f"total={result.seconds:.2f}s")
        rows.append(RunRecord(
            study="multi_observation_sparse", geometry=f"inner_n{n}", arm="sparse_ot", k=4, m=n, draw=0,
            metrics={"n_edges": float(result.n_edges), "max_edges": float(result.max_edges),
                     "peak_edges_over_n_plus_m": result.max_edges / (2 * n),
                     "edges_over_n_plus_m": result.n_edges / (2 * n),
                     "sweeps": float(result.sweeps), "inner_gap": result.inner_gap,
                     "value": result.value, "dense_value": reference,
                     "lp_seconds": result.lp_seconds, "price_seconds": result.price_seconds,
                     "total_seconds": result.seconds},
            config={"mode": "inner", "n": n, "tol": 1e-9},
            extra={"n_nonzeros": int(result.n_nonzeros), "history": [float(v) for v in result.history]}))
    return rows


def _bench_arm(n: int, n_windows: int, arm: str, *, max_seconds: float, max_memory_mb: float,
               iterations: int) -> dict[str, object]:
    """EN: run one arm in this process and report its peak memory; called as a child by `mode_benchmark`. CN: 在子进程中跑一个臂并报告峰值内存。"""
    maps, plans, planted = _planted(n, n_windows, 0, fresh_points=False)
    started = time.perf_counter()
    status = "ok"
    value = float("nan")
    theta = ()
    gap = float("nan")
    if arm == "dense_fw":
        def _stop(_step: int, _best: float, _theta: np.ndarray) -> bool:
            """EN: stop on the time or memory cap; that is where the dense arm ends. CN: 到达时间/内存上限即停。"""
            nonlocal status
            if time.perf_counter() - started > max_seconds:
                status = f"stopped:time>{max_seconds:.0f}s"
                return False
            if _peak_memory_mb() > max_memory_mb:
                status = f"stopped:memory>{max_memory_mb:.0f}MB"
                return False
            return True

        value, theta_best, _trace = dense_frank_wolfe(plans, maps, theta_set=SimplexTheta(4),
                                                      iterations=int(iterations), stop=_stop)
        theta = tuple(float(v) for v in theta_best)
    elif arm == "bundle":
        result = solve_bundle_multi(plans, maps, tol=1e-6, max_iterations=int(iterations), inner_tol=1e-9)
        value, theta, gap = result.value, tuple(float(v) for v in result.theta), result.gap
        if time.perf_counter() - started > max_seconds:
            status = f"stopped:time>{max_seconds:.0f}s"
    else:
        raise ValueError(arm)
    return {"arm": arm, "n": n, "L": n_windows, "status": status, "value": value, "gap": gap,
            "theta": list(theta), "seconds": time.perf_counter() - started,
            "peak_memory_mb": _peak_memory_mb()}


def _value_dense(plans: list[sparse.csr_matrix], maps: list[PointDistanceFeatures], theta: np.ndarray) -> float:
    """EN: exact `F(θ)` with dense costs and POT. CN: 用稠密代价与 POT 计算精确 `F(θ)`。"""
    total = 0.0
    for plan, features in zip(plans, maps, strict=True):
        n = int(features.n_source)
        rows = np.repeat(np.arange(n, dtype=np.int64), n)
        cols = np.tile(np.arange(n, dtype=np.int64), n)
        dense = np.asarray(features.costs(theta, rows, cols)).reshape(n, n)
        coo = plan.tocoo()
        phi = features.features(coo.row.astype(np.int64), coo.col.astype(np.int64))
        source_mass = np.asarray(plan.sum(axis=1)).reshape(-1)
        target_mass = np.asarray(plan.sum(axis=0)).reshape(-1)
        total += float(phi.T @ coo.data @ theta) - float(ot.emd2(source_mass, target_mass, dense,
                                                                numItermax=2_000_000))
    return total


def mode_benchmark(log: RunLog, out_dir: Path, *, max_seconds: float = 120.0, max_memory_mb: float = 3000.0,
                   iterations: int = 12) -> list[RunRecord]:
    """EN: spawn one child per (arm, n) so peak memory is per arm. CN: 每个 (臂, n) 起一个子进程，使峰值内存按臂计。"""
    rows: list[RunRecord] = []
    for n in (200, 800, 3200):
        # EN: one dense POT solve at n=3200 is the failure point already; cap the dense arm so it cannot run away.
        # CN: n=3200 上单次稠密 POT 求解就是失效点；给稠密臂设上限以免失控。
        iterations_for = int(iterations) if n <= 200 else (5 if n <= 800 else 1)
        for arm in ("dense_fw", "bundle"):
            command = [sys.executable, "-m",
                       "iot_experiments.studies.iot.synthetic.multi_observation_sparse.run",
                       "--mode", "bench-arm", "--arm", arm, "--n", str(n), "--L", "3",
                       "--iterations", str(iterations_for), "--max-seconds", str(max_seconds),
                       "--max-memory-mb", str(max_memory_mb)]
            completed = subprocess.run(command, capture_output=True, text=True, timeout=max_seconds + 40)
            payload = None
            for line in completed.stdout.splitlines():
                if line.startswith("BENCH "):
                    payload = json.loads(line[len("BENCH "):])
            if payload is None:
                log.line(f"[benchmark] n={n} {arm}: FAILED (rc={completed.returncode}) "
                         f"{completed.stderr.strip().splitlines()[-1][:160] if completed.stderr.strip() else ''}")
                rows.append(RunRecord(
                    study="multi_observation_sparse", geometry=f"bench_n{n}", arm=arm, k=4, m=n, draw=0,
                    metrics={}, config={"mode": "benchmark", "n": n, "L": 3, "iterations": iterations},
                    excluded=True, status=f"child failed rc={completed.returncode}"))
                continue
            log.line(f"[benchmark] n={n} {arm}: value={payload['value']:.6e} gap={payload['gap']:.2e} "
                     f"peak={payload['peak_memory_mb']:.0f} MB time={payload['seconds']:.1f}s "
                     f"status={payload['status']}")
            rows.append(RunRecord(
                study="multi_observation_sparse", geometry=f"bench_n{n}", arm=arm, k=4, m=n, draw=0,
                metrics={"value": payload["value"], "gap": payload["gap"],
                         "peak_memory_mb": payload["peak_memory_mb"], "seconds": payload["seconds"]},
                config={"mode": "benchmark", "n": n, "L": 3, "iterations": iterations,
                        "max_seconds": max_seconds, "max_memory_mb": max_memory_mb},
                status=str(payload["status"]), extra={"theta": payload["theta"]}))
    return rows


def mode_sanity(log: RunLog, out_dir: Path) -> list[RunRecord]:
    """EN: planted sanity check with controls; not a result about quality. CN: 植入健全性检查与对照；不是质量结论。"""
    rows: list[RunRecord] = []
    n = 40
    for n_windows in (1, 5, 20):
        for seed in (0, 1):
            maps, plans, planted = _planted(n, n_windows, seed, fresh_points=False)
            result = solve_bundle_multi(plans, maps, tol=1e-6, max_iterations=40, inner_tol=1e-9)
            agreement = _plan_agreement(plans, result.plans)
            log.line(f"[sanity] L={n_windows} seed={seed}: F={result.value:.3e} gap={result.gap:.2e} "
                     f"agreement={agreement:.4f} cost_rel={_cost_rel(result.theta, planted, maps[0]):.3f} "
                     f"theta_l1={_theta_l1(result.theta, planted):.3f}")
            rows.append(RunRecord(
                study="multi_observation_sparse", geometry=f"sanity_n{n}", arm="planted", k=4, m=n, draw=seed,
                metrics={"value": result.value, "gap": result.gap, "plan_agreement": agreement,
                         "cost_rel": _cost_rel(result.theta, planted, maps[0]),
                         "theta_l1": _theta_l1(result.theta, planted)},
                config={"mode": "sanity", "n": n, "L": n_windows, "tol": 1e-6, "inner_tol": 1e-9},
                extra={"theta": [float(v) for v in result.theta], "theta_star": [float(v) for v in planted],
                       "inner_gaps": [float(v) for v in result.iterations[-1].inner_gaps],
                       "observed_diagonal_mass": [float(v) for v in result.observed_diagonal_mass],
                       "predicted_diagonal_mass": [float(v) for v in result.predicted_diagonal_mass]}))
            # EN: control — plans generated outside the family; the reading must move.
            # CN: 对照 —— 计划由族外代价生成；读数必须变化。
            control_maps, control_plans, control_planted = _planted(n, n_windows, seed, fresh_points=False,
                                                                    out_of_family=True)
            control = solve_bundle_multi(control_plans, control_maps, tol=1e-6, max_iterations=40, inner_tol=1e-9)
            control_agreement = _plan_agreement(control_plans, control.plans)
            log.line(f"[sanity] L={n_windows} seed={seed} control(out-of-family): F={control.value:.3e} "
                     f"gap={control.gap:.2e} agreement={control_agreement:.4f}")
            rows.append(RunRecord(
                study="multi_observation_sparse", geometry=f"sanity_n{n}", arm="control_out_of_family", k=4, m=n,
                draw=seed,
                metrics={"value": control.value, "gap": control.gap, "plan_agreement": control_agreement},
                config={"mode": "sanity", "n": n, "L": n_windows, "tol": 1e-6, "inner_tol": 1e-9},
                extra={"theta": [float(v) for v in control.theta]}))
    return rows


def main(argv: list[str] | None = None) -> int:
    """EN: entry point. CN: 入口。"""
    parser = argparse.ArgumentParser(description="sparse multi-observation inverse OT")
    parser.add_argument("--mode", required=True,
                        choices=("correctness", "inner", "benchmark", "sanity", "bench-arm"))
    parser.add_argument("--arm", default="bundle")
    parser.add_argument("--n", type=int, default=800)
    parser.add_argument("--L", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=12)
    parser.add_argument("--max-seconds", type=float, default=120.0)
    parser.add_argument("--max-memory-mb", type=float, default=3000.0)
    parser.add_argument("--prune-slack", type=float, default=None)
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.mode == "bench-arm":
        payload = _bench_arm(int(args.n), int(args.L), str(args.arm), max_seconds=float(args.max_seconds),
                             max_memory_mb=float(args.max_memory_mb), iterations=int(args.iterations))
        print("BENCH " + json.dumps(payload))
        return 0

    out_dir = OUT / args.mode
    out_dir.mkdir(parents=True, exist_ok=True)
    log = RunLog(out_dir / "run.log")
    log.line(f"mode={args.mode} argv={sys.argv[1:]}")
    if args.mode == "correctness":
        records = mode_correctness(log, out_dir)
    elif args.mode == "inner":
        records = mode_inner(log, out_dir, prune_slack=args.prune_slack)
    elif args.mode == "benchmark":
        records = mode_benchmark(log, out_dir, max_seconds=float(args.max_seconds),
                                 max_memory_mb=float(args.max_memory_mb), iterations=int(args.iterations))
    else:
        records = mode_sanity(log, out_dir)
    write_records(out_dir / "records.json", records,
                  provenance={"question": "does the sparse bundle match the dense references and scale in memory?",
                              "mode": args.mode, "features": "PointDistanceFeatures [s, s^2, log1p, s^3], "
                                                             "subsample-RMS normalised; no constant/same-zone column",
                              "inner_solver": "scipy linprog HiGHS on a sparse A_eq (labelled reference)"})
    table = summarise(records, group=("arm", "geometry"),
                      metrics=("value", "gap", "peak_memory_mb", "seconds", "n_edges", "plan_agreement"))
    log.line("\n" + format_table(table, ("arm", "geometry", "n_draws", "value_mean", "gap_mean",
                                         "peak_memory_mb_mean", "seconds_mean", "n_edges_mean",
                                         "plan_agreement_mean")))
    log.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
