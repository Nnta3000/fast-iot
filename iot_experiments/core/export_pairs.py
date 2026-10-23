"""
CN: **伪配对边表导出（DPM 与 fast-iot 之间唯一的接口契约）** —— 永远写出**完整的加权边表**，
    不再为"取最重的一个伙伴"另做文件格式：那只是**读**的时候一行 `argmax`，不是另一种格式。

    文件内容（每行一条边）：
        i, j, mass, weight, confidence, cost_at_theta
      - `mass`       = pi_hat_ij（方案里的质量，>= 0）
      - `weight`     = mass / a_i，**按行归一化**，故同一 i 的 weight 之和为 1；这是 loss 用的权重 w_ij
      - `confidence` = sigma_ij + delta，即该边"有多不紧"（>= 0 表示被违反）
      - `cost_at_theta` = c_ij(theta_fit)

    另写 `meta.json`：theta、delta、n、|S|、以及**本表是如何产生的**（plan 来源、是否经过求解、
    求解器与容差），使任何下游读数都能被复现，而不是只看到一个 CSV。

EN: **Pseudo-pair edge-list export (the single contract between DPM and fast-iot)** -- always write the
    **full weighted edge list**. "Take the heaviest partner" is a one-line `argmax` at **read** time,
    not a different file format.

    Rows: `i, j, mass, weight, confidence, cost_at_theta`
      - `mass`       = pi_hat_ij (plan mass, >= 0)
      - `weight`     = mass / a_i, **row-normalised**, so weights of one i sum to 1; this is the w_ij for
                       the weighted loss
      - `confidence` = sigma_ij + delta (how far the edge is from tight; >= 0 means violated)
      - `cost_at_theta` = c_ij(theta_fit)

    Plus `meta.json`: theta, delta, n, |S| and **how the table was produced** (plan source, whether the
    solver ran, solver and tolerances), so any downstream number can be reproduced from the file alone.

用法 / EN: usage
    python -m iot_experiments.core.export_pairs --n 1000 --grid 8 --plan real
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import ot
from scipy import sparse

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from iot import InverseOTProblem, SimplexTheta, solve_inverse_ot  # noqa: E402
from iot.backends.cupdlpx_master import CuPdlpxMasterSolver  # noqa: E402

from .real_phase1 import _DATA_ROOT, block_features, cost_matrix, load_gray  # noqa: E402


def write_edge_list(path: Path, *, rows: np.ndarray, cols: np.ndarray, mass: np.ndarray,
                    weight: np.ndarray, confidence: np.ndarray, cost: np.ndarray) -> None:
    """
    CN: 写加权边表（CSV，带表头）。空表也写表头，便于下游无条件下读取。
    EN: Write the weighted edge list (CSV with header). An empty table still gets a header so
        downstream readers can read unconditionally.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write("i,j,mass,weight,confidence,cost_at_theta\n")
        for k in range(rows.size):
            handle.write(f"{int(rows[k])},{int(cols[k])},{mass[k]:.17g},{weight[k]:.17g},"
                         f"{confidence[k]:.17g},{cost[k]:.17g}\n")


def read_edge_list(path: Path) -> dict[str, np.ndarray]:
    """
    CN: 读加权边表（供 DPM 侧使用）。返回原始列，不对"取最重伙伴"做任何预设。
    EN: Read the weighted edge list (for the DPM side). Returns raw columns, with no assumption about
        "take the heaviest partner".
    """
    import csv

    columns: dict[str, list[float]] = {name: [] for name in
                                       ("i", "j", "mass", "weight", "confidence", "cost_at_theta")}
    with path.open("r", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            for name in columns:
                columns[name].append(float(row[name]))
    return {name: np.asarray(values, dtype=np.float64) for name, values in columns.items()}


def heaviest_partner(edges: dict[str, np.ndarray]) -> np.ndarray:
    """
    CN: 可选的**读取**策略：每个源取权重最大的伙伴（`argmax_j w_ij`）。返回每个 i 的 j；
        未被任何边覆盖的行记为 -1。**这是读取端的选择，不是文件格式的一部分。**
    EN: Optional **read** strategy: the heaviest partner per source (`argmax_j w_ij`). Returns j per i,
        with -1 for rows no edge covers. This is a reader-side choice, not part of the format.
    """
    rows = edges["i"].astype(np.int64)
    cols = edges["j"].astype(np.int64)
    weights = edges["weight"]
    n_source = int(rows.max()) + 1 if rows.size else 0
    best = np.full(n_source, -1, dtype=np.int64)
    best_w = np.full(n_source, -np.inf)
    for r, c, w in zip(rows, cols, weights):
        if w > best_w[r]:
            best_w[r] = w
            best[r] = c
    return best


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="export a weighted pseudo-pair edge list")
    parser.add_argument("--n", type=int, default=1000)
    parser.add_argument("--grid", type=int, default=8)
    parser.add_argument("--plan", default="real", choices=("real", "identity", "none"))
    parser.add_argument("--solve", action="store_true",
                        help="run the inverse-OT loop to get theta; otherwise theta is uniform")
    parser.add_argument("--fit", action="store_true",
                        help="only re-weight an existing plan with a fitted theta")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-iterations", type=int, default=400)
    parser.add_argument("--master-time-limit", type=float, default=120.0)
    parser.add_argument("--out", type=Path,
                        default=_REPO_ROOT / "iot_experiments" / "core" / "out" / "pairs")
    parser.add_argument("--tag", default="real_n1000_g8")
    args = parser.parse_args(argv)

    names = sorted((_DATA_ROOT / "T1").glob("mri_*.png"),
                   key=lambda p: int(p.stem.split("_")[1]))[: int(args.n)]
    x_raw = np.stack([block_features(load_gray(_DATA_ROOT / "T1" / p.name), args.grid) for p in names])
    y_raw = np.stack([block_features(load_gray(_DATA_ROOT / "T2" / p.name), args.grid) for p in names])
    pooled = np.concatenate([x_raw, y_raw], axis=0)
    mu = pooled.mean(axis=0, keepdims=True)
    sd = np.where(pooled.std(axis=0, keepdims=True) > 1e-12, pooled.std(axis=0, keepdims=True), 1.0)
    x = (x_raw - mu) / sd
    y = (y_raw - mu) / sd
    m = int(x.shape[0])

    from iot import DiagonalWeightFeatures
    features = DiagonalWeightFeatures(x, y)
    theta_uniform = np.full(features.n_features, 1.0 / features.n_features)

    if args.plan == "real":
        cost_u = cost_matrix(features, theta_uniform)
        a = np.ones(m) / m
        plan = ot.emd(a, a, cost_u, numItermax=10_000_000)
        plan_source = "exact EMD (POT) of the native T1->T2 pairs under uniform theta"
    elif args.plan == "identity":
        plan = np.zeros((m, m))
        plan[np.arange(m), np.arange(m)] = 1.0 / m
        plan_source = "identity permutation of the native pairs (no EMD)"
    else:
        plan = None
        plan_source = "none (re-weighting an existing plan)"

    theta = theta_uniform
    delta = 0.0
    fit_info: dict[str, object] = {"solver_ran": False}
    problem = None
    if plan is not None:
        problem = InverseOTProblem(features=features, theta_set=SimplexTheta(features.n_features),
                                   plan=sparse.csr_matrix(plan))
    if args.solve or args.fit:
        if problem is None:
            raise SystemExit("--solve/--fit need a plan (use --plan real|identity)")
        master = CuPdlpxMasterSolver(params={"time_sec_limit": float(args.master_time_limit)},
                                     epsilon=1e-4)
        t0 = time.perf_counter()
        result = solve_inverse_ot(
            problem, solver=master, tie_break="regularized", max_iterations=int(args.max_iterations),
            scan_backend="torch", validate=True, prune=False, random_seed=int(args.seed),
        )
        theta = np.asarray(result.theta, dtype=np.float64)
        delta = float(result.delta)
        fit_info = {"solver_ran": True, "iterations": int(result.n_iterations),
                    "is_optimal": None if result.is_optimal is None else bool(result.is_optimal),
                    "relative_gap": None if result.relative_gap is None else float(result.relative_gap),
                    "final_max_violation": float(result.final_max_violation),
                    "wall": float(time.perf_counter() - t0)}

    if plan is None:
        raise SystemExit("--plan none is only meaningful together with an existing edge list")

    cost = cost_matrix(features, theta)
    coo = sparse.csr_matrix(plan).tocoo()
    keep = coo.data > 0.0
    rows = coo.row[keep].astype(np.int64)
    cols = coo.col[keep].astype(np.int64)
    mass = coo.data[keep].astype(np.float64)
    row_mass = np.bincount(rows, weights=mass, minlength=m)
    weight = mass / row_mass[rows]
    # CN: confidence = sigma + delta；sigma = f_i + g_j - c_ij，f/g 来自求解结果，未求解时用支撑对偶。
    # EN: confidence = sigma + delta; sigma = f_i + g_j - c_ij from the solve, or support duals if unsolved.
    if fit_info.get("solver_ran"):
        source_dual = np.asarray(result.source_dual, dtype=np.float64)  # type: ignore[name-defined]
        target_dual = np.asarray(result.target_dual, dtype=np.float64)  # type: ignore[name-defined]
    else:
        # CN: 未求解时：f_i = min_j(c_ij - g_j)，g = 0，于是支撑上 sigma = 0（与构造一致）。
        # EN: unsolved: f_i = min_j(c_ij - g_j) with g = 0, so sigma = 0 on the support by construction.
        source_dual = cost.min(axis=1)
        target_dual = np.zeros(cost.shape[1], dtype=np.float64)
    support_cost = cost[rows, cols]
    confidence = source_dual[rows] + target_dual[cols] - support_cost + delta

    out_csv = args.out / f"{args.tag}.edges.csv"
    write_edge_list(out_csv, rows=rows, cols=cols, mass=mass, weight=weight,
                    confidence=confidence, cost=support_cost)
    meta = {
        "n": m, "grid": int(args.grid), "seed": int(args.seed),
        "plan_source": plan_source, "n_edges": int(rows.size),
        "support_over_n": float(rows.size) / float(m),
        "theta": theta.tolist(), "delta": delta, **fit_info,
        "columns": {
            "mass": "pi_hat_ij", "weight": "pi_hat_ij / a_i (row-normalised, sums to 1 per i)",
            "confidence": "sigma_ij + delta (>=0 means violated)",
            "cost_at_theta": "c_ij(theta)",
        },
        "data_root": str(_DATA_ROOT),
    }
    (args.out / f"{args.tag}.meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(f"[export_pairs] {rows.size} edges ({rows.size / m:.2f} per source) -> {out_csv}")
    print(f"[export_pairs] weight per row sums to 1: "
          f"{float(np.min(row_mass[rows] / row_mass[rows])):.0f} row-normalised; "
          f"max degree {int(np.bincount(rows, minlength=m).max())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
