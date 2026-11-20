"""
EN: **the arms table against the neutral baseline.**

    Reads `out/multi_observation_sparse/arms/<tag>/records.json` and prints, per (K, L, p): each arm's held-out
    agreement and `cost_rel` as mean±sd across seeds, and the gain over `E_neutral` as the mean±sd of the
    **per-seed** difference. `A_single` has one row per plan, so its per-seed value is the mean over that seed's
    plans; this is what makes the gain comparable across arms.

    θ is not a headline reading and is not printed here; it stays in the records.

CN: **对照中性基线的各臂表。** 读取 `out/multi_observation_sparse/arms/<tag>/records.json`，按 (K, L, p) 打印每臂留出
    复现率与 `cost_rel` 的跨种子 mean±sd，以及相对 `E_neutral` 的增益（**逐种子**差值的 mean±sd）。`A_single` 每个计划
    一行，故其逐种子值为该种子各计划的均值——这样各臂的增益才可比。θ 不作头条读数，此处不打印，仍在记录中。

    python -m iot_experiments.studies.synthetic.multi_observation_sparse.analyse_arms --tag g3
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

OUT = Path("iot_experiments/out/multi_observation_sparse/arms")
ARMS = ("E_neutral", "A_single", "B_averaged", "B_informative", "C_joint", "A_single_big", "D_oracle")
DIAGNOSTICS = ("train_regret", "outer_gap", "iterations", "hit_iteration_cap", "weight_on_zero_mass_bins",
               "windows_at_zero_gradient", "theta_star_draws")


def _mean_sd(values: np.ndarray) -> str:
    """EN: `mean±sd`, with `n/a` when a single seed makes the spread undefined. CN: `mean±sd`，单种子时离散度为 `n/a`。"""
    return f"{values.mean():.3f}±n/a" if values.size < 2 else f"{values.mean():.3f}±{values.std(ddof=1):.3f}"


def main(argv: list[str] | None = None) -> int:
    """EN: print the table. CN: 打印表格。"""
    parser = argparse.ArgumentParser(description="arms table against the neutral baseline")
    parser.add_argument("--tag", default="g3")
    args = parser.parse_args(argv)
    records = json.loads((OUT / args.tag / "records.json").read_text())["records"]

    # EN: per (cell, arm, seed) -> readings of that seed (several rows only for A_single).
    # CN: (格子, 臂, 种子) -> 该种子的读数（只有 A_single 会有多行）。
    agreement: dict[tuple, dict[str, dict[int, list[float]]]] = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    cost: dict[tuple, dict[str, dict[int, list[float]]]] = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    diagnostics: dict[tuple, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for record in records:
        key = (record["config"]["K"], record["config"]["L"], record["config"]["noise"])
        seed = int(record["draw"])
        arm = record["arm"]
        agreement[key][arm][seed].append(record["metrics"]["agreement"])
        cost[key][arm][seed].append(record["metrics"]["cost_rel"])
        for name in DIAGNOSTICS:
            value = record["metrics"].get(name)
            if value is None:
                value = record["extra"].get("gap" if name == "outer_gap" else name)
            if value is not None and np.isfinite(value):
                diagnostics[key][f"{arm}:{name}"].append(float(value))

    def per_seed(table: dict[int, list[float]], seeds: list[int]) -> np.ndarray:
        """EN: one value per seed (mean over that seed's rows). CN: 每个种子一个值（该种子各行的均值）。"""
        return np.array([np.mean(table[seed]) for seed in seeds if seed in table])

    print(f"{'K':>3s} {'L':>3s} {'p':>4s} {'arm':13s} {'draws':>5s} {'agreement':>15s} "
          f"{'gain vs neutral':>17s} {'cost_rel':>15s}")
    for key in sorted(agreement):
        seeds = sorted({seed for arm in agreement[key] for seed in agreement[key][arm]})
        neutral = (per_seed(agreement[key]["E_neutral"], seeds)
                   if "E_neutral" in agreement[key] else None)
        for arm in ARMS:
            if arm not in agreement[key]:
                continue
            values = per_seed(agreement[key][arm], seeds)
            costs = per_seed(cost[key][arm], seeds)
            gain = "n/a" if neutral is None or neutral.size != values.size else (
                "—" if arm == "E_neutral" else _mean_sd(values - neutral))
            print(f"{key[0]:3d} {key[1]:3d} {key[2]:4g} {arm:13s} {values.size:5d} "
                  f"{_mean_sd(values):>15s} {gain:>17s} {_mean_sd(costs):>15s}")
        cap = diagnostics[key].get("C_joint:hit_iteration_cap")
        iters = diagnostics[key].get("C_joint:iterations")
        objective = diagnostics[key].get("C_joint:train_regret")
        skipped = diagnostics[key].get("B_averaged:windows_at_zero_gradient")
        if cap is not None and iters is not None and objective is not None:
            skipped_text = f"{np.mean(skipped):.1f}" if skipped else "n/a"
            print(f"{'':12s} C diag: objective {np.mean(objective):.2e}  gap {np.mean(diagnostics[key]['C_joint:outer_gap']):.1e}"
                  f"  iterations {np.mean(iters):.1f}  capped {np.mean(cap):.2f}  skipped_windows {skipped_text}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
