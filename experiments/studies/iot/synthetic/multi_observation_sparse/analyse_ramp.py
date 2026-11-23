"""
EN: **the ramp-basis reading, against a neutral baseline.**

    Reads `out/.../diagnose/<tag>/records.json` and prints one row per (K, L, arm): held-out agreement, the gain over
    the neutral (uniform-θ) plan with its spread across seeds, `cost_rel`, and the support readings. Gain is the
    per-seed difference `arm − neutral`, so its spread is the spread of the difference, not of two means.

    θ error is **not** printed as a headline: for this family it is a diagnostic (see the note). It is available in
    the records and is summarized by `--theta` if asked.

CN: **ramp 基读数，对照中性基线。** 读取 `out/.../diagnose/<tag>/records.json`，每个 (K, L, arm) 打印一行：留出复现率、
相对中性（均匀 θ）计划的增益及其跨种子离散度、`cost_rel`、支撑读数。增益为逐种子差值 `arm − neutral`，故其离散度是差值
的离散度。θ 误差**不作为**头条：对该族它只是诊断量（见笔记），记录中仍有，`--theta` 可打印。

    python -m experiments.studies.iot.synthetic.multi_observation_sparse.analyse_ramp --tag ramp4
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

OUT = Path("experiments/out/multi_observation_sparse/diagnose")


def _mean_sd(values: np.ndarray) -> str:
    """EN: `mean±sd` with one seed's spread left undefined. CN: `mean±sd`，单种子时离散度未定义。"""
    if values.size < 2:
        return f"{values.mean():.3f}±n/a"
    return f"{values.mean():.3f}±{values.std(ddof=1):.3f}"


def main(argv: list[str] | None = None) -> int:
    """EN: print the per-cell table. CN: 打印逐格表。"""
    parser = argparse.ArgumentParser(description="ramp table against the neutral baseline")
    parser.add_argument("--tag", default="ramp4")
    parser.add_argument("--theta", action="store_true")
    args = parser.parse_args(argv)
    payload = json.loads((OUT / args.tag / "records.json").read_text())
    records = payload["records"]
    cells: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for record in records:
        cells[record["geometry"]][record["arm"]].append(record)

    print(f"{'cell':14s} {'arm':13s} {'draws':>5s} {'agreement':>15s} {'gain vs neutral':>17s} "
          f"{'cost_rel':>15s} {'mass':>6s} {'w>zero':>7s}")
    for geometry in sorted(cells):
        neutral = np.array([item["metrics"]["agreement"] for item in cells[geometry]["E_neutral"]])
        for arm in ("E_neutral", "C_joint", "B_averaged", "B_informative", "D_oracle"):
            group = cells[geometry].get(arm)
            if not group:
                continue
            agreement = np.array([item["metrics"]["agreement"] for item in group])
            cost = np.array([item["metrics"]["cost_rel"] for item in group])
            share = np.array([item["metrics"]["observed_mass_in_weighted_bins"] for item in group])
            zero = np.array([item["metrics"]["weight_on_zero_mass_ramps"] for item in group])
            gain = "—" if arm == "E_neutral" else _mean_sd(agreement - neutral)
            print(f"{geometry:14s} {arm:13s} {len(group):5d} {_mean_sd(agreement):>15s} {gain:>17s} "
                  f"{_mean_sd(cost):>15s} {share.mean():6.3f} {zero.mean():7.3f}")
        zero_gradient = np.array([item["metrics"]["windows_at_zero_gradient"]
                                  for item in cells[geometry]["C_joint"]])
        draws = np.array([item["metrics"]["theta_star_draws"] for item in cells[geometry]["C_joint"]])
        capped = np.array([item["metrics"]["hit_iteration_cap"] for item in cells[geometry]["C_joint"]])
        n_windows = cells[geometry]["B_averaged"][0]["metrics"]["windows_averaged"]
        print(f"{'':14s} skipped windows (zero gradient at iteration 1) {zero_gradient.mean():.1f}/{n_windows:.0f}, "
              f"draws to accept {draws.mean():.1f}, capped {capped.mean():.1f}")
        if args.theta:
            star = [np.asarray(item["extra"]["theta_star"], dtype=np.float64)
                    for item in cells[geometry]["C_joint"]]
            fitted = [np.asarray(item["extra"]["theta"], dtype=np.float64)
                      for item in cells[geometry]["C_joint"]]
            distance = np.array([np.abs(one - two).sum() for one, two in zip(fitted, star, strict=True)])
            print(f"{'':14s} theta diagnostic |theta-theta*|_1 = {_mean_sd(distance)} (not a headline)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
