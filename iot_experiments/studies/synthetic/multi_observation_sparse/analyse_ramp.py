"""
EN: **the ramp-basis reading: does a monotone (hinge) basis identify θ better with more observations?**

    Reads `out/.../diagnose/ramp3/records.json` and prints one row per (K, L) cell: the two arms' held-out
    agreement and scale-free cost error, how far the fitted θ is from θ*, and the two support readings. The table
    is the evidence for the "Addendum 2" correction in `notes/MULTI_OBSERVATION_SPARSE_BUNDLE.md`; it recomputes
    nothing that is not in the records.

CN: **ramp 基读数：单调（hinge）基在观测更多时是否更好地钉住 θ？** 读取
    `out/.../diagnose/ramp3/records.json`，每个 (K, L) 格子打印一行：两臂的留出复现率与无量纲代价误差、拟合 θ 与
    θ* 的距离，以及两个支撑读数。该表是 `notes/MULTI_OBSERVATION_SPARSE_BUNDLE.md` 中"附录 2"更正的证据；
    不重算任何记录之外的东西。
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

RECORDS = Path("iot_experiments/out/multi_observation_sparse/diagnose/ramp3/records.json")


def main() -> int:
    """EN: print the per-cell table. CN: 打印逐格表。"""
    payload = json.loads(RECORDS.read_text())
    rows = payload["records"]
    cells: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        cells[row["geometry"]][row["arm"]].append(row)
    header = (f"{'cell':14s} {'arm':10s} {'draws':>5s} {'agreement':>17s} {'cost_rel':>17s} "
              f"{'|theta-theta*|_1':>17s} {'mass_share':>10s} {'w_on_zero_mass':>14s}")
    print(header)
    for geometry in sorted(cells):
        for arm in ("C_joint", "B_averaged", "D_oracle"):
            group = cells[geometry][arm]
            agreement = np.array([item["metrics"]["agreement"] for item in group])
            cost = np.array([item["metrics"]["cost_rel"] for item in group])
            share = np.array([item["metrics"]["observed_mass_in_weighted_bins"] for item in group])
            zero = np.array([item["metrics"]["weight_on_zero_mass_ramps"] for item in group])
            distance = np.array([np.abs(np.asarray(item["extra"]["theta"], dtype=np.float64)
                                        - np.asarray(item["extra"]["theta_star"], dtype=np.float64)).sum()
                                 for item in group])
            print(f"{geometry:14s} {arm:10s} {len(group):5d} "
                  f"{agreement.mean():.3f}±{agreement.std(ddof=1):.3f}   "
                  f"{cost.mean():.3f}±{cost.std(ddof=1):.3f}   "
                  f"{distance.mean():.3f}±{distance.std(ddof=1):.3f}   "
                  f"{share.mean():10.3f} {zero.mean():14.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
