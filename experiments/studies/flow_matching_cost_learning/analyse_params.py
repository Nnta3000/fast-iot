"""
EN: Compare the parametric costs with the L2 control and the v2 10k baselines. CN: 将参数化代价与 L2 控制及 v2 10k 基线比较。
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

OUT = Path("experiments/out/flow_matching_cost_learning")
V2 = Path("experiments/out/records/flow_matching_coupling_v2/eval.json")
STEPS = 10000
NFES = (1, 2, 5, 10)


def _load(path: Path) -> list[dict]:
    """EN: records. CN: 记录。"""
    return json.loads(path.read_text())["records"]


def _table(records: list[dict]) -> dict[tuple[str, str, int], dict[int, float]]:
    """EN: (dataset, arm, seed) -> nfe -> sw2. CN: 同上。"""
    table: dict[tuple[str, str, int], dict[int, float]] = defaultdict(dict)
    for record in records:
        config = record["config"]
        if config.get("sampler") != "euler" or int(config.get("steps", -1)) != STEPS:
            continue
        nfe = int(config.get("nfe", -1))
        if nfe in NFES and "sw2" in record["metrics"]:
            table[(record["geometry"], record["arm"], int(record["draw"]))][nfe] = record["metrics"]["sw2"]
    return table


def _mean_sd(values: list[float]) -> str:
    """EN: mean±sd. CN: mean±sd。"""
    array = np.asarray(values, dtype=float)
    return f"{array.mean():.4f}±{array.std(ddof=1):.4f}" if array.size > 1 else f"{array.mean():.4f}"


def main() -> int:
    """EN: entry point. CN: 入口。"""
    fixed = OUT / "fixed_costs" / "records.json"
    parametric = OUT / "parametric" / "records.json"
    if not fixed.exists() or not parametric.exists() or not V2.exists():
        raise SystemExit(f"missing records: {fixed}, {parametric}, {V2}")
    table = _table([*_load(V2), *_load(fixed), *_load(parametric)])
    params = sorted({arm for (dataset, arm, seed) in table
                     if arm.startswith("cost_") and arm not in ("cost_l2", "cost_pca", "cost_rbf")})
    for dataset in ("eight_gaussians", "two_moons"):
        print(f"\n== {dataset} (10k, Euler sliced W2, mean±sd)")
        print(f"{'arm':20s} " + " ".join(f"{'NFE'+str(n):>16s}" for n in NFES))
        for arm in ("independent", "ot_pool_B256", "global_ot", "cost_l2", "cost_pca", "cost_rbf", *params):
            seeds = sorted({seed for (name, label, seed) in table if name == dataset and label == arm})
            if not seeds:
                continue
            cells = [np.array([table[(dataset, arm, seed)][n] for seed in seeds
                               if n in table.get((dataset, arm, seed), {})]) for n in NFES]
            print(f"{arm:20s} " + " ".join(f"{_mean_sd(cell.tolist()):>16s}" if cell.size else f"{'—':>16s}"
                                           for cell in cells))
        print("paired vs cost_l2 (same seeds; negative is better):")
        for arm in params:
            for nfe in (1, 2, 5, 10):
                values = []
                for (name, label, seed), cells in table.items():
                    if name == dataset and label == arm and nfe in cells:
                        base = table.get((dataset, "cost_l2", seed), {}).get(nfe)
                        if base is not None:
                            values.append(cells[nfe] - base)
                if len(values) >= 2:
                    array = np.asarray(values)
                    sd = float(array.std(ddof=1))
                    threshold = 2.0 * sd
                    wins = int((array < -threshold).sum())
                    print(f"  {arm:20s} NFE{nfe:2d}: {array.mean():+.4f}±{sd:.4f}  "
                          f"2sd {threshold:.4f}  wins {wins}/{array.size}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
