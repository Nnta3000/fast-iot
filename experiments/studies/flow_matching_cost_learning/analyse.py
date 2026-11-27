"""
EN: Compare the fixed-cost screen with the v2 10k baselines. CN: 将固定代价筛选与 v2 10k 基线比较。
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

OUT = Path("experiments/out/flow_matching_cost_learning")
V2 = Path("experiments/out/records/flow_matching_coupling_v2/eval.json")
STEPS = 10000


def _load(path: Path) -> list[dict]:
    """EN: records list. CN: 记录列表。"""
    return json.loads(path.read_text())["records"]


def _table(records: list[dict]) -> dict[tuple[str, str, int], dict[int, float]]:
    """EN: (dataset, arm, seed) -> nfe -> sw2, 10k Euler only. CN: 同上，仅 10k Euler。"""
    table: dict[tuple[str, str, int], dict[int, float]] = defaultdict(dict)
    for record in records:
        config = record["config"]
        if config.get("sampler") != "euler" or int(config.get("steps", -1)) != STEPS:
            continue
        nfe = int(config.get("nfe", -1))
        if nfe in (1, 2, 5, 10) and "sw2" in record["metrics"]:
            table[(record["geometry"], record["arm"], int(record["draw"]))][nfe] = record["metrics"]["sw2"]
    return table


def _mean_sd(values: list[float]) -> str:
    """EN: mean±sd. CN: mean±sd。"""
    array = np.asarray(values, dtype=float)
    return f"{array.mean():.4f}±{array.std(ddof=1):.4f}" if array.size > 1 else f"{array.mean():.4f}"


def main() -> int:
    """EN: entry point. CN: 入口。"""
    if not V2.exists():
        raise SystemExit(f"missing {V2}")
    ours = OUT / "fixed_costs" / "records.json"
    if not ours.exists():
        raise SystemExit(f"missing {ours}")
    table = _table([*_load(V2), *_load(ours)])
    for dataset in ("eight_gaussians", "two_moons"):
        print(f"\n== {dataset} (10k, Euler sliced W2)")
        print(f"{'arm':18s} " + " ".join(f"{'NFE'+str(nfe):>16s}" for nfe in (1, 2, 5, 10)))
        for arm in ("independent", "ot_pool_B256", "global_ot", "cost_l2", "cost_rbf", "cost_pca"):
            seeds = sorted({seed for (name, label, seed) in table if name == dataset and label == arm})
            if not seeds:
                continue
            cells = [np.array([table[(dataset, arm, seed)][nfe] for seed in seeds
                               if nfe in table.get((dataset, arm, seed), {})]) for nfe in (1, 2, 5, 10)]
            print(f"{arm:18s} " + " ".join(f"{_mean_sd(cell.tolist()):>16s}" if cell.size else f"{'—':>16s}"
                                           for cell in cells))
        print("paired vs v2 ot_pool_B256, seeds present in both (negative is better):")
        for arm in ("cost_l2", "cost_rbf", "cost_pca"):
            for nfe in (1, 2, 5, 10):
                differences = []
                for (name, label, seed), cells in table.items():
                    if name == dataset and label == arm and nfe in cells:
                        base = table.get((dataset, "ot_pool_B256", seed), {}).get(nfe)
                        if base is not None:
                            differences.append(cells[nfe] - base)
                if differences:
                    array = np.asarray(differences)
                    print(f"  {arm:10s} NFE{nfe:2d}: {array.mean():+.4f}±{array.std(ddof=1):.4f}  "
                          f"better {int((array < 0).sum())}/{array.size}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
