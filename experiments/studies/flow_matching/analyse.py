"""
EN: **Tables for the flow-matching coupling study.**

    Reads `experiments/out/flow_matching/<tag>/records.json` (training sweep) and `precheck.json` (the coupling
    diagnostic), and prints:

      * the pre-check: differing partners, both displacements, cost gap;
      * W2 at a low Euler budget (NFE 10) and a low RK4 budget (NFE 20), where a straighter coupling can matter;
      * the gain over `fixed_random`, the control that shares the fixed noise set with `global_ot`;
      * mode coverage for eight-Gaussians.

    Every number is a seed mean ± sd read from the records; the finite-sample floor is printed alongside. One CSV per
    tag is written next to the records so the table can be re-plotted without re-running.

CN: **flow-matching 耦合研究的表格。** 读取 `records.json` 与 `precheck.json`，打印：预检查（伙伴不同比例、两种位移、代价差）；
    低 Euler 预算（NFE 10）与低 RK4 预算（NFE 20）下的 W2；相对 `fixed_random`（与 `global_ot` 共享固定噪声集的对照）的
    增益；八高斯的模态覆盖。每个数都是记录里的种子均值±标准差，并列出有限样本下限。每个 tag 写一个 CSV。

    python -m experiments.studies.flow_matching.analyse --tag coupling_v1
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

OUT = Path("experiments/out/flow_matching")
ARMS = ("independent", "fixed_random", "ot_pool_B256", "ot_pool_B1024", "global_ot")


def _mean_sd(values: np.ndarray) -> str:
    """EN: `mean±sd`, or `mean` when a single seed makes the spread undefined. CN: `mean±sd`，单种子时为 `mean`。"""
    return f"{values.mean():.3f}±n/a" if values.size < 2 else f"{values.mean():.3f}±{values.std(ddof=1):.3f}"


def _table(records: list[dict]) -> None:
    """EN: print the W2 tables and the control-relative gains. CN: 打印 W2 表与相对对照的增益。"""
    cells: dict[tuple, dict[tuple, list[float]]] = defaultdict(lambda: defaultdict(list))
    floor: dict[str, float] = {}
    for record in records:
        key = (record["geometry"], record["arm"], record["config"]["steps"],
               record["config"]["sampler"], record["config"]["nfe"])
        cells[key][record["draw"]].append(record["metrics"]["w2"])
        floor[record["geometry"]] = record["metrics"]["floor_w2"]
    print(f"{'dataset':15s} {'steps':>6s} {'arm':14s} {'euler@10':>15s} {'euler@50':>15s} "
          f"{'rk4@20':>15s} {'cover@10':>8s}")
    for dataset in sorted({key[0] for key in cells}):
        print(f"{dataset:15s} floor W2 = {floor[dataset]:.4f}")
        for steps in (1000, 10000):
            for arm in ARMS:
                values = {}
                for label, sampler, nfe in (("e", "euler", 10), ("e50", "euler", 50), ("r", "rk4", 20)):
                    per_seed = cells.get((dataset, arm, steps, sampler, nfe))
                    values[label] = (np.mean([np.mean(v) for v in per_seed.values()])
                                     if per_seed else float("nan"))
                coverage = [record["metrics"]["coverage"] for record in records
                            if record["geometry"] == dataset and record["arm"] == arm
                            and record["config"]["steps"] == steps and record["config"]["sampler"] == "euler"
                            and record["config"]["nfe"] == 10]
                cover = float(np.nanmean(coverage)) if coverage else float("nan")
                print(f"{'':15s} {steps:6d} {arm:14s} {values['e']:15.3f} {values['e50']:15.3f} "
                      f"{values['r']:15.3f} {cover:8.3f}")
        print()
    print("gain over fixed_random (per-seed arm-minus-control; negative is better)")
    for dataset in sorted({key[0] for key in cells}):
        for steps in (1000, 10000):
            control = {seed: np.mean(v) for seed, v in
                       cells.get((dataset, "fixed_random", steps, "euler", 10), {}).items()}
            for arm in ARMS:
                if arm == "fixed_random":
                    continue
                table = cells.get((dataset, arm, steps, "euler", 10), {})
                differences = np.array([np.mean(v) - control[seed] for seed, v in table.items() if seed in control])
                if differences.size:
                    print(f"  {dataset:15s} t{steps:<6d} {arm:14s} euler@10 {_mean_sd(differences):>16s}")
    print()


def _precheck(records: list[dict]) -> None:
    """EN: print the coupling diagnostic. CN: 打印耦合诊断。"""
    print(f"{'dataset':15s} {'pool':>5s} {'differing':>10s} {'msd local':>10s} {'msd global':>10s} {'cost gap':>9s}")
    for record in records:
        m = record["metrics"]
        print(f"{record['geometry']:15s} {record['config']['pool']:5d} {m['differing']:10.3f} "
              f"{m['msd_local']:10.3f} {m['msd_global']:10.3f} {m['cost_gap']:+9.1%}")
    print()


def main(argv: list[str] | None = None) -> int:
    """EN: entry point. CN: 入口。"""
    parser = argparse.ArgumentParser(description="flow-matching coupling tables")
    parser.add_argument("--tag", default="coupling_v1")
    args = parser.parse_args(argv)
    root = OUT / args.tag
    precheck = json.loads((root / "precheck.json").read_text())["records"]
    records = json.loads((root / "records.json").read_text())["records"]
    _precheck(precheck)
    _table(records)

    rows = []
    for record in records:
        rows.append({"dataset": record["geometry"], "arm": record["arm"], "steps": record["config"]["steps"],
                     "seed": record["draw"], "sampler": record["config"]["sampler"],
                     "nfe": record["config"]["nfe"], "w2": record["metrics"]["w2"],
                     "floor_w2": record["metrics"]["floor_w2"], "coverage": record["metrics"]["coverage"],
                     "coupling_share": record["metrics"]["coupling_share"]})
    csv_path = root / "w2.csv"
    with csv_path.open("w", encoding="utf-8") as handle:
        handle.write("dataset,arm,steps,seed,sampler,nfe,w2,floor_w2,coverage,coupling_share\n")
        for row in rows:
            handle.write(",".join(str(row[key]) for key in
                                  ("dataset", "arm", "steps", "seed", "sampler", "nfe", "w2",
                                   "floor_w2", "coverage", "coupling_share")) + "\n")
    print(f"wrote {csv_path} ({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
