"""
EN: **Tables for the flow-matching coupling study, v2.**

    Reads `precheck.json`, `train.json` (plan/train times) and `eval.json` (metrics at the NFE grid), and prints:

      * the coupling diagnostic;
      * sliced W2 at Euler NFE 1/2/5/10/50 and the floor;
      * the gain over `fixed_random` at Euler 5 and 10, with the per-seed sd and the standard error;
      * flow straightness (displacement / path length, 1 = straight);
      * mode weights, centre and covariance errors for the mixtures;
      * an **equal-wall-clock** table: per arm, the largest-step run whose plan + training time fits a budget.

    Every number is read from the records; one CSV is written next to them.

CN: **flow-matching 耦合研究 v2 的表格。** 读取 `precheck.json`、`train.json`（计划/训练时间）与 `eval.json`（NFE 网格指标），
    打印：耦合诊断；Euler NFE 1/2/5/10/50 的切片 W2 与下限；相对 `fixed_random` 在 Euler 5/10 的增益（逐种子 sd 与标准误）；
    flow 直线度；混合的模态权重、中心与协方差误差；以及**等墙钟**表（每个臂取计划+训练时间在预算内的最大步数运行）。每个数都
    来自记录，并在旁边写一个 CSV。

    python -m experiments.studies.flow_matching.analyse --tag coupling_v2
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

OUT = Path("experiments/out/flow_matching")
ARMS = ("independent", "fixed_random", "ot_pool_B256", "ot_pool_B1024", "global_ot")
EULER = (1, 2, 5, 10, 50)


def _mean_sd(values: np.ndarray) -> str:
    """EN: `mean±sd` over seeds. CN: 跨种子的 `mean±sd`。"""
    return f"{values.mean():.3f}±n/a" if values.size < 2 else f"{values.mean():.3f}±{values.std(ddof=1):.3f}"


def _mean_sd_sem(values: np.ndarray) -> str:
    """EN: `mean±sd (sem)` with `sem = sd/√n`. CN: `mean±sd (标准误)`，`sem=sd/√n`。"""
    if values.size < 2:
        return f"{values.mean():.3f}"
    sem = values.std(ddof=1) / np.sqrt(values.size)
    return f"{values.mean():.3f}±{values.std(ddof=1):.3f} (sem {sem:.3f})"


def _precheck(records: list[dict]) -> None:
    """EN: print the coupling diagnostic. CN: 打印耦合诊断。"""
    print(f"{'dataset':15s} {'pool':>5s} {'differing':>10s} {'msd local':>10s} {'msd global':>10s} {'cost gap':>9s}")
    for record in records:
        m = record["metrics"]
        print(f"{record['geometry']:15s} {record['config']['pool']:5d} {m['differing']:10.3f} "
              f"{m['msd_local']:10.3f} {m['msd_global']:10.3f} {m['cost_gap']:+9.1%}")
    print()


def _quality(eval_rows: list[dict]) -> None:
    """EN: sliced W2 at the Euler grid, per dataset/steps/arm. CN: Euler 网格上的切片 W2，按数据集/步数/臂。"""
    cells: dict[tuple, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    floor: dict[str, float] = {}
    for record in eval_rows:
        if record["config"]["sampler"] != "euler":
            continue
        key = (record["geometry"], record["arm"], record["config"]["steps"])
        cells[key][record["config"]["nfe"]].append(record["metrics"]["sw2"])
        floor[record["geometry"]] = record["metrics"]["floor_sw2"]
    print(f"{'dataset':15s} {'steps':>6s} {'arm':14s} " + " ".join(f"{'NFE'+str(n):>9s}" for n in EULER))
    for dataset in sorted({key[0] for key in cells}):
        print(f"{dataset:15s} floor sliced W2 = {floor[dataset]:.4f}")
        for steps in (1000, 10000):
            for arm in ARMS:
                values = [cells.get((dataset, arm, steps), {}).get(nfe, []) for nfe in EULER]
                if not values[0]:
                    continue
                print(f"{'':15s} {steps:6d} {arm:14s} " +
                      " ".join(f"{np.mean(v):9.3f}" if v else f"{'—':>9s}" for v in values))
        print()


def _gains(eval_rows: list[dict]) -> None:
    """EN: per-seed gain over fixed_random at Euler 5 and 10, sd and sem. CN: Euler 5/10 相对 fixed_random 的逐种子增益、sd 与标准误。"""
    table: dict[tuple, dict[str, dict[int, float]]] = defaultdict(lambda: defaultdict(dict))
    for record in eval_rows:
        if record["config"]["sampler"] != "euler" or record["config"]["nfe"] not in (5, 10):
            continue
        key = (record["geometry"], record["config"]["steps"], record["config"]["nfe"])
        table[key][record["arm"]][record["draw"]] = record["metrics"]["sw2"]
    print("gain over fixed_random (per-seed arm-minus-control; negative is better)")
    for (dataset, steps, nfe), arms in sorted(table.items()):
        control = arms.get("fixed_random", {})
        for arm in ARMS:
            if arm == "fixed_random" or arm not in arms:
                continue
            differences = np.array([value - control[seed] for seed, value in arms[arm].items() if seed in control])
            if differences.size:
                print(f"  {dataset:15s} t{steps:<6d} NFE {nfe:<3d} {arm:14s} {_mean_sd_sem(differences)}")
    print()


def _diagnostics(eval_rows: list[dict]) -> None:
    """EN: straightness, and mode metrics at Euler 50. CN: 直线度与 Euler 50 的模态指标。"""
    print("straightness (displacement / path length; 1 = straight) and mode diagnostics at Euler 50")
    for dataset in sorted({record["geometry"] for record in eval_rows}):
        for arm in ARMS:
            rows = [record for record in eval_rows if record["geometry"] == dataset and record["arm"] == arm
                    and record["config"]["sampler"] == "euler" and record["config"]["nfe"] == 50]
            if not rows:
                continue
            straight = np.array([record["metrics"]["straightness"] for record in rows])
            line = f"  {dataset:15s} {arm:14s} straightness {_mean_sd(straight)}"
            if "mode_weight_l1" in rows[0]["metrics"] and dataset != "two_moons":
                weights = np.array([record["metrics"]["mode_weight_l1"] for record in rows])
                means = np.array([record["metrics"]["mode_mean_error"] for record in rows])
                covs = np.array([record["metrics"]["mode_cov_error"] for record in rows])
                line += f" | mode L1 {weights.mean():.3f} mean {means.mean():.3f} cov {covs.mean():.3f}"
            print(line)
    print()


def _wall_clock(train_rows: list[dict], eval_rows: list[dict]) -> None:
    """EN: per arm, the largest-step run that fits a budget; quality is sliced W2 at Euler 10. CN: 每个臂取预算内最大步数的运行；质量为 Euler 10 的切片 W2。"""
    timing = {(record["geometry"], record["arm"], record["config"]["steps"], record["draw"]):
              record["metrics"]["plan_seconds"] + record["metrics"]["train_seconds"] for record in train_rows}
    quality = {(record["geometry"], record["arm"], record["config"]["steps"], record["draw"]):
               record["metrics"]["sw2"] for record in eval_rows
               if record["config"]["sampler"] == "euler" and record["config"]["nfe"] == 10}
    print("equal wall-clock: best run within the budget (plan + training), sliced W2 at Euler 10")
    for budget in (10.0, 40.0, 120.0):
        print(f"  budget {budget:.0f}s")
        for dataset in sorted({key[0] for key in timing}):
            for arm in ARMS:
                candidates = [(timing[key], key[2], quality.get(key, float("nan")))
                              for key in timing if key[0] == dataset and key[1] == arm and timing[key] <= budget]
                if not candidates:
                    continue
                best = max(candidates, key=lambda item: item[1])
                print(f"    {dataset:15s} {arm:14s} {best[1]:6d} steps, {best[0]:6.1f}s, W2 {best[2]:.3f}")
        print()


def main(argv: list[str] | None = None) -> int:
    """EN: entry point. CN: 入口。"""
    parser = argparse.ArgumentParser(description="flow-matching coupling tables (v2)")
    parser.add_argument("--tag", default="coupling_v2")
    args = parser.parse_args(argv)
    root = OUT / args.tag
    precheck = json.loads((root / "precheck.json").read_text())["records"]
    train = json.loads((root / "train.json").read_text())["records"]
    evaluation = json.loads((root / "eval.json").read_text())["records"]
    _precheck(precheck)
    _quality(evaluation)
    _gains(evaluation)
    _diagnostics(evaluation)
    _wall_clock(train, evaluation)

    csv_path = root / "table.csv"
    with csv_path.open("w", encoding="utf-8") as handle:
        handle.write("dataset,arm,steps,seed,sampler,nfe,sw2,floor_sw2,straightness,energy,mode_weight_l1,"
                     "mode_mean_error,mode_cov_error\n")
        for record in evaluation:
            m = record["metrics"]
            handle.write(",".join(str(value) for value in (
                record["geometry"], record["arm"], record["config"]["steps"], record["draw"],
                record["config"]["sampler"], record["config"]["nfe"], m["sw2"], m["floor_sw2"],
                m["straightness"], m.get("energy", ""), m.get("mode_weight_l1", ""),
                m.get("mode_mean_error", ""), m.get("mode_cov_error", ""))) + "\n")
    print(f"wrote {csv_path} ({len(evaluation)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
