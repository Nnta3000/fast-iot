"""
EN: **Tables and the pre-registered decision for the coupling-headroom study.**

    Stage 0 reads the v2 flow-matching records for the baselines (`independent`, `w2_mb` = `ot_pool_B256`,
    `global_ot`) and the headroom records for `mac_full`, `teacher_routed_ot`, `reflow_ref`, pairs them by seed and
    reports sliced W2 at NFE 1/2/5/10, the paired gain over `w2_mb`, and the compute of the coupling.

    Stage 1 reads the CIFAR-10 records and reports FID at NFE 2/4/8/32 with the real and test floors, plus
    straightness and the paired decision at NFE 4.

CN: **耦合 headroom 研究的表格与预定判据。** Stage 0 读 v2 记录的基线（`independent`、`w2_mb`=`ot_pool_B256`、
    `global_ot`）与 headroom 记录的 `mac_full`、`teacher_routed_ot`、`reflow_ref`，按种子配对，报 NFE 1/2/5/10 的切片 W2、
    相对 `w2_mb` 的配对增益与耦合计算量。Stage 1 读 CIFAR-10 记录，报 NFE 2/4/8/32 的 FID 与真实/测试下限、直线度，以及
    NFE 4 的配对判据。

    python -m experiments.studies.flow_matching_headroom.analyse --stage toys
    python -m experiments.studies.flow_matching_headroom.analyse --stage cifar
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

OUT = Path("experiments/out")
V2 = OUT / "records/flow_matching_coupling_v2/eval.json"
TOYS = sorted((OUT / "flow_matching_headroom").glob("stage0*/records.json"))
CIFAR = sorted((OUT / "flow_matching_headroom").glob("cifar*/records.json"))
CIFAR_STEPS = 50000


def _load(paths: list[Path]) -> list[dict]:
    """EN: merge records from one or more tags. CN: 合并一个或多个 tag 的记录。"""
    rows: list[dict] = []
    for path in paths:
        rows.extend(json.loads(path.read_text())["records"])
    return rows
ARMS = ("independent", "ot_pool_B256", "global_ot", "mac_full", "teacher_routed_ot", "reflow_ref")


def _mean_sd(values: np.ndarray) -> str:
    """EN: `mean±sd`. CN: `mean±sd`。"""
    return f"{values.mean():.3f}±n/a" if values.size < 2 else f"{values.mean():.3f}±{values.std(ddof=1):.3f}"


def _sw2_table(records: list[dict], *, steps: int,
               nfes: tuple[int, ...] = (1, 2, 5, 10)) -> dict[tuple, dict[int, float]]:
    """EN: `(geometry, arm, seed) -> nfe -> sw2`, Euler only, **one training budget**. CN: 同上，且只取一个训练预算。"""
    table: dict[tuple, dict[int, float]] = defaultdict(dict)
    for record in records:
        if record["config"].get("sampler") != "euler":
            continue
        # A table must never mix budgets: the v2 file contains both 1k and 10k records.
        if int(record["config"].get("steps", -1)) != int(steps):
            continue
        nfe = int(record["config"]["nfe"])
        if nfe in nfes and "sw2" in record["metrics"]:
            table[(record["geometry"], record["arm"], int(record["draw"]))][nfe] = record["metrics"]["sw2"]
    return table


def toys() -> None:
    """EN: Stage 0 tables and decision. CN: Stage 0 表与判据。"""
    if not V2.exists() or not TOYS:
        raise SystemExit(f"missing records: {V2} or {TOYS}")
    baseline_records = json.loads(V2.read_text())["records"]
    headroom_records = _load(TOYS)
    table = _sw2_table([*baseline_records, *headroom_records], steps=10000)
    datasets = sorted({key[0] for key in table})
    for dataset in datasets:
        print(f"\n== {dataset}")
        print(f"{'arm':18s} " + " ".join(f"{'NFE'+str(n):>15s}" for n in (1, 2, 5, 10)))
        for arm in ARMS:
            seeds = sorted({key[2] for key in table if key[0] == dataset and key[1] == arm})
            if not seeds:
                continue
            cells = [np.array([table[(dataset, arm, seed)][nfe] for seed in seeds if nfe in table[(dataset, arm, seed)]])
                     for nfe in (1, 2, 5, 10)]
            print(f"{arm:18s} " + " ".join(f"{_mean_sd(cell):>15s}" if cell.size else f"{'—':>15s}" for cell in cells))
        print(f"{'gain vs w2_mb':18s} (paired; negative is better)")
        for arm in ("mac_full", "teacher_routed_ot", "reflow_ref"):
            line = []
            for nfe in (1, 2):
                seeds = sorted({key[2] for key in table if key[0] == dataset and key[1] == "ot_pool_B256"})
                differences = np.array([table[(dataset, arm, seed)][nfe] - table[(dataset, "ot_pool_B256", seed)][nfe]
                                        for seed in seeds
                                        if nfe in table.get((dataset, arm, seed), {})
                                        and nfe in table.get((dataset, "ot_pool_B256", seed), {})])
                if differences.size:
                    sem = differences.std(ddof=1) / np.sqrt(differences.size) if differences.size > 1 else float("nan")
                    line.append(f"NFE{nfe} {differences.mean():+.3f}±{differences.std(ddof=1):.3f} "
                                f"(sem {sem:.3f}, {int(np.sum(differences < 0))}/{differences.size} better)")
            if line:
                print(f"  {arm:16s} " + " | ".join(line))
        for arm in ("mac_full", "teacher_routed_ot", "reflow_ref"):
            rows = [record for record in headroom_records if record["geometry"] == dataset and record["arm"] == arm]
            if not rows:
                continue
            metrics = rows[0]["metrics"]
            print(f"  compute {arm:16s} train {metrics['train_seconds']:.1f}s "
                  f"coupling {metrics['coupling_seconds']:.1f}s teacher {metrics['teacher_seconds']:.1f}s "
                  f"bank {metrics['bank_seconds']:.1f}s extra-forwards/step {metrics['extra_forward_evals']:.0f}")
    print("\npre-registered decision: headroom at NFE 1-2 only if the better new arm is below `w2_mb` by more than "
          "2 standard errors; otherwise 'no headroom at this scale', no tuning.")


def cifar() -> None:
    """EN: Stage 1 FID table and decision. CN: Stage 1 FID 表与判据。"""
    if not CIFAR:
        raise SystemExit(f"missing records: {CIFAR}")
    records = [record for record in _load(CIFAR)
               if int(record["config"].get("steps", -1)) == CIFAR_STEPS]
    if not records:
        raise SystemExit(f"no CIFAR records with steps={CIFAR_STEPS}")
    floors = {record["metrics"].get("fid_floor_real") for record in records}
    floors_test = {record["metrics"].get("fid_floor_test") for record in records}
    print(f"FID floors: held-out real {sorted(v for v in floors if v is not None)}, "
          f"test {sorted(v for v in floors_test if v is not None)}")
    table: dict[tuple, dict[int, float]] = defaultdict(dict)
    for record in records:
        if record["config"]["nfe"] > 0 and "fid" in record["metrics"]:
            table[(record["arm"], int(record["draw"]))][int(record["config"]["nfe"])] = record["metrics"]["fid"]
    arms = sorted({key[0] for key in table})
    nfes = sorted({nfe for cells in table.values() for nfe in cells})
    print(f"{'arm':18s} " + " ".join(f"{'NFE'+str(nfe):>15s}" for nfe in nfes))
    for arm in arms:
        seeds = sorted({key[1] for key in table if key[0] == arm})
        cells = [np.array([table[(arm, seed)][nfe] for seed in seeds if nfe in table[(arm, seed)]]) for nfe in nfes]
        print(f"{arm:18s} " + " ".join(f"{_mean_sd(cell):>15s}" if cell.size else f"{'—':>15s}" for cell in cells))
    print("straightness:")
    for record in records:
        if record["config"]["nfe"] == 0 and "straightness" in record["metrics"]:
            print(f"  {record['arm']:18s} s{record['draw']} {record['metrics']['straightness']:.3f} "
                  f"train {record['metrics']['train_seconds']:.0f}s coupling {record['metrics']['coupling_seconds']:.0f}s")
    if 4 in nfes:
        print("decision at NFE 4 (paired vs w2_mb; headroom if gain < -max(1.0, 2 sd) in >= 2/3 seeds)")
        for arm in ("mac_full", "teacher_routed_ot"):
            if arm not in arms or "w2_mb" not in arms:
                continue
            seeds = sorted({key[1] for key in table if key[0] == arm and key[0] == arm})
            for seed in seeds:
                if 4 in table.get((arm, seed), {}) and 4 in table.get(("w2_mb", seed), {}):
                    difference = table[(arm, seed)][4] - table[("w2_mb", seed)][4]
                    print(f"  {arm:16s} s{seed}: {difference:+.2f} FID")
            differences = np.array([table[(arm, seed)][4] - table[("w2_mb", seed)][4] for seed in seeds
                                    if 4 in table.get((arm, seed), {}) and 4 in table.get(("w2_mb", seed), {})])
            if differences.size >= 2:
                threshold = max(1.0, 2.0 * differences.std(ddof=1))
                wins = int(np.sum(differences < -threshold))
                print(f"  {arm:16s} mean {differences.mean():+.2f}, sd {differences.std(ddof=1):.2f}, "
                      f"threshold {threshold:.2f}, wins {wins}/{differences.size}")


def main(argv: list[str] | None = None) -> int:
    """EN: entry point. CN: 入口。"""
    parser = argparse.ArgumentParser(description="headroom tables")
    parser.add_argument("--stage", required=True, choices=("toys", "cifar"))
    args = parser.parse_args(argv)
    if args.stage == "toys":
        toys()
    else:
        cifar()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
