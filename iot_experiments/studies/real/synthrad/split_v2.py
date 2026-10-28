"""
CN: **v2 患者划分** —— 分层（每中心 60 人）、固定种子、写入 manifest。

    CN: 60/中心 = 180 人，按你同意的比例：
        * **FIT 105**（35/中心）—— 学习代价在此拟合；**包含 Phase 1 的 30 个 probe 患者**
          （已见过的患者只进 fit，绝不进 eval/test）；
        * **DEV 45**（15/中心）—— 开发评估，所有 arm、所有 draw 共用同一批人；
        * **FINAL TEST 30**（10/中心）—— **在最终一次运行前不触碰**。

    CN: 写入 manifest 时可复现：种子固定，且记录每组的患者 id 与每中心计数。
    EN: **v2 patient split** -- stratified (60 per centre), fixed seed, written to a manifest.

        180 patients = **FIT 105** (35/centre, including the 30 Phase-1 probe patients, since already-seen
        patients go to fit and never to eval/test), **DEV 45** (15/centre, shared by every arm and draw) and
        **FINAL TEST 30** (10/centre, untouched until one final run).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[4]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from iot_experiments.harness.records import RunLog  # noqa: E402
from iot_experiments.harness.synthrad import PatientPaths, discover_patients  # noqa: E402

SPLIT_SEED: int = 2024
FIT_PER_CENTRE: int = 35
DEV_PER_CENTRE: int = 15
TEST_PER_CENTRE: int = 10


class SplitError(RuntimeError):
    """CN: 划分前置条件失败。EN: a split precondition failed."""


def stratified_split(patients: list[PatientPaths], *, probe_force_fit: set[str],
                     seed: int = SPLIT_SEED) -> dict[str, object]:
    """
    CN: 每个中心先放 probe 患者进 FIT，再从剩余里补足 FIT 到 `FIT_PER_CENTRE`，然后是 DEV、TEST。
        三个集合**互不相交且覆盖全部**患者。
    EN: per centre, place the probe patients in FIT first, top FIT up to `FIT_PER_CENTRE`, then DEV, then
        TEST. The three sets are **disjoint and cover every patient**.
    """
    by_centre: dict[str, list[PatientPaths]] = {}
    for p in patients:
        by_centre.setdefault(p.centre, []).append(p)
    rng = np.random.default_rng(int(seed))
    fit: list[str] = []
    dev: list[str] = []
    test: list[str] = []
    for centre in sorted(by_centre):
        group = sorted(by_centre[centre], key=lambda p: p.patient)
        names = [p.patient for p in group]
        forced = [n for n in names if n in probe_force_fit]
        rest = [n for n in names if n not in probe_force_fit]
        rng.shuffle(rest)
        need = int(FIT_PER_CENTRE) - len(forced)
        if need < 0:
            raise SplitError(f"centre {centre}: {len(forced)} probe patients exceed the fit budget")
        if len(rest) < need + int(DEV_PER_CENTRE) + int(TEST_PER_CENTRE):
            raise SplitError(f"centre {centre}: only {len(rest)} patients after probes; "
                             f"need {need + DEV_PER_CENTRE + TEST_PER_CENTRE}")
        taken = rest[:need]
        remaining = rest[need:]
        fit += forced + taken
        dev += remaining[: int(DEV_PER_CENTRE)]
        test += remaining[int(DEV_PER_CENTRE): int(DEV_PER_CENTRE) + int(TEST_PER_CENTRE)]
    all_ids = sorted(fit + dev + test)
    if len(all_ids) != len(set(all_ids)):
        raise SplitError("the three sets overlap")
    if len(all_ids) != len(patients):
        raise SplitError(f"split covers {len(all_ids)} of {len(patients)} patients")
    return {
        "seed": int(seed),
        "fit": sorted(fit),
        "dev": sorted(dev),
        "test": sorted(test),
        "counts": {"fit": len(fit), "dev": len(dev), "test": len(test)},
        "per_centre": {
            "fit": {c: sum(1 for n in fit if n[2] == c) for c in sorted(by_centre)},
            "dev": {c: sum(1 for n in dev if n[2] == c) for c in sorted(by_centre)},
            "test": {c: sum(1 for n in test if n[2] == c) for c in sorted(by_centre)},
        },
        "probe_in_fit": sorted(n for n in fit if n in probe_force_fit),
    }


def main(argv: list[str] | None = None) -> int:
    """CN: 生成并保存 manifest。EN: build and save the manifest."""
    parser = argparse.ArgumentParser(description="SynthRAD brain v2 stratified split")
    parser.add_argument("--out", type=Path,
                        default=_REPO_ROOT / "iot_experiments" / "out" / "synthrad_v2")
    args = parser.parse_args(argv)
    log = RunLog(args.out / "split.log")
    patients = discover_patients(_REPO_ROOT / "data" / "Task1", anatomy="brain")
    probe_manifest = _REPO_ROOT / "iot_experiments" / "reports" / "synthrad_probe" / "split_manifest.json"
    probe = set(json.loads(probe_manifest.read_text())["probe"]) if probe_manifest.is_file() else set()
    log.line(f"brain patients: {len(patients)}; Phase-1 probe patients forced into FIT: {len(probe)}")
    split = stratified_split(patients, probe_force_fit=probe)
    path = args.out / "split_v2.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(split, indent=2) + "\n", encoding="utf-8")
    log.line(f"counts: {split['counts']}")
    for group in ("fit", "dev", "test"):
        log.line(f"  {group} per centre: {split['per_centre'][group]}")
    log.line(f"probe patients placed in FIT: {len(split['probe_in_fit'])}")
    log.line(f"wrote {path}")
    log.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
