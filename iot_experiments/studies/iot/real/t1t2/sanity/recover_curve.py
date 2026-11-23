"""
EN: **Recover check B's records from its run log, because the run died before writing them.**

    The learning-curve run (`sanity/run.py --check curve`) completed 11 of its 12 cells and then died with
    `cudaErrorMemoryAllocation` in the CuPDLPx master on the last cell (`K=256, m=600`). The driver writes
    `records.json` only at the end, so nothing was written; the measurements survive **only** in
    `out/sanity/curve/run.log`.

    This script parses that log back into `RunRecord`s and writes `records.recovered.json` **next to** the
    log. It does not touch the (missing) `records.json` and it never invents a missing cell:

      - every recovered row carries `"recovered_from": "run.log"` and `"complete": false` in `extra`;
      - the missing cell is written as an `excluded=True` row with `status="NOT RECORDED: run died ..."`,
        so any table built from the file shows the hole instead of silently averaging over it;
      - the `hit` values are the log's 4-decimal output, so they are **rounded at the fourth decimal** and
        are not bit-identical to what the driver would have written;
      - `config` is **not** recoverable from the log line and is left empty with a pointer to the command.

    CN: **从 `run.log` 恢复检查 B 的记录**：运行在最后一个单元因 GPU 显存不足而死，驱动只在结束时写文件，
        因此测量值只存在于日志里。本脚本把它们解析回 `RunRecord`，写到日志旁边的 `records.recovered.json`，
        **不伪造缺失单元**，并且每一行都带 `recovered_from` 与 `complete=false`。

    Run: `.venv/bin/python -m iot_experiments.studies.iot.real.t1t2.sanity.recover_curve`
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[5]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from iot_experiments.harness.records import RunRecord, format_table, summarise  # noqa: E402

OUT = _REPO_ROOT / "iot_experiments" / "out" / "sanity" / "curve"
COMMAND = (".venv/bin/python -u -m iot_experiments.studies.iot.real.t1t2.sanity.run --check curve "
           "--ks 64 128 256 --ms 50 100 300 600 --draws 0")

PROVENANCE: dict[str, object] = {
    "status": "RECOVERED FROM LOG -- not a driver-written record",
    "why": "the run died with cudaErrorMemoryAllocation in the CuPDLPx master on the last cell "
           "(K=256, m=600); the driver writes records.json only at the end, so nothing was written",
    "source": "iot_experiments/out/sanity/curve/run.log (unchanged; preflight.json also present)",
    "command": COMMAND,
    "precision": "hit/R@1 are the log's 4-decimal output -- rounded, not bit-identical to a driver record",
    "completeness": "11 of 12 cells; the missing cell is present as an excluded row, never as a number",
    "pool": "ONE fixed 400-pair held-out pool; labelled subjects disjoint from it (600 available)",
    "arm": "F2's iot arm (select_iot -> fit_full_lp, box x simplex, rel chosen by 5-fold CV on labelled pairs)",
    "cosine": "label-free, so identical at every m for a given K -- recorded once per K by the driver",
    "reading": "SATURATED if hit(600) - hit(300) < 0.03, RISING otherwise (rule fixed before the run)",
}

_LINE = re.compile(
    r"\[(K=(?P<k>\d+))\] cosine hit=(?P<chit>[\d.]+) R@1=(?P<cr1>[\d.]+)")
_CELL = re.compile(
    r"\[K=(?P<k>\d+) m=(?P<m>\d+)\] iot hit=(?P<hit>[\d.]+) R@1=(?P<r1>[\d.]+) "
    r"\(\+?(?P<delta>-?[\d.]+) pts vs cosine\) (?P<secs>\d+)s sel=(?P<sel>\{.*\})")


def main() -> int:
    """CN: 入口。EN: entry point."""
    log_path = OUT / "run.log"
    text = log_path.read_text(encoding="utf-8")
    cosine = {int(m.group("k")): (float(m.group("chit")), float(m.group("cr1")))
              for m in _LINE.finditer(text)}
    cells = [(int(m.group("k")), int(m.group("m")), float(m.group("hit")), float(m.group("r1")),
              int(m.group("secs")), m.group("sel")) for m in _CELL.finditer(text)]
    if not cells:
        raise SystemExit(f"no iot cells found in {log_path}; nothing to recover")

    records: list[RunRecord] = []
    ks = sorted(cosine)
    ms = sorted({c[1] for c in cells} | {600})
    for k in ks:
        hit, r1 = cosine[k]
        records.append(RunRecord(
            study="sanity", geometry="G2|pool400", arm="COSINE", k=k, m=min(ms), draw=0,
            metrics={"hit": hit, "R@1": r1}, config={},
            extra={"label_free": True, "recovered_from": "run.log", "complete": False,
                   "note": "cosine is label-free; recorded once per K by the driver"}))
        for kk, mm, h, rr, secs, sel in cells:
            if kk != k:
                continue
            records.append(RunRecord(
                study="sanity", geometry="G2|pool400", arm="iot", k=kk, m=mm, draw=0,
                metrics={"hit": h, "R@1": rr}, config={},
                extra={"selected": sel, "seconds_reported": secs, "delta_pts_vs_cosine": h - hit,
                       "recovered_from": "run.log", "complete": False}))

    present = {(r.k, r.m) for r in records if r.arm == "iot" and not r.excluded}
    for k in ks:
        for m in ms:
            if (k, m) not in present:
                records.append(RunRecord(
                    study="sanity", geometry="G2|pool400", arm="iot", k=k, m=m, draw=0,
                    metrics={}, config={}, excluded=True,
                    status="NOT RECORDED: run died (cudaErrorMemoryAllocation) before this cell wrote",
                    extra={"recovered_from": "run.log", "complete": False}))

    path = OUT / "records.recovered.json"
    path.write_text(json.dumps({"provenance": PROVENANCE,
                                "records": [r.to_dict() for r in records]}, indent=2) + "\n",
                    encoding="utf-8")
    table = summarise(records, group=("arm", "k", "m"), metrics=("hit",))
    print(format_table(table, ("arm", "k", "m", "n_draws", "hit_mean")))
    print(f"wrote {path} ({len(records)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
