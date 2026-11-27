"""
EN: Phase D timing gate. Profile w2_mb at pool 512 and 1024 on the assigned GPU, parse the `[profile]` total
    ms/step, and write `experiments/out/flow_matching_headroom/pool_gate_ok` only if a 50k run would stay under the
    requested wall-clock budget. The all-phases jobs file makes the two pool runs conditional on that marker.

CN: Phase D 计时门。在当前分配的 GPU 上对 pool 512/1024 的 w2_mb 做 profile，解析 `[profile]` 的 ms/step；
    只有 50k 运行低于预算时才写 `pool_gate_ok`。all-phases jobs 文件据此决定是否运行两个 pool 作业。

    python -m experiments.studies.flow_matching_headroom.pool_gate --threshold-hours 2.0
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

OUT = Path("experiments/out/flow_matching_headroom")
MARKER = OUT / "pool_gate_ok"


def _profile(pool: int, *, steps: int) -> float:
    """EN: run one 100-step profile and return the worst reported total ms/step. CN: 跑一次 100 步 profile，返回最差总 ms/step。"""
    tag = f"profile_pool{pool}"
    command = [
        sys.executable, "-m", "experiments.studies.flow_matching_headroom.cifar",
        "--profile-steps", str(steps), "--batch", "128", "--arms", "w2_mb", "--seeds", "0",
        "--pool", str(pool), "--tag", tag,
    ]
    print("+ " + " ".join(command), flush=True)
    subprocess.run(command, check=True)
    log = OUT / tag / "run.log"
    text = log.read_text(encoding="utf-8")
    values = [float(value) for value in re.findall(r"total ([0-9.]+) ms/step", text)]
    if not values:
        raise RuntimeError(f"no profile total in {log}")
    return max(values)


def main(argv: list[str] | None = None) -> int:
    """EN: entry point. CN: 入口。"""
    parser = argparse.ArgumentParser(description="Phase D pool timing gate")
    parser.add_argument("--threshold-hours", type=float, default=2.0)
    parser.add_argument("--profile-steps", type=int, default=100)
    args = parser.parse_args(argv)
    MARKER.unlink(missing_ok=True)
    rows: list[tuple[int, float, float]] = []
    for pool in (512, 1024):
        milliseconds = _profile(pool, steps=args.profile_steps)
        hours = milliseconds * 50_000.0 / 3_600_000.0
        rows.append((pool, milliseconds, hours))
        print(f"pool {pool}: {milliseconds:.1f} ms/step -> {hours:.2f} h for 50k", flush=True)
    passed = all(hours <= args.threshold_hours for _, _, hours in rows)
    print(f"GATE {'PASS' if passed else 'FAIL'} (threshold {args.threshold_hours:.2f} h)", flush=True)
    if passed:
        MARKER.parent.mkdir(parents=True, exist_ok=True)
        MARKER.write_text("ok\n", encoding="utf-8")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
