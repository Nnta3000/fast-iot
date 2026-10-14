#!/usr/bin/env python
"""
CN: A100 批处理的**唯一入口**。每个 batch 由固定参数展开，输出带 git SHA 与环境清单到
    `gpu_bench/out/<batch>/`。用法见 `BATCHES.md`。
EN: The *single* entry point for A100 batches. Each batch expands fixed parameters and writes
    outputs (with git SHA + env manifest) to `gpu_bench/out/<batch>/`. See `BATCHES.md`.

CN: 关键顺序：先 `gpu.configure(...)`（在任何 torch import 之前设好 `CUDA_VISIBLE_DEVICES`），
    再惰性 import `iot_experiments.core` / `iot`。
EN: Ordering matters: `gpu.configure(...)` runs first (sets `CUDA_VISIBLE_DEVICES` before any
    torch import), then `iot_experiments.core` / `iot` are imported lazily.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from iot_experiments.gpu_bench import gpu  # noqa: E402  (imports no torch)

_OUT_DIR = Path(__file__).resolve().parent / "out"


def _git_sha() -> str:
    """CN: 当前 git SHA（写入 manifest）。EN: Current git SHA (recorded in the manifest)."""
    try:
        return subprocess.run(
            ["git", "-C", str(_REPO_ROOT), "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def _write(path: Path, payload: dict[str, object]) -> None:
    """CN: 写 JSON（保留 numpy/奇异类型）。EN: Write JSON (tolerating odd types)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")
    print(f"[gpu_bench] wrote {path}", flush=True)


def run_env(args: argparse.Namespace, out_dir: Path) -> dict[str, object]:
    """CN: 环境自检 + 一次极小已验证运行。EN: Env self-check + one tiny validated run."""
    import numpy as np

    from iot_experiments.core import RunSpec, run_spec

    info = gpu.device_info()
    scan_ok = False
    try:
        from iot import DiagonalWeightFeatures, fast_scan_supported

        rng = np.random.default_rng(0)
        probe = DiagonalWeightFeatures(rng.normal(size=(16, 4)), rng.normal(size=(16, 4)))
        scan_ok = bool(fast_scan_supported(probe))
    except Exception as error:  # noqa: BLE001
        info["scan_probe_error"] = repr(error)

    tiny = run_spec(
        RunSpec(
            family="diagonal", marginal="gmm", m=64, n=64, point_dimension=4,
            seed=0, validate=True,
        )
    )
    payload: dict[str, object] = {
        "batch": "env",
        "git_sha": _git_sha(),
        "device": info,
        "diagonal_scan_hello_supported": scan_ok,
        "tiny_run": asdict(tiny),
    }
    _write(out_dir / "env.json", payload)
    return payload


def run_e2e(args: argparse.Namespace, out_dir: Path) -> dict[str, object]:
    """CN: 端到端整环，比较 scan/master 后端。EN: End-to-end full loop comparing scan/master backends."""
    from iot_experiments.core import RunSpec, run_grid

    try:
        from iot import DiagonalWeightFeatures, fast_scan_supported

        if str(args.family) != "diagonal":
            print(
                f"[gpu_bench] NOTE: hello scan only supports diagonal weights; "
                f"family={args.family} will fall back to NumPy (not GPU-scalable).",
                flush=True,
            )
    except Exception:  # noqa: BLE001
        pass

    specs: list[RunSpec] = []
    for master in args.masters:
        for scan in args.scans:
            for size in args.sizes:
                for seed in args.seeds:
                    prune = master != "cupdlpx"
                    specs.append(
                        RunSpec(
                            family=str(args.family), marginal=str(args.marginal),
                            m=int(size), n=int(size),
                            point_dimension=int(args.point_dimension),
                            lowrank_k=int(args.lowrank_k), seed=int(seed),
                            solver=("regularized" if master == "highs" else str(master)),
                            scan_backend=str(scan), prune=bool(prune),
                            plan_backend=str(getattr(args, "plan_backend", "auto")),
                            scan_device=str(getattr(args, "scan_device", "auto")),
                            cupdlpx_time_limit=getattr(args, "cupdlpx_time_limit", None),
                            cupdlpx_warm_start=bool(getattr(args, "cupdlpx_warm_start", False)),
                            max_iterations=int(args.max_iterations), validate=bool(args.validate),
                            tag=f"{master}-{scan}-{'prune' if prune else 'noprune'}",
                        )
                    )
    meta = {"batch": "e2e", "git_sha": _git_sha(), "device": gpu.device_info()}
    run_grid(specs, out=out_dir / "e2e.json", meta=meta)
    return {"batch": "e2e", "n_specs": len(specs), "out": str(out_dir)}


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="A100 batch entry point")
    parser.add_argument("--batch", required=True, choices=("env", "e2e"))
    parser.add_argument("--gpu", default="auto", help='"auto" or a physical GPU index')
    parser.add_argument("--out", type=Path, default=None)
    # e2e
    parser.add_argument("--sizes", type=int, nargs="+", default=[800, 3200])
    parser.add_argument("--point-dimension", type=int, default=8)
    parser.add_argument("--lowrank-k", type=int, default=8)
    parser.add_argument("--scans", nargs="+", default=["hello"])
    parser.add_argument("--masters", nargs="+", default=["cupdlpx"])
    parser.add_argument("--plan-backend", default="auto", choices=("auto", "hello", "pot"))
    parser.add_argument("--scan-device", default="auto")
    parser.add_argument("--cupdlpx-warm-start", action="store_true")
    parser.add_argument("--max-iterations", type=int, default=300)
    parser.add_argument("--validate", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    gpu.configure(args.gpu)  # MUST run before any torch import
    out_dir = args.out or (_OUT_DIR / str(args.batch))
    if args.batch == "env":
        run_env(args, out_dir)
    else:
        run_e2e(args, out_dir)
    print(f"[gpu_bench] batch={args.batch} done; outputs in {out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
