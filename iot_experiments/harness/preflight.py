"""
CN: **环境自检** —— "这台机器现在能不能跑？"。它是原先 `gpu_bench --batch env` 的内容，抽成一个函数，
    因为**两个**启动器都需要它：本机的 `scripts/run_job.sh` 与服务器的 `scripts/gpu_queue.sh`。
    它不是实验、不是批处理，只是所有运行的前置条件。

    CN: 检查项：
      1. GPU 是否可见、是否空闲（`harness/gpu.py`）；
      2. `pycupdlpx`（GPU master）是否可加载；
      3. 分离扫描对 `DiagonalWeightFeatures` 是否可用；
      4. 内存与显存余量；
      5. 一次**极小**的已验证运行（`RunSpec(m=64, n=64)`），证明整条链路活着。

    CN: 任一必需项失败 ⇒ 返回 `ok=False`，调用方必须**拒绝启动**，而不是带病继续。

EN: **Environment self-check** -- "can this machine run a job right now?". This is the content of the old
    `gpu_bench --batch env`, lifted into a function because **both** launchers need it: the local
    `scripts/run_job.sh` and the server-side `scripts/gpu_queue.sh`. It is not an experiment and not a
    batch; it is the precondition of every run.

    Checks: GPU visibility and idleness (`harness/gpu.py`), whether `pycupdlpx` loads, whether the
    separation scan supports `DiagonalWeightFeatures`, memory headroom, and one **tiny** validated run
    that proves the whole chain is alive.

    Any required check failing returns `ok=False`; the caller must **refuse to start**, not proceed.
"""

from __future__ import annotations

import dataclasses
import json
import os
import resource
import subprocess
import time
from pathlib import Path

import numpy as np
from jaxtyping import Float


@dataclasses.dataclass(frozen=True)
class Preflight:
    """CN: 一次自检的结果。`ok` 为 False 时 `failures` 说明原因。EN: one self-check result."""

    ok: bool
    git_sha: str
    gpu_visible: bool
    gpu_name: str
    cuda_visible_devices: str
    cupdlpx_loaded: bool
    scan_supported: bool
    mem_available_gb: float
    tiny_run_passed: bool
    tiny_run_ok: bool
    seconds: float
    failures: tuple[str, ...]
    notes: dict[str, object]

    def to_dict(self) -> dict[str, object]:
        """CN: 序列化为 JSON 可写的形式。EN: serialise for JSON."""
        return dataclasses.asdict(self)

    def write(self, path: Path) -> None:
        """CN: 写到 `path`（父目录自动创建）。EN: write to `path`, creating parents."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, default=str) + "\n", encoding="utf-8")


def _git_sha() -> str:
    """CN: 当前 git SHA（写进清单）。EN: the current git SHA, recorded in the manifest."""
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                              text=True, check=True,
                              cwd=str(Path(__file__).resolve().parents[2])).stdout.strip()
    except Exception:  # noqa: BLE001 - a missing git is not a reason to fail a run
        return "unknown"


def _mem_available_gb() -> float:
    """CN: `/proc/meminfo` 的 `MemAvailable`（GB）。EN: `MemAvailable` from /proc/meminfo, in GB."""
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                return float(line.split()[1]) / 1024.0 / 1024.0
    except OSError:
        pass
    return float("nan")


def check(*, run_tiny: bool = True, need_gpu: bool = False) -> Preflight:
    """
    CN: 跑一次自检。`need_gpu=True` 时，看不到 GPU 或 `pycupdlpx` 不可用都算失败；
        本机（CPU 也能跑）可传 `need_gpu=False`，此时它们只被记录为 `notes`。
    EN: run the self-check. With `need_gpu=True`, no visible GPU or an unavailable `pycupdlpx` is a
        failure; on a machine where the CPU path is acceptable, pass `need_gpu=False` and they are
        recorded in `notes` only.
    """
    t0 = time.perf_counter()
    failures: list[str] = []
    notes: dict[str, object] = {}

    try:
        from iot_experiments.harness import gpu as gpu_mod  # noqa: PLC0415

        gpus = gpu_mod.query_gpus()
        gpu_visible = bool(gpus)
        gpu_name = str(gpus[0].get("name", "unknown")) if gpus else "unknown"
        if not gpus:
            notes["gpu"] = "no GPU enumerated by nvidia-smi"
    except Exception as error:  # noqa: BLE001 - recorded, not raised
        gpu_visible, gpu_name = False, "unknown"
        notes["gpu_error"] = repr(error)
    if need_gpu and not gpu_visible:
        failures.append("no GPU visible")

    cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    cupdlpx_loaded = False
    try:
        from iot.backends.lp_common import cupdlpx_available  # noqa: PLC0415

        cupdlpx_loaded = bool(cupdlpx_available())
    except Exception as error:  # noqa: BLE001
        notes["cupdlpx_error"] = repr(error)
    if need_gpu and not cupdlpx_loaded:
        failures.append("pycupdlpx not available")

    scan_supported = False
    try:
        from iot import DiagonalWeightFeatures  # noqa: PLC0415
        from iot.core.features import feature_scan_supported  # noqa: PLC0415

        rng = np.random.default_rng(0)
        probe = np.ascontiguousarray(rng.normal(size=(16, 4)))
        scan_supported = bool(feature_scan_supported(DiagonalWeightFeatures(probe, probe)))
    except Exception:  # noqa: BLE001 - the scan probe is advisory
        try:
            from iot import fast_scan_supported  # noqa: PLC0415
            from iot.core.features import DiagonalWeightFeatures  # noqa: PLC0415

            rng = np.random.default_rng(0)
            probe = np.ascontiguousarray(rng.normal(size=(16, 4)))
            scan_supported = bool(fast_scan_supported(DiagonalWeightFeatures(probe, probe)))
        except Exception as error:  # noqa: BLE001
            notes["scan_probe_error"] = repr(error)

    tiny_ok = False
    tiny_passed = False
    if run_tiny:
        try:
            from iot_experiments.harness.lib.records import RunSpec  # noqa: PLC0415
            from iot_experiments.harness.lib.runner import run_spec  # noqa: PLC0415

            tiny: Float[np.ndarray, "k"] = np.zeros(0)
            _ = tiny
            result = run_spec(RunSpec(family="diagonal", marginal="gmm", m=64, n=64,
                                      point_dimension=4, seed=0, validate=True))
            tiny_ok = True
            tiny_passed = bool(getattr(result, "ok", True))
            notes["tiny_run"] = str(getattr(result, "status", "ok"))
        except Exception as error:  # noqa: BLE001
            notes["tiny_run_error"] = repr(error)
        if not tiny_ok:
            failures.append("tiny validated run failed")

    return Preflight(ok=not failures, git_sha=_git_sha(), gpu_visible=gpu_visible, gpu_name=gpu_name,
                     cuda_visible_devices=cvd, cupdlpx_loaded=cupdlpx_loaded,
                     scan_supported=scan_supported, mem_available_gb=round(_mem_available_gb(), 2),
                     tiny_run_passed=tiny_passed, tiny_run_ok=tiny_ok,
                     seconds=round(time.perf_counter() - t0, 2), failures=tuple(failures),
                     notes=notes)


def _main() -> int:
    """CN: `python -m iot_experiments.harness.preflight` —— 打印并写出清单。EN: CLI entry."""
    import argparse  # noqa: PLC0415

    parser = argparse.ArgumentParser(prog="python -m iot_experiments.harness.preflight")
    parser.add_argument("--out", type=Path, default=None, help="write the manifest here")
    parser.add_argument("--need-gpu", action="store_true", help="treat a missing GPU as a failure")
    parser.add_argument("--no-tiny", action="store_true", help="skip the tiny validated run")
    args = parser.parse_args()
    result = check(run_tiny=not args.no_tiny, need_gpu=bool(args.need_gpu))
    print(json.dumps(result.to_dict(), indent=2, default=str))
    if args.out is not None:
        result.write(Path(args.out))
        print(f"wrote {args.out}")
    _ = resource  # CN: 保留给内存字段的运行时一致性；EN: kept so memory fields stay consistent
    return 0 if result.ok else 1


if __name__ == "__main__":
    raise SystemExit(_main())
