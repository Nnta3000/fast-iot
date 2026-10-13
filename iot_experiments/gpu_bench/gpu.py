"""
CN: 在无 SLURM 的服务器上把整个进程**只**绑到一个空闲 GPU 上，并记录环境清单。
EN: Pin a whole process to a single *idle* GPU on a SLURM-less server, and record an env
    manifest.

CN: 关键点：必须在 **import torch 之前** 设置 `CUDA_VISIBLE_DEVICES`，因此 `configure()`
    要在任何 `iot`/torch import 之前调用。只暴露一个设备，避免任何"多卡并行"的意外。
EN: Key point: `CUDA_VISIBLE_DEVICES` must be set *before* torch is imported, so `configure()`
    runs before any `iot`/torch import. Exactly one device is exposed, so no accidental
    multi-GPU use is possible.
"""

from __future__ import annotations

import os
import subprocess


def _smi(fields: str) -> list[list[str]]:
    """CN: 运行 nvidia-smi 查询并解析为 CSV。EN: Run an nvidia-smi query and parse the CSV."""
    completed = subprocess.run(
        ["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        check=True,
    )
    rows: list[list[str]] = []
    for line in completed.stdout.splitlines():
        if line.strip():
            rows.append([cell.strip() for cell in line.split(",")])
    return rows


def query_gpus() -> list[dict[str, object]]:
    """CN: 列出所有 GPU 的占用与利用率。EN: List every GPU's memory use and utilization."""
    gpus: list[dict[str, object]] = []
    for row in _smi("index,name,memory.used,memory.total,utilization.gpu"):
        gpus.append(
            {
                "index": int(row[0]),
                "name": row[1],
                "memory_used_mb": int(row[2]),
                "memory_total_mb": int(row[3]),
                "utilization": int(row[4]),
            }
        )
    return gpus


def pick_idle_gpu(*, max_used_mb: int = 1024) -> int:
    """CN: 返回第一个空闲 GPU 的物理序号；没有则报错。EN: First idle GPU's physical index, else fail."""
    gpus = query_gpus()
    idle = [
        gpu for gpu in gpus
        if int(gpu["memory_used_mb"]) <= int(max_used_mb) and int(gpu["utilization"]) == 0
    ]
    if not idle:
        raise SystemExit(
            "no idle GPU found (all have >= %d MiB used or nonzero utilization); "
            "check nvidia-smi and pass --gpu N explicitly" % int(max_used_mb)
        )
    return int(idle[0]["index"])


def configure(choice: object) -> int:
    """
    CN: 解析 `--gpu`（"auto" 或物理序号），设置 `CUDA_VISIBLE_DEVICES`，并打印一次。
        若环境里已有 `CUDA_VISIBLE_DEVICES` 且选择为 auto，则**尊重它**（便于整会话锁定单卡）。
    EN: Resolve `--gpu` ("auto" or a physical index), set `CUDA_VISIBLE_DEVICES`, print once.
        If `CUDA_VISIBLE_DEVICES` is already set and the choice is auto, respect it (pins the
        whole session to one card).
    """
    existing = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if (choice is None or choice == "auto") and existing not in ("", "-1"):
        first = existing.split(",")[0].strip()
        print(f"[gpu] respecting existing CUDA_VISIBLE_DEVICES={existing}", flush=True)
        return int(first)
    if choice is None or choice == "auto":
        index = pick_idle_gpu()
    else:
        index = int(choice)  # type: ignore[arg-type]
    os.environ["CUDA_VISIBLE_DEVICES"] = str(index)
    match = next((g for g in query_gpus() if int(g["index"]) == index), None)
    if match is not None and int(match["utilization"]) != 0:
        print(
            f"[gpu] WARNING: physical GPU {index} is busy "
            f"(util={match['utilization']}%, used={match['memory_used_mb']}MiB)",
            flush=True,
        )
    detail = "" if match is None else f" name={match['name']}"
    print(f"[gpu] pinned CUDA_VISIBLE_DEVICES={index}{detail}", flush=True)
    return index


def device_info() -> dict[str, object]:
    """CN: torch/CUDA/POT/native 扩展的版本与可用性。EN: torch/CUDA/POT/native-extension versions."""
    info: dict[str, object] = {"cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")}
    try:
        import torch

        info["torch"] = str(torch.__version__)
        info["cuda_available"] = bool(torch.cuda.is_available())
        info["cuda_version"] = str(torch.version.cuda)
        if torch.cuda.is_available():
            info["device_name"] = str(torch.cuda.get_device_name(0))
            info["visible_device_count"] = int(torch.cuda.device_count())
    except Exception as error:  # noqa: BLE001 - report any import/runtime failure plainly
        info["torch_error"] = repr(error)
    try:
        import ot

        info["pot"] = str(ot.__version__)
    except Exception as error:  # noqa: BLE001
        info["pot_error"] = repr(error)
    info["pycupdlpx"] = _pycupdlpx_available()
    return info


def _pycupdlpx_available() -> bool:
    """CN: 检测 CuPDLPx native 扩展是否可加载。EN: Whether the CuPDLPx native extension loads."""
    try:
        import torch  # noqa: F401 - load libtorch DSOs first
    except Exception:  # noqa: BLE001
        pass
    for module in ("hello_ot._native.pycupdlpx", "pycupdlpx"):
        try:
            __import__(module)
            return True
        except Exception:  # noqa: BLE001
            continue
    return False


__all__ = ["configure", "device_info", "pick_idle_gpu", "query_gpus"]
