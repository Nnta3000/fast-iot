"""
CN: torch 运行时工具：惰性加载 torch、解析 `auto/cpu/cuda` 设备、报告后端能力。
    本模块是 `src/iot` 中唯一接触 torch 的地方，且**绝不** import `hello_ot`。
EN: torch runtime helpers: lazy torch import, `auto/cpu/cuda` device resolution, backend capability
    reporting. This is the only module in `src/iot` that touches torch, and it **never** imports
    `hello_ot`.
"""

from __future__ import annotations

from typing import Any

from ..core.problem import InverseOTError


def require_torch() -> Any:
    """
    CN: 返回 torch 模块；缺失时抛出带安装提示的错误（纯 torch 用户不需要 native 扩展）。
    EN: Return the torch module; raise with an install hint when missing (pure-torch users need no
        native extension).
    """
    try:
        import torch  # noqa: PLC0415 - lazy by design
    except ImportError as error:  # pragma: no cover - environment dependent
        raise InverseOTError(
            "the torch scan requires `torch`; install it (`pip install torch`) or use "
            "scan_backend='numpy'."
        ) from error
    return torch


def torch_available() -> bool:
    """CN: torch 是否可用。EN: Whether torch is importable."""
    try:
        require_torch()
        return True
    except InverseOTError:
        return False


def cuda_available() -> bool:
    """CN: torch 是否可用且 CUDA 可用。EN: Whether torch is importable and CUDA is available."""
    if not torch_available():
        return False
    torch = require_torch()
    return bool(torch.cuda.is_available())


def resolve_torch_device(value: str) -> Any:
    """
    CN: 解析 `auto`/`cpu`/`cuda`：`auto` 在 CUDA 可用时选 cuda，否则 cpu；显式 cuda 不可用则 fail-fast。
    EN: Resolve `auto`/`cpu`/`cuda`: `auto` picks cuda when available else cpu; an explicit cuda
        request fails fast when unavailable.
    """
    torch = require_torch()
    requested = str(value).strip().lower()
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise InverseOTError("torch_device='cuda' requested but CUDA is not available")
    if device.type not in {"cpu", "cuda"}:
        raise ValueError("torch_device must resolve to CPU or CUDA")
    return device


def describe_backend(device: str = "auto") -> dict[str, object]:
    """CN: 描述 torch 后端（用于 `env` 自检）。EN: Describe the torch backend (for the `env` self-check)."""
    info: dict[str, object] = {"torch_available": torch_available(), "cuda_available": cuda_available()}
    if info["torch_available"]:
        torch = require_torch()
        info["torch_version"] = str(torch.__version__)
        info["resolved_device"] = str(resolve_torch_device(device))
        if cuda_available():
            info["device_name"] = str(torch.cuda.get_device_name(0))
    return info


__all__ = ["require_torch", "torch_available", "cuda_available", "resolve_torch_device", "describe_backend"]
