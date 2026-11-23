"""
CN: 在 CUDA >= 12/13 的工具链上构建 HELLO-OT 的 `pycupdlpx` native 扩展，**不修改** HELLO-OT 源码。
    唯一的不兼容点：CUDA 13 的 CCCL 不再传递包含 `thrust/tuple.h`，于是 `solver.cu` 里的
    `thrust::get` / `thrust::make_tuple` 未声明。给 nvcc 加
    `-include <toolkit>/cccl/thrust/tuple.h` 即可修复。
    本脚本生成一个临时 setup.py，从既有源码构建顶层 `pycupdlpx` 模块并安装进当前解释器。

    EN: Build HELLO-OT's `pycupdlpx` native extension against a CUDA >= 12/13 toolkit **without
    modifying HELLO-OT sources**. The only incompatibility: CUDA 13's CCCL no longer pulls in
    `thrust/tuple.h` transitively, so `solver.cu`'s `thrust::get` / `thrust::make_tuple` are
    undeclared. Adding `-include <toolkit>/cccl/thrust/tuple.h` fixes it. This script generates a
    temporary setup.py that builds a top-level `pycupdlpx` module from the existing sources and
    installs it into the current interpreter.

用法 / Usage:
    python iot_experiments/methods/exact/build_pycupdlpx_cuda13.py
    python iot_experiments/methods/exact/build_pycupdlpx_cuda13.py --dry-run   # print only

说明 / Notes:
    - 结果是一个顶层 `pycupdlpx` 模块；`iot.cupdlpx_master._load_pycupdlpx` 会找到它。
      EN: The result is a top-level `pycupdlpx` module; `iot.cupdlpx_master._load_pycupdlpx` finds it.
    - 正式发布路径仍是官方 cu118 环境；本脚本只用于 CUDA 12/13 上的本地实验。
      EN: The official release path is still the cu118 environment; this script is only for local
      experiments on CUDA 12/13.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
_CUPDLPX_DIR = _REPO_ROOT / "src" / "hello_ot" / "_native" / "cupdlpx"

_SETUP_TEMPLATE = """\
from setuptools import setup
import pybind11
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

ROOT = {root!r}
TOOLKIT_TUPLE = {tuple_header!r}

setup(
    name="pycupdlpx",
    version="0.1.0",
    packages=[],
    ext_modules=[
        CUDAExtension(
            name="pycupdlpx",
            sources=[
                f"{{ROOT}}/pycupdlpx/cupdlpx_bindings.cpp",
                f"{{ROOT}}/cupdlpx/io.c",
                f"{{ROOT}}/cupdlpx/utils.cu",
                f"{{ROOT}}/cupdlpx/solver.cu",
                f"{{ROOT}}/cupdlpx/cupdlpx.c",
            ],
            include_dirs=[f"{{ROOT}}/cupdlpx", pybind11.get_include()],
            libraries=["cublas", "cusparse", "z"],
            extra_compile_args={{
                "cxx": ["-O3", "-fpermissive"],
                "nvcc": ["-O3", "-include", TOOLKIT_TUPLE],
            }},
            extra_link_args=["-Wl,-rpath,$ORIGIN", "-Wl,--disable-new-dtags"],
        )
    ],
    cmdclass={{"build_ext": BuildExtension}},
)
"""


def _toolkit_tuple_header() -> Path:
    """
    CN: 找到**工具链**自带的 `thrust/tuple.h`；优先 CCCL 3.x 的 `cccl/thrust/tuple.h`。
    EN: Find the **toolkit's own** `thrust/tuple.h`; prefer CCCL 3.x's `cccl/thrust/tuple.h`.
    """
    import torch
    from torch.utils.cpp_extension import CUDA_HOME

    if not CUDA_HOME:
        raise SystemExit("CUDA_HOME is not set; install a CUDA toolkit first")
    root = Path(CUDA_HOME)
    candidates = (
        root / "targets" / "x86_64-linux" / "include" / "cccl" / "thrust" / "tuple.h",
        root / "include" / "cccl" / "thrust" / "tuple.h",
        root / "include" / "thrust" / "tuple.h",
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    del torch
    raise SystemExit(f"could not find thrust/tuple.h under CUDA_HOME={CUDA_HOME}")


def _arch_list() -> str:
    """CN: 未显式设置时按当前 GPU 推断 `TORCH_CUDA_ARCH_LIST`。EN: Infer the arch list from the GPU."""
    import torch

    if not torch.cuda.is_available():
        return "8.9"
    major, minor = torch.cuda.get_device_capability()
    return f"{major}.{minor}"


def _installer_command(workdir: Path) -> list[str]:
    """CN: 优先 uv，其次 pip；都不可用则报错。EN: Prefer uv, then pip."""
    from shutil import which

    arguments = [
        str(workdir),
        "--no-build-isolation",
        "--no-deps",
        "--force-reinstall",
    ]
    if which("uv"):
        return ["uv", "pip", "install", "--python", sys.executable, *arguments]
    return [sys.executable, "-m", "pip", "install", *arguments]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="print the generated setup.py and exit")
    parser.add_argument("--keep", action="store_true", help="keep the temporary build directory")
    args = parser.parse_args(argv)

    if not _CUPDLPX_DIR.is_dir():
        raise SystemExit(f"CuPDLPx sources not found at {_CUPDLPX_DIR}")
    tuple_header = _toolkit_tuple_header()
    setup_source = _SETUP_TEMPLATE.format(root=str(_CUPDLPX_DIR), tuple_header=str(tuple_header))

    if args.dry_run:
        print(f"# toolkit CCCL tuple header: {tuple_header}")
        print(setup_source)
        return 0

    temporary = Path(tempfile.mkdtemp(prefix="build_pycupdlpx_"))
    (temporary / "setup.py").write_text(setup_source, encoding="utf-8")
    environment = dict(os.environ)
    environment.setdefault("TORCH_CUDA_ARCH_LIST", _arch_list())
    environment.setdefault("MAX_JOBS", "4")
    command = _installer_command(temporary)
    print(f"[build_pycupdlpx] tuple header: {tuple_header}")
    print(f"[build_pycupdlpx] TORCH_CUDA_ARCH_LIST={environment['TORCH_CUDA_ARCH_LIST']}")
    print(f"[build_pycupdlpx] {' '.join(command)}", flush=True)
    completed = subprocess.run(command, env=environment, check=False)
    if not args.keep:
        for child in sorted(temporary.rglob("*"), reverse=True):
            child.unlink() if child.is_file() else child.rmdir()
        temporary.rmdir()
    if completed.returncode != 0:
        return int(completed.returncode)
    print(
        "[build_pycupdlpx] done. Verify with:\n"
        "  python -c \"import torch, pycupdlpx; "
        "s=pycupdlpx.cupdlpx(); print(s.num_variables() if s else 'ok')\"\n"
        "  python -m pytest tests/iot_cupdlpx_master_test.py -q"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
