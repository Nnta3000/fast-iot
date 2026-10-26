"""
CN: 逆 OT 实验的共享核心：三个轴（family / marginal / scale）与统一的 RunSpec/RunRecord。
EN: Shared core for the inverse-OT experiments: the three axes (family / marginal / scale)
    and the unified RunSpec/RunRecord.

CN: 研究脚本只 import 这里；它们不直接碰 `benchmark/`（除生成器）或 `iot`。
EN: Study scripts import only this; they do not touch `benchmark/` (beyond the generators) or
    `iot` directly.
"""

from __future__ import annotations

from .families import ALL_FAMILIES, LP_FAMILIES, FamilySpec, build_family
from .marginals import PRESETS, MarginalSpec, marginal_spec
from .records import RunRecord, RunSpec, write_csv, write_json
from .runner import run_grid, run_spec

__all__ = [
    "ALL_FAMILIES",
    "LP_FAMILIES",
    "FamilySpec",
    "MarginalSpec",
    "PRESETS",
    "RunRecord",
    "RunSpec",
    "build_family",
    "marginal_spec",
    "run_grid",
    "run_spec",
    "write_csv",
    "write_json",
]
