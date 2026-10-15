"""
CN: 实验的统一记录 schema：`RunSpec`（输入）+ `RunRecord`（结果）+ JSON/CSV 写出。
EN: The unified experiment record schema: `RunSpec` (input) + `RunRecord` (result) + JSON/CSV IO.

CN: 所有研究共用这一 schema，因此不同研究的结果可以直接合并、画图、引用（AGENTS §7.1 的可引用性
    规则由此自动化：每个数字都带着它的 RunSpec）。
EN: Every study shares this schema, so results from different studies merge, plot, and cite
    directly. This automates AGENTS §7.1's "a number needs its config" rule: every record
    carries its RunSpec.
"""

from __future__ import annotations

import csv
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path


def _slug(value: str) -> str:
    """CN: 把名字转成文件名安全的形式。EN: Turn a name into a filename-safe slug."""
    return re.sub(r"[^0-9A-Za-z]+", "-", value).strip("-")


@dataclass(frozen=True)
class RunSpec:
    """
    CN: 一次运行的输入规格；三个轴即 family / marginal / (m, n, point_dimension)。
    EN: Input spec of one run; the three axes are family / marginal / (m, n, point_dimension).
    """

    family: str
    marginal: str
    m: int
    n: int
    point_dimension: int = 4
    lowrank_k: int = 8
    seed: int = 0
    solver: str = "auto"
    scan_backend: str = "hello"
    plan_backend: str = "auto"
    scan_device: str = "auto"
    prune: bool = True
    prune_hysteresis: bool = True
    cupdlpx_time_limit: float | None = None
    cupdlpx_warm_start: bool = False
    max_iterations: int = 300
    num_itermax: int = 2_000_000
    validate: bool = False
    tag: str = ""

    def __post_init__(self) -> None:
        if int(self.m) < 2 or int(self.n) < 2:
            raise ValueError("m and n must be >= 2")
        if int(self.max_iterations) < 1:
            raise ValueError("max_iterations must be >= 1")
        if int(self.num_itermax) < 1:
            raise ValueError("num_itermax must be >= 1")
        if self.solver not in {"auto", "regularized", "sdp", "cupdlpx"}:
            raise ValueError("solver must be one of auto/regularized/sdp/cupdlpx")
        if self.scan_backend not in {"numpy", "hello"}:
            raise ValueError("scan_backend must be 'numpy' or 'hello'")
        if self.plan_backend not in {"auto", "hello", "pot"}:
            raise ValueError("plan_backend must be one of auto/hello/pot")

    def run_id(self) -> str:
        """CN: 稳定运行 id。EN: Stable run id."""
        base = (
            f"{self.family}__{_slug(self.marginal)}__m{int(self.m)}_n{int(self.n)}"
            f"_d{int(self.point_dimension)}_k{int(self.lowrank_k)}_seed{int(self.seed)}"
        )
        return f"{base}__{self.tag}" if self.tag else base


@dataclass(frozen=True)
class RunRecord:
    """
    CN: 一次运行的结果：判决（verdict）+ 代价（cost）。`is_optimal` 是权威判据（POT）。
    EN: One run's result: verdict + cost. `is_optimal` is the authoritative check (POT).
    """

    spec: RunSpec
    n_features: int
    support_size: int
    n_components: int
    iterations: int
    converged: bool
    delta: float
    final_max_violation: float
    max_support_over_m_plus_n: float
    is_optimal: bool | None
    relative_gap: float | None
    certified: bool
    cert_support_residual: float
    min_positive_plan_entry: float
    outcome: str
    error: str | None
    scaffold_seconds: float
    solve_seconds: float
    mean_master_seconds: float
    mean_oracle_seconds: float
    trajectory: tuple[dict[str, float], ...] = ()

    @property
    def run_id(self) -> str:
        """CN: 便捷访问 spec.run_id。EN: Convenience access to spec.run_id."""
        return self.spec.run_id()

    def flatten(self) -> dict[str, object]:
        """CN: 展平成 CSV 行（spec 字段加前缀 spec_；轨迹不进 CSV）。EN: Flatten to a CSV row (spec_* prefixed; trajectory excluded)."""
        row: dict[str, object] = {}
        for key, value in asdict(self.spec).items():
            row[f"spec_{key}"] = value
        for key, value in asdict(self).items():
            if key not in ("spec", "trajectory"):
                row[key] = value
        row["run_id"] = self.run_id
        return row


def write_json(path: Path, records: list[RunRecord], *, meta: dict[str, object] | None = None) -> None:
    """CN: 写出结果 JSON（含可选的顶层 meta）。EN: Write records to JSON with optional top-level meta."""
    payload: dict[str, object] = {"meta": dict(meta or {}), "records": [asdict(record) for record in records]}
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def write_csv(path: Path, records: list[RunRecord]) -> None:
    """CN: 写出结果 CSV（每个 record 一行）。EN: Write records to CSV (one row per record)."""
    if not records:
        return
    rows = [record.flatten() for record in records]
    fieldnames = list(rows[0].keys())
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


__all__ = ["RunRecord", "RunSpec", "write_csv", "write_json"]
