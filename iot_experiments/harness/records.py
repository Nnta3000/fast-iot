"""
CN: **运行记录与表格** —— 每一次读数都携带产生它的配置，表格与判定从这里生成。

    CN: `RunRecord` 内嵌 `config`（字典快照），因此"说不出来源的数字"无法被写进表格；
        这是 AGENTS §7.1 的要求在代码里的落实。

EN: **Run records and tables** -- every reading carries the configuration that produced it; tables and
    verdicts are generated from here. `RunRecord` embeds the config snapshot, so a number without its
    provenance cannot reach a table.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from collections.abc import Iterable, Mapping, Sequence

import numpy as np
from jaxtyping import Float


@dataclass(frozen=True)
class RunRecord:
    """
    CN: 一行读数：`(study, geometry, arm, K, m, draw)` + 指标 + 配置快照。
        `excluded=True` 的行（失败/超时）**不进入**任何表格或均值。
    EN: one reading: `(study, geometry, arm, K, m, draw)` plus metrics plus the config snapshot.
        Rows with `excluded=True` (failed/timeout) never enter a table or a mean.
    """

    study: str
    geometry: str
    arm: str
    k: int
    m: int
    draw: int
    metrics: Mapping[str, float]
    config: Mapping[str, object] = field(default_factory=dict)
    excluded: bool = False
    status: str = "ok"
    extra: Mapping[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        """CN: JSON 可序列化。EN: JSON-serialisable."""
        return {"study": self.study, "geometry": self.geometry, "arm": self.arm, "k": self.k,
                "m": self.m, "draw": self.draw,
                "metrics": dict(self.metrics), "config": dict(self.config),
                "excluded": bool(self.excluded), "status": self.status,
                "extra": dict(self.extra)}


def mean_std(values: Sequence[float]) -> tuple[float, float]:
    """
    CN: `(mean, std)`；std 用总体标准差（与旧表一致，`np.std` 默认 ddof=0）。
    EN: `(mean, std)` with the population standard deviation (`np.std` default, ddof=0).
    """
    arr: Float[np.ndarray, "draws"] = np.asarray(list(values), dtype=np.float64)
    return float(arr.mean()), float(np.std(arr))


def summarise(records: Iterable[RunRecord], *, group: Sequence[str] = ("geometry", "arm"),
              metrics: Sequence[str] = ("hit", "R@1")) -> list[dict[str, object]]:
    """
    CN: 按 `group` 聚合（排除 `excluded`），每个指标给出 `mean`/`std`/`n`。
    EN: Aggregate by `group` (excluding `excluded` rows); each metric gets `mean`/`std`/`n`.
    """
    buckets: dict[tuple[str, ...], list[RunRecord]] = {}
    for rec in records:
        if rec.excluded:
            continue
        key = tuple(str(getattr(rec, g)) for g in group)
        buckets.setdefault(key, []).append(rec)
    out: list[dict[str, object]] = []
    for key, recs in sorted(buckets.items()):
        row: dict[str, object] = {g: v for g, v in zip(group, key)}
        row["n_draws"] = len(recs)
        for metric in metrics:
            values = [r.metrics[metric] for r in recs if metric in r.metrics]
            if values:
                mean, std = mean_std(values)
                row[f"{metric}_mean"] = mean
                row[f"{metric}_std"] = std
        out.append(row)
    return out


def verdict(learned: Sequence[float], cosine: Sequence[float], *,
            min_gain: float = 0.03) -> str:
    """
    CN: **预注册判定**：learned 在**每一个** draw 上都比 cosine 高至少 `min_gain` → `WIN`；
        全部不高于 cosine → `LOSS`；否则 `TIE`。两臂 draw 数必须相同。
    EN: **Pre-registered verdict**: `WIN` if learned beats cosine by at least `min_gain` on **every** draw;
        `LOSS` if it never does; otherwise `TIE`. Both arms must have the same number of draws.
    """
    if len(learned) != len(cosine) or not learned:
        raise ValueError(f"need equal, non-empty draw counts; got {len(learned)} and {len(cosine)}")
    diffs = [float(a) - float(b) for a, b in zip(learned, cosine)]
    if all(d >= float(min_gain) for d in diffs):
        return "WIN"
    if all(d <= 0.0 for d in diffs):
        return "LOSS"
    return "TIE"


def format_table(rows: Sequence[Mapping[str, object]], columns: Sequence[str]) -> str:
    """
    CN: 生成等宽文本表（用于 stdout）。缺列以 `-` 填充。
    EN: Render a fixed-width text table for stdout; missing columns are filled with `-`.
    """
    def cell(row: Mapping[str, object], col: str) -> str:
        value = row.get(col)
        if value is None:
            return "-"
        if isinstance(value, float):
            return f"{value:.4f}" if abs(value) < 1e3 else f"{value:.3e}"
        return str(value)

    widths = [max(len(col), *(len(cell(r, col)) for r in rows)) if rows else len(col)
              for col in columns]
    header = "  ".join(col.rjust(w) for col, w in zip(columns, widths))
    lines = [header, "  ".join("-" * w for w in widths)]
    for row in rows:
        lines.append("  ".join(cell(row, col).rjust(w) for col, w in zip(columns, widths)))
    return "\n".join(lines)


def write_records(path: Path, records: Sequence[RunRecord], *,
                  provenance: Mapping[str, object] | None = None) -> Path:
    """
    CN: 写 JSON：`{"provenance": {...}, "records": [...]}`。`provenance` 用于记录
        "标准化/PCA 使用池图像（transductive, 无标签）"这类跨表说明。
    EN: Write JSON as `{"provenance": {...}, "records": [...]}`; `provenance` carries cross-table notes
        such as "standardisation/PCA use pool images (transductive, label-free)".
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"provenance": dict(provenance or {}), "records": [r.to_dict() for r in records]}
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


__all__ = ["RunRecord", "format_table", "mean_std", "summarise", "verdict", "write_records"]
