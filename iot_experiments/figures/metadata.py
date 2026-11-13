"""
EN: **Figure metadata** -- the `.csv` of plotted points and the `.caption.md` that sits beside the PNG.

    Both live in `figures/metadata/`, not in the git-ignored `figures/out/`, because "what is plotted here"
    has to survive a clone. A caption records facts (axes, instance, methods, config, sources, conventions)
    and **no takeaway**: interpretation is the reader's job (see `FIGURE_STYLE.md` §7).

    CN: **图的元数据** —— 每个被画的点一行的 `.csv`，以及与之配套的 `.caption.md`；两者都放在
        `figures/metadata/`（不在被 git 忽略的 `out/` 里）。卡片只记事实，不写结论。
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

from iot_experiments.figures.style import METADATA_DIR


class FigureError(RuntimeError):
    """CN: 记录缺失或数值不符。EN: a record is missing or a value disagrees."""


def load_records(path: Path) -> tuple[list[dict], dict]:
    """
    CN: 读一个 `records.json`，返回 `(records, provenance)`。缺文件即抛错，绝不静默返回空。
    EN: read a `records.json` and return `(records, provenance)`. A missing file raises; never returns empty.
    """
    if not path.is_file():
        raise FigureError(f"missing record: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload["records"] if isinstance(payload, dict) else payload
    return [dict(row) for row in rows], dict(payload.get("provenance", {})) if isinstance(payload, dict) else {}


def write_metadata(name: str, *, caption: dict[str, str], header: list[str],
                   rows: list[list[object]]) -> tuple[Path, Path]:
    """
    CN: 写出 `<name>.csv` 与 `<name>.caption.md`。CSV 每个被画的点一行；卡片按给定字段顺序。
    EN: write `<name>.csv` (one row per plotted point) and `<name>.caption.md` (fields in the given order).
    """
    METADATA_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = METADATA_DIR / f"{name}.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)
    caption_path = METADATA_DIR / f"{name}.caption.md"
    body = "\n".join(f"**{key}:** {value}" for key, value in caption.items() if value)
    caption_path.write_text(f"# {name}\n\n{body}\n", encoding="utf-8")
    return csv_path, caption_path


def print_caption(name: str, *, experiment: str, comparing: str, config: str) -> None:
    """CN: 生成时把同一段页脚打到 stdout。EN: print the same footer text to stdout on generation."""
    print(f"{name}:\n  {experiment}\n  {comparing}\n  {config}")


def check_values(name: str, got: dict[str, float], expected: dict[str, float], *, tol: float = 5e-3) -> None:
    """
    CN: 把从记录解析出的数与预期比对，超过 `tol` 即停止。EN: compare parsed values with the expected ones
        and stop beyond `tol`, so a figure cannot drift away from the brief silently.
    """
    problems = [f"{key}: record {got[key]:.4f} vs expected {want:.4f} (diff {got[key] - want:+.4f})"
                for key, want in expected.items() if key in got and abs(got[key] - want) > tol]
    missing = [key for key in expected if key not in got]
    if missing:
        problems.append(f"not present in the record: {missing}")
    if problems:
        raise FigureError(f"{name} disagrees with the expected values:\n  " + "\n  ".join(problems))


__all__ = ["FigureError", "check_values", "load_records", "print_caption", "write_metadata"]
