"""
EN: **Analysis for the FAUST study -- tables and figure captions, from the saved records only.**

    `AGENTS.md` §7.3 asks each study to carry an `analyse.py` so that the numbers in a report come from a
    committed script rather than from someone's shell history, and §7.2 asks every figure to carry a
    `<name>.caption.md`. Both live here.

    It never re-runs the solver: it reads `records.json` written by `correspondence.py` / `heatmaps.py` and
    prints (and, for the heat maps, writes) what the report cites.

    Usage:
        .venv/bin/python -m iot_experiments.studies.real.faust.analyse --study correspondence
        .venv/bin/python -m iot_experiments.studies.real.faust.analyse --study heatmaps
        .venv/bin/python -m iot_experiments.studies.real.faust.analyse --study correspondence \\
            --out faust_correspondence_box

    CN: FAUST 研究的**分析脚本**：只读已保存的 `records.json`，产出报告引用的表格，并为热图写 `.caption.md`。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[4]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from iot_experiments.harness.records import (  # noqa: E402
    RunRecord,
    format_table,
    summarise,
)
import re  # noqa: E402


def _load(out: Path) -> tuple[list[RunRecord], dict[str, object]]:
    """CN: 读 `records.json`。EN: read `records.json`."""
    payload = json.loads((out / "records.json").read_text(encoding="utf-8"))
    records = [RunRecord(study=r["study"], geometry=r["geometry"], arm=r["arm"], k=r["k"], m=r["m"],
                         draw=r["draw"], metrics=r["metrics"], config=r["config"],
                         excluded=r["excluded"], status=r["status"], extra=r["extra"])
               for r in payload["records"]]
    return records, payload.get("provenance", {})


def _correspondence(records: list[RunRecord]) -> None:
    """CN: 测试集上的分臂汇总 + 自助法行。EN: per-arm summary on the test pairs plus the bootstrap rows."""
    test = [r for r in records if not r.geometry.startswith("val:") and r.geometry != "bootstrap"]
    table = summarise(test, group=("arm",), metrics=("mean", "miss_1pct", "miss_5pct",
                                                     "flip_rate", "flip_share_of_errors"))
    print(format_table(table, ("arm", "n_draws", "mean_mean", "miss_1pct_mean", "miss_5pct_mean",
                               "flip_rate_mean", "flip_share_of_errors_mean")))
    print()
    for r in records:
        if r.geometry == "bootstrap":
            m = r.metrics
            print(f"{r.arm:<42} mean {m['mean']:+.4f}  CI [{m['ci_lo']:+.4f}, {m['ci_hi']:+.4f}]  "
                  f"wins {int(m['wins'])}/{int(m['n_pairs'])}")
    val = [r for r in records if r.geometry.startswith("val:")]
    if val:
        print("\nvalidation miss@5% (the box-width selection criterion):")
        table = summarise(val, group=("arm",), metrics=("miss_5pct",))
        print(format_table(table, ("arm", "n_draws", "miss_5pct_mean")))


def _heatmaps(records: list[RunRecord], out: Path, provenance: dict[str, object]) -> None:
    """CN: 翻转表。EN: the flip table."""
    table = summarise(records, group=("arm",), metrics=("mean", "miss_5pct", "flip_rate",
                                                        "flip_share_of_errors"))
    print(format_table(table, ("arm", "n_draws", "mean_mean", "miss_5pct_mean",
                               "flip_rate_mean", "flip_share_of_errors_mean")))

    # CN: **每张图一份 caption**（AGENTS.md §7.2）：坐标轴含义、实例、方法、配置、结论。
    # EN: **one caption per figure** (`AGENTS.md` §7.2): axes, instance, method, config, takeaway.
    by_arm: dict[str, dict[str, float]] = {}
    for r in records:
        by_arm.setdefault(r.arm, {})
    for png in sorted(out.glob("*.png")):
        stem = png.stem
        # CN: 文件名是 `pair000-066_<arm>_<view>.png`；用正则取出三段，不再靠字符串切片猜。
        # EN: filenames are `pair000-066_<arm>_<view>.png`; parse with a regex rather than guessing slices.
        match = re.match(r"^pair(?P<a>\d+)-(?P<b>\d+)_(?P<arm>.+)_(?P<view>[a-z]+)$", stem)
        if match is None:
            continue
        pair = f"{match.group('a')}-{match.group('b')}"
        arm = match.group("arm")
        view = match.group("view")
        metrics = next((r.metrics for r in records if r.arm == arm
                        and r.geometry == f"pair{pair}"), {})
        caption = (
            f"# {png.name}\n\n"
            f"**Axes.** No axes: an orthographic projection of the FAUST target surface "
            f"(`{view}` view). Colour is the per-vertex geodesic error between the predicted and the true "
            f"correspondence, normalised by `sqrt(surface area)` of the target.\n\n"
            f"**Instance.** MPI-FAUST training registrations, pair `{pair}`; 6890 vertices; "
            f"the source is registered to the target, so `pi_hat = I/n` is the ground truth.\n\n"
            f"**Method.** `{arm}`; cost fitted on pair `0-13` (subjects 0 and 1), evaluated on this pair "
            f"(subjects {int(pair.split('-')[0]) // 10} and {int(pair.split('-')[1]) // 10}). "
            f"Descriptors `K = 13` (extrinsic xyz, log area, log curvature, 8 HKS); `Theta` = simplex.\n\n"
            f"**Config.** Driver `studies/real/faust/heatmaps.py`; colour scale fixed at 0..the 98th percentile "
            f"of all arms on this pair, so panels are comparable; geodesics from `potpourri3d` heat method.\n\n"
            f"**Takeaway.** mean error {metrics.get('mean', float('nan')):.4f}, miss@5 % "
            f"{metrics.get('miss_5pct', float('nan')):.3f}, left/right flip rate "
            f"{metrics.get('flip_rate', float('nan')):.3f}. "
            f"{provenance.get('flip_definition', '')}\n"
        )
        png.with_name(f"{png.name}.caption.md").write_text(caption, encoding="utf-8")
    print(f"\nwrote {len(list(out.glob('*.png.caption.md')))} captions next to the PNGs")


def main(argv: list[str] | None = None) -> int:
    """CN: 入口。EN: entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", choices=["correspondence", "heatmaps"], required=True)
    parser.add_argument("--out", default=None, help="study directory name under out/ (default: the study)")
    args = parser.parse_args(argv)
    name = args.out or ("faust_correspondence" if args.study == "correspondence" else "faust_heatmaps")
    out = _REPO_ROOT / "iot_experiments" / "out" / name
    records, provenance = _load(out)
    print(f"{name}: {len(records)} rows\n")
    if args.study == "correspondence":
        _correspondence(records)
    else:
        _heatmaps(records, out, provenance)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
