"""
CN: 研究 1 —— 规模增长（K, m, n）。固定 family/marginal，扫 m=n 与点维度（K）或矩形 m×n。
EN: Study 1 -- scale growth (K, m, n). Fix family/marginal and sweep square m=n vs point
    dimension (K), or a rectangular m x n grid.

CN: 产出 `out/<family>_<marginal>.json` + `.csv`，每条记录都带完整 RunSpec（AGENTS §7.1）。
EN: Writes `out/<family>_<marginal>.json` + `.csv`; every record carries its full RunSpec.

用法 / Usage:
    python iot_experiments/scaling/run.py --family quadratic --marginal gmm \
        --sizes 200 400 800 1600 3200 --dims 2 4 8 16
    python iot_experiments/scaling/run.py --family quadratic --marginal gmm \
        --rect 200:3200 3200:200 800:1600 --dims 8
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from iot_experiments.core import RunSpec, run_grid  # noqa: E402

_OUT_DIR = Path(__file__).resolve().parent / "out"


def _parse_rect(values: list[str]) -> list[tuple[int, int]]:
    """CN: 解析 "m:n" 对。EN: Parse "m:n" pairs."""
    pairs: list[tuple[int, int]] = []
    for value in values:
        left, _, right = value.partition(":")
        pairs.append((int(left), int(right)))
    return pairs


def build_specs(args: argparse.Namespace) -> list[RunSpec]:
    """CN: 展开规模轴为一个 RunSpec 列表。EN: Expand the scale axis into a list of RunSpecs."""
    specs: list[RunSpec] = []
    if args.rect:
        for m, n in _parse_rect([str(value) for value in args.rect]):
            for d in args.dims:
                for seed in args.seeds:
                    specs.append(
                        RunSpec(
                            family=str(args.family), marginal=str(args.marginal),
                            m=int(m), n=int(n), point_dimension=int(d), seed=int(seed),
                            max_iterations=int(args.max_iterations), validate=bool(args.validate),
                            tag="rect",
                        )
                    )
    else:
        for d in args.dims:
            for size in args.sizes:
                for seed in args.seeds:
                    specs.append(
                        RunSpec(
                            family=str(args.family), marginal=str(args.marginal),
                            m=int(size), n=int(size), point_dimension=int(d), seed=int(seed),
                            max_iterations=int(args.max_iterations), validate=bool(args.validate),
                            tag="square",
                        )
                    )
    return specs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="scale growth (K, m, n)")
    parser.add_argument("--family", default="quadratic")
    parser.add_argument("--marginal", default="gmm")
    parser.add_argument("--sizes", type=int, nargs="+", default=[200, 400, 800, 1600, 3200])
    parser.add_argument("--dims", type=int, nargs="+", default=[2, 4, 8, 16])
    parser.add_argument("--rect", nargs="*", default=None, help='explicit "m:n" pairs (overrides --sizes)')
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--max-iterations", type=int, default=300)
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    out = args.out or (_OUT_DIR / f"{args.family}_{args.marginal}.json")
    specs = build_specs(args)
    meta = {
        "study": "scaling",
        "family": str(args.family),
        "marginal": str(args.marginal),
        "axis": "K/m/n",
    }
    run_grid(specs, out=out, meta=meta)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
