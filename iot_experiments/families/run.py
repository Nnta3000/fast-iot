"""
CN: 研究 3 —— 代价族（diagonal / quadratic / lowrank，外加 tiny-n bilinear-PSD）。
    固定 marginal 与规模，扫 family。
EN: Study 3 -- cost families (diagonal / quadratic / lowrank, plus tiny-n bilinear-PSD).
    Fix the marginal and scale, sweep the family.

CN: `bilinear-signed` 有意不在列（0∈Θ 使可行性平凡）。
EN: `bilinear-signed` is deliberately excluded (0 in Theta makes feasibility trivial).

用法 / Usage:
    python iot_experiments/families/run.py --families diagonal quadratic lowrank \
        --marginal gmm --sizes 200 800 1600 --point-dimension 4
    python iot_experiments/families/run.py --families bilinear-psd --marginal gmm --psd-ns 8 16 32
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


def build_specs(args: argparse.Namespace) -> list[RunSpec]:
    """CN: 展开代价族轴为 RunSpec 列表。EN: Expand the cost-family axis into RunSpecs."""
    specs: list[RunSpec] = []
    for family in args.families:
        sizes = [int(value) for value in args.psd_ns] if family == "bilinear-psd" else [int(value) for value in args.sizes]
        for size in sizes:
            for seed in args.seeds:
                specs.append(
                    RunSpec(
                        family=str(family), marginal=str(args.marginal),
                        m=int(size), n=int(size), point_dimension=int(args.point_dimension),
                        seed=int(seed), max_iterations=int(args.max_iterations),
                        validate=bool(args.validate),
                    )
                )
    return specs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="cost-family study")
    parser.add_argument("--families", nargs="+", default=["diagonal", "quadratic", "lowrank"])
    parser.add_argument("--marginal", default="gmm")
    parser.add_argument("--sizes", type=int, nargs="+", default=[200, 800, 1600])
    parser.add_argument("--psd-ns", type=int, nargs="+", default=[8, 16, 32])
    parser.add_argument("--point-dimension", type=int, default=4)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--max-iterations", type=int, default=300)
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    out = args.out or (_OUT_DIR / f"families_{args.marginal}.json")
    specs = build_specs(args)
    meta = {"study": "families", "marginal": str(args.marginal), "axis": "family"}
    run_grid(specs, out=out, meta=meta)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
