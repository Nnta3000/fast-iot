"""
CN: 研究 2 —— 不同边际。固定 family 与规模，扫 marginal（uniform / gaussian / gmm /
    参数化尖锐度 gaussian:<center>:<width>）。
EN: Study 2 -- different marginals. Fix family and scale, sweep the marginal (uniform /
    gaussian / gmm / parameterised peakedness `gaussian:<center>:<width>`).

用法 / Usage:
    python iot_experiments/marginals/run.py --families diagonal quadratic lowrank \
        --marginals uniform gaussian gmm --n 800
    python iot_experiments/marginals/run.py --families quadratic \
        --marginals gaussian:0.5:0.30 gaussian:0.5:0.10 gaussian:0.5:0.03 --n 800
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
    """CN: 展开边际轴为 RunSpec 列表。EN: Expand the marginal axis into RunSpecs."""
    specs: list[RunSpec] = []
    for family in args.families:
        for marginal in args.marginals:
            for seed in args.seeds:
                specs.append(
                    RunSpec(
                        family=str(family), marginal=str(marginal),
                        m=int(args.n), n=int(args.n), point_dimension=int(args.point_dimension),
                        seed=int(seed), max_iterations=int(args.max_iterations),
                        validate=bool(args.validate),
                    )
                )
    return specs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="marginal study")
    parser.add_argument("--families", nargs="+", default=None)
    parser.add_argument("--marginals", nargs="+", default=["uniform", "gaussian", "gmm"])
    parser.add_argument("--n", type=int, default=800)
    parser.add_argument("--point-dimension", type=int, default=4)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--max-iterations", type=int, default=300)
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)
    if args.families is None:
        args.families = ["diagonal", "quadratic", "lowrank"]

    out = args.out or (_OUT_DIR / f"marginals_n{int(args.n)}.json")
    specs = build_specs(args)
    meta = {"study": "marginals", "n": int(args.n), "axis": "marginal"}
    run_grid(specs, out=out, meta=meta)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
