"""
CN: 研究 2 的分析：把边际结果整理成表（支撑结构 p、迭代数、求解时间、认证）。
EN: Study 2 analysis: tabulate marginal results (support structure p, iterations, solve time,
    certificate).

用法 / Usage:
    python iot_experiments/marginals/analyse.py iot_experiments/marginals/out/marginals_n800.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="analyse the marginal study")
    parser.add_argument("path", type=Path)
    args = parser.parse_args(argv)
    records = list(json.loads(Path(args.path).read_text(encoding="utf-8"))["records"])

    print(f"# {args.path}  ({len(records)} records)\n")
    print(
        f"{'family':>10} {'marginal':>26} {'|S|':>7} {'p':>6} {'it':>4} {'conv':>5} "
        f"{'opt':>5} {'cert':>5} {'cert_resid':>11} {'solve_s':>8}"
    )
    for record in sorted(
        records, key=lambda r: (str(r["spec"]["family"]), str(r["spec"]["marginal"]))
    ):
        spec = record["spec"]
        print(
            f"{str(spec['family']):>10} {str(spec['marginal']):>26} {int(record['support_size']):>7} "
            f"{int(record['n_components']):>6} {int(record['iterations']):>4} "
            f"{str(record['converged']):>5} {str(record['is_optimal']):>5} {str(record['certified']):>5} "
            f"{float(record['cert_support_residual']):>11.2e} {float(record['solve_seconds']):>8.3f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
