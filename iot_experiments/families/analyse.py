"""
CN: 研究 3 的分析：把代价族结果整理成表。
EN: Study 3 analysis: tabulate the cost-family results.

用法 / Usage:
    python iot_experiments/families/analyse.py iot_experiments/families/out/families_gmm.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="analyse the cost-family study")
    parser.add_argument("path", type=Path)
    args = parser.parse_args(argv)
    records = list(json.loads(Path(args.path).read_text(encoding="utf-8"))["records"])

    print(f"# {args.path}  ({len(records)} records)\n")
    print(
        f"{'family':>14} {'marginal':>10} {'n':>6} {'K':>5} {'|S|':>7} {'p':>5} {'it':>4} "
        f"{'conv':>5} {'opt':>5} {'cert':>5} {'delta':>10} {'solve_s':>8}"
    )
    for record in sorted(
        records, key=lambda r: (str(r["spec"]["family"]), int(r["spec"]["n"]))
    ):
        spec = record["spec"]
        print(
            f"{str(spec['family']):>14} {str(spec['marginal']):>10} {int(spec['n']):>6} "
            f"{int(record['n_features']):>5} {int(record['support_size']):>7} "
            f"{int(record['n_components']):>5} {int(record['iterations']):>4} "
            f"{str(record['converged']):>5} {str(record['is_optimal']):>5} {str(record['certified']):>5} "
            f"{float(record['delta']):>10.3e} {float(record['solve_seconds']):>8.3f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
