"""
CN: 研究 1 的分析：把规模结果整理成表，并给出「总求解 < 2s」的经验前沿。
EN: Study 1 analysis: tabulate the scale results and report the empirical "total solve < 2s"
    frontier.

用法 / Usage:
    python iot_experiments/scaling/analyse.py iot_experiments/scaling/out/quadratic_gmm.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

TWO_SECONDS = 2.0


def _load(path: Path) -> list[dict]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return list(payload["records"])


def _frontier(records: list[dict], budget: float) -> list[dict]:
    """CN: 对每个 K，找 n 上 <budget> 秒的边界。EN: For each K, find the n-boundary under budget."""
    by_k: dict[int, list[dict]] = {}
    for record in records:
        by_k.setdefault(int(record["n_features"]), []).append(record)
    rows: list[dict] = []
    for k, group in sorted(by_k.items()):
        group = sorted(group, key=lambda r: int(r["spec"]["n"]))
        under = [r for r in group if float(r["solve_seconds"]) < budget]
        over = [r for r in group if float(r["solve_seconds"]) >= budget]
        largest_under = max((int(r["spec"]["n"]) for r in under), default=None)
        smallest_over = min((int(r["spec"]["n"]) for r in over), default=None)
        mnk = None
        if largest_under is not None:
            ref = next(r for r in under if int(r["spec"]["n"]) == largest_under)
            mnk = int(ref["spec"]["m"]) * int(ref["spec"]["n"]) * k
        rows.append(
            {
                "K": k,
                "largest_n_under_budget": largest_under,
                "smallest_n_over_budget": smallest_over,
                "m_n_K_at_boundary": mnk,
            }
        )
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="analyse the scale study")
    parser.add_argument("path", type=Path)
    parser.add_argument("--budget", type=float, default=TWO_SECONDS)
    args = parser.parse_args(argv)
    records = _load(args.path)

    print(f"# {args.path}  ({len(records)} records)\n")
    print(f"{'K':>5} {'m':>6} {'n':>6} {'it':>4} {'master_s/it':>12} {'oracle_s/it':>12} {'solve_s':>9} {'m*n*K':>11}")
    for record in sorted(
        records, key=lambda r: (int(r["n_features"]), int(r["spec"]["n"]))
    ):
        spec = record["spec"]
        print(
            f"{int(record['n_features']):>5} {int(spec['m']):>6} {int(spec['n']):>6} "
            f"{int(record['iterations']):>4} {float(record['mean_master_seconds']):>12.3f} "
            f"{float(record['mean_oracle_seconds']):>12.3f} {float(record['solve_seconds']):>9.3f} "
            f"{int(spec['m']) * int(spec['n']) * int(record['n_features']):>11.3e}"
        )

    print(f"\n## frontier: largest n with total solve < {args.budget:g}s\n")
    print(f"{'K':>5} {'largest_n_under':>16} {'smallest_n_over':>16} {'m*n*K@boundary':>16}")
    for row in _frontier(records, float(args.budget)):
        print(
            f"{row['K']:>5} {str(row['largest_n_under_budget']):>16} "
            f"{str(row['smallest_n_over_budget']):>16} {str(row['m_n_K_at_boundary']):>16}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
