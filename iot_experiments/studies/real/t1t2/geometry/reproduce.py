"""
CN: **复现检查（重构前必须通过）** —— 用新 harness 重跑两个已知结果，与冻结驱动的旧值对比：

    | 结果 | 旧值（`legacy/core/geometry_sweep.py`） |
    |---|---|
    | G2 cosine, K=256, m=100 | hit ≈ 0.8304, R@1 ≈ 0.7681（3 draws 均值） |
    | G2 learned, K=64, m=300 | hit ≈ 0.5910（3 draws 均值） |

    CN: 通过条件：新值与旧值的差在**噪声**内 —— hit 与 R@1 的绝对差 ≤ 0.01（旧值的 draw 间 std 为
        0.003–0.018，故 0.01 约为 1 个标准差的上界），且方向一致。若不通过，**不进入后续研究**，
        先报告差异来源。

EN: **Reproduction gate** -- rerun two known results through the new harness and compare with the frozen
    driver's values. Pass condition: `|new - old| <= 0.01` for hit and R@1 (the old across-draw std was
    0.003-0.018, so 0.01 is about a one-sigma upper bound) with the same sign. If it fails, report the
    discrepancy and do **not** proceed to further studies.

用法 / EN: usage
    PYTHONPATH=/home/agent/Documents/dpm-inverse-ot python -u -m iot_experiments.studies.real.t1t2.geometry.reproduce
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from jaxtyping import Float, Int

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from iot_experiments.harness.data import dinov2_pool4_pca  # noqa: E402
from iot_experiments.harness.registry import geometry  # noqa: E402
from iot_experiments.harness.solver import (  # noqa: E402
    diagonal_cost,
    fit_full_lp,
    matching_hit,
    rank_metrics,
    uniform_cosine,
)

# CN: 旧值来自 `legacy/core/geometry_sweep.py` 的 3-draw 均值（见 REFACTOR_NOTES.md）。
# EN: reference values are the frozen driver's 3-draw means.
REFERENCE: dict[str, dict[str, float]] = {
    "G2-cosine-K256-m100": {"hit": 0.8304, "R@1": 0.7681},
    "G2-learned-K64-m300": {"hit": 0.5910, "R@1": 0.5043},
}
TOLERANCE: float = 0.01


@dataclass(frozen=True)
class ReproRow:
    """CN: 一行复现结果（新值、旧值、差值）。EN: one reproduction row (new, reference, delta)."""

    label: str
    metric: str
    new: float
    reference: float
    delta: float
    passed: bool


def run_cell(pca: tuple[Float[np.ndarray, "subjects K"], Float[np.ndarray, "subjects K"]],
             *, dim: int, m: int, draws: tuple[int, ...], solver: str,
             solve_limit_s: float) -> dict[str, float]:
    """
    CN: 在一个 (K, m) 单元上跑 3 draws，返回 **cosine 与 learned 的均值**（G2 几何）。
    EN: Run 3 draws for one (K, m) cell and return the **cosine and learned means** under G2.
    """
    x, y = pca
    n = int(x.shape[0])
    cos_hits: list[float] = []
    cos_r1: list[float] = []
    lrn_hits: list[float] = []
    lrn_r1: list[float] = []
    for draw in draws:
        order: Int[np.ndarray, "subjects"] = np.random.default_rng(int(draw)).permutation(n)
        labelled: Int[np.ndarray, "m"] = order[: int(m)]
        unlabelled: Int[np.ndarray, "n"] = order[int(m):]
        geo = geometry(x, y, "G2", labelled=labelled, unlabelled=unlabelled)
        xu, yu = geo.x_unlabelled, geo.y_unlabelled
        # CN: cosine 与 uniform θ 的排序一致（registry 已断言），这里显式用 cosine。
        # EN: cosine and uniform theta rank identically (asserted in registry); use cosine explicitly.
        c_cos = uniform_cosine(xu, yu)
        cos_hits.append(matching_hit(c_cos, seed=int(draw)))
        cos_r1.append(rank_metrics(c_cos)["R@1"])

        fit = fit_full_lp(geo.x_label, geo.y_label, solver=solver, solve_limit_s=solve_limit_s)
        if fit.status != "ok":
            raise SystemExit(f"learned fit FAILED at K={dim} m={m} draw={draw}: {fit.status}")
        c_lrn = diagonal_cost(xu, yu, fit.theta)
        lrn_hits.append(matching_hit(c_lrn, seed=int(draw)))
        lrn_r1.append(rank_metrics(c_lrn)["R@1"])
        print(f"  K={dim} m={m} d={draw}: cos hit={cos_hits[-1]:.4f} R@1={cos_r1[-1]:.4f} | "
              f"learned hit={lrn_hits[-1]:.4f} R@1={lrn_r1[-1]:.4f} "
              f"R'/R_uni={fit.R_ratio:.2e} {fit.solve_s:.1f}s", flush=True)
    return {"cos_hit": float(np.mean(cos_hits)), "cos_R@1": float(np.mean(cos_r1)),
            "lrn_hit": float(np.mean(lrn_hits)), "lrn_R@1": float(np.mean(lrn_r1))}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="reproduction gate for the refactored harness")
    parser.add_argument("--draws", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--solver", default="cupdlpx", choices=("cupdlpx", "highs"))
    parser.add_argument("--solve-limit-s", type=float, default=300.0)
    parser.add_argument("--out", type=Path,
                        default=_REPO_ROOT / "iot_experiments" / "gpu_bench" / "out" / "reproduce")
    args = parser.parse_args(argv)

    t_all = time.perf_counter()
    print("# reproduction gate: new harness vs frozen legacy driver", flush=True)
    print("# note: standardisation and PCA use the POOL images (transductive, label-free)", flush=True)

    rows: list[ReproRow] = []

    print("\n[1] G2 cosine, K=256, m=100", flush=True)
    pca256 = dinov2_pool4_pca(256)
    cell_a = run_cell(pca256, dim=256, m=100, draws=tuple(args.draws), solver=str(args.solver),
                      solve_limit_s=float(args.solve_limit_s))
    ref_a = REFERENCE["G2-cosine-K256-m100"]
    for metric, key in (("hit", "cos_hit"), ("R@1", "cos_R@1")):
        new = float(cell_a[key])
        old = ref_a[metric]
        rows.append(ReproRow("G2 cosine K256 m100", metric, new, old, new - old,
                             abs(new - old) <= TOLERANCE))

    print("\n[2] G2 learned, K=64, m=300", flush=True)
    pca64 = dinov2_pool4_pca(64)
    cell_b = run_cell(pca64, dim=64, m=300, draws=tuple(args.draws), solver=str(args.solver),
                      solve_limit_s=float(args.solve_limit_s))
    ref_b = REFERENCE["G2-learned-K64-m300"]
    for metric, key in (("hit", "lrn_hit"), ("R@1", "lrn_R@1")):
        new = float(cell_b[key])
        old = ref_b[metric]
        rows.append(ReproRow("G2 learned K64 m300", metric, new, old, new - old,
                             abs(new - old) <= TOLERANCE))

    print(f"\n{'cell':>24} {'metric':>6} {'new':>8} {'old':>8} {'delta':>8} {'pass':>6}", flush=True)
    for row in rows:
        print(f"{row.label:>24} {row.metric:>6} {row.new:>8.4f} {row.reference:>8.4f} "
              f"{row.delta:>+8.4f} {str(row.passed):>6}", flush=True)

    all_pass = all(r.passed for r in rows)
    print(f"\ngate: {'PASS' if all_pass else 'FAIL'} (tolerance {TOLERANCE}) "
          f"in {time.perf_counter() - t_all:.0f}s", flush=True)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "reproduce.json").write_text(
        json.dumps({"rows": [r.__dict__ for r in rows], "passed": all_pass}, indent=2) + "\n",
        encoding="utf-8")
    print(f"[reproduce] wrote {args.out}", flush=True)
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
