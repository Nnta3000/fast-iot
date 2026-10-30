"""
CN: **合并前的数值基线** —— 在一个小实例上同时跑精确循环与松弛循环，把结果写成 JSON，
    作为每次合并后的比对标准。**精确版是基准**：它更可信，松弛版只跟着它走。

    CN: 固定输入（一个合成实例，seed 固定），记录：θ、δ* 或 [L,U]、|F|、轮数、是否收敛、
        以及最终 `is_optimal_plan` / 最大违反量。任何合并只要让**精确版**的数字变化，
        就是合并有错，应当回退。

EN: **Numeric baseline before merging** -- run both the exact loop and the relaxed loop on one small
    instance and write the results to JSON as the comparison point for every merge. **The exact
    version is the baseline**: it is the one trusted; the relaxed one follows it.

    Fixed synthetic instance (fixed seed). Records theta, delta* or [L, U], |F|, rounds, convergence,
    and the final `is_optimal_plan` / max violation. If a merge changes the **exact** numbers, the
    merge is wrong and must be reverted.

    Usage:  python -m iot_experiments.tools.loop_baseline --out <path>
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sparse
from jaxtyping import Float, Int

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from iot import DiagonalWeightFeatures, InverseOTProblem  # noqa: E402
from iot.core.theta import SimplexTheta  # noqa: E402
from iot.eval.validation import is_optimal_plan  # noqa: E402
from iot.exact.loop import solve_inverse_ot  # noqa: E402
from iot.relaxed.loop_regret import solve_inverse_ot_regret  # noqa: E402
from iot.relaxed.master_regret import RegretMasterSolver  # noqa: E402


@dataclasses.dataclass(frozen=True)
class Instance:
    """CN: 一个固定的合成实例。EN: one fixed synthetic instance."""

    name: str
    m: int
    n: int
    d: int
    seed: int

    def build(self) -> InverseOTProblem:
        """
        CN: 建一个**可精确识别**的实例：代价由已知 θ* 生成，π̂ 是 θ* 下的最优方案。
            这样精确循环有解（δ* ≥ 0），两个算法可以直接比较。
        EN: build an instance that is **exactly identifiable**: costs come from a known theta* and pi_hat
            is the optimal plan under it, so the exact loop is feasible (delta* >= 0) and the two
            algorithms are directly comparable.
        """
        import ot  # noqa: PLC0415

        rng = np.random.default_rng(int(self.seed))
        x: Float[np.ndarray, "m d"] = np.ascontiguousarray(rng.normal(size=(self.m, self.d)))
        y: Float[np.ndarray, "n d"] = np.ascontiguousarray(rng.normal(size=(self.n, self.d)))
        # CN: θ* 用固定的、非均匀的权重，保证方案不是退化对称的。
        # EN: theta* uses fixed, non-uniform weights so the plan is not degenerately symmetric.
        theta_star: Float[np.ndarray, "d"] = np.abs(rng.normal(size=self.d))
        theta_star = theta_star / theta_star.sum()
        features = DiagonalWeightFeatures(x, y)
        rows: Int[np.ndarray, "m n"] = np.repeat(np.arange(self.m), self.n)
        cols: Int[np.ndarray, "m n"] = np.tile(np.arange(self.n), self.m)
        cost: Float[np.ndarray, "m n"] = np.asarray(
            features.costs(theta_star, rows, cols), dtype=np.float64).reshape(self.m, self.n)
        a: Float[np.ndarray, "m"] = np.full(self.m, 1.0 / self.m)
        b: Float[np.ndarray, "n"] = np.full(self.n, 1.0 / self.n)
        plan: Float[np.ndarray, "m n"] = np.asarray(
            ot.emd(a, b, cost, numItermax=10_000_000), dtype=np.float64)
        problem = InverseOTProblem(features=features, theta_set=SimplexTheta(dimension=self.d), plan=sparse.csr_matrix(plan))
        _ = a, b
        return problem


def _summarise(name: str, payload: dict[str, object]) -> dict[str, object]:
    """CN: 给结果加名字与键排序，便于逐字段比较。EN: tag a result with its name for field-wise compare."""
    return {"name": name, **payload}


def run_exact(instance: Instance) -> dict[str, object]:
    """CN: 跑精确循环并汇总。EN: run the exact loop and summarise."""
    problem = instance.build()
    result = solve_inverse_ot(problem, tol=1e-9, gamma=1, max_iterations=50, prune=False)
    theta: Float[np.ndarray, "k"] = np.asarray(result.theta, dtype=np.float64)
    return _summarise("exact", {
        "delta": float(result.delta),
        "converged": bool(result.converged),
        "rounds": int(len(result.iterations)),
        "edges": int(result.iterations[-1].support_size) if result.iterations else 0,
        "max_violation": float(result.final_max_violation),
        "theta": [round(float(v), 12) for v in theta],
        "theta_l1": round(float(np.abs(theta).sum()), 12),
        "is_optimal_plan": bool(is_optimal_plan(problem, theta).is_optimal),
    })


def run_relaxed(instance: Instance, *, prune: bool = False, tau_prune: float = 1e-3,
                prune_hysteresis: bool = True) -> dict[str, object]:
    """
    CN: 跑松弛循环并汇总。`prune=True` 用来覆盖**剪枝分支**——`prune=False` 时那段代码根本不执行，
        基线会漏掉它。
    EN: run the relaxed loop and summarise. `prune=True` covers the **pruning branch**; with
        `prune=False` that code never executes and the baseline would miss it.
    """
    problem = instance.build()
    result = solve_inverse_ot_regret(
        problem, tol=None, gamma=1, max_iterations=50,
        master=RegretMasterSolver(backend="highs", tie_break=False,
                                  params={"time_sec_limit": 120.0}),
        eps_r_rel=1e-6, prune=bool(prune), prune_hysteresis=bool(prune_hysteresis),
        tau_prune=float(tau_prune), check_pot=False, selection="none")
    theta: Float[np.ndarray, "k"] = np.asarray(result.theta, dtype=np.float64)
    tag = "relaxed"
    if prune:
        tag = "relaxed_prune_hyst" if prune_hysteresis else "relaxed_prune_nohyst"
    return _summarise(tag, {
        "prune": bool(prune), "tau_prune": float(tau_prune),
        "prune_hysteresis": bool(prune_hysteresis),
        "lower": float(result.lower),
        "upper": float(result.upper),
        "gap": float(result.upper - result.lower),
        "converged": bool(result.converged),
        "stall_status": str(result.stall_status),
        "rounds": int(result.n_iterations),
        "edges": int(result.finite_edges),
        "theta": [round(float(v), 12) for v in theta],
        "theta_l1": round(float(np.abs(theta).sum()), 12),
        "is_optimal_plan": bool(is_optimal_plan(problem, theta).is_optimal),
    })


def main(argv: list[str] | None = None) -> int:
    """CN: 入口：跑两个算法，写 JSON。EN: run both algorithms, write JSON."""
    parser = argparse.ArgumentParser(prog="python -m iot_experiments.tools.loop_baseline")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--compare-to", type=Path, default=None,
                        help="compare against an earlier baseline and report field differences")
    args = parser.parse_args(list(argv) if argv is not None else None)

    instance = Instance(name="identifiable_m12_n12_d3", m=12, n=12, d=3, seed=0)
    results = [run_exact(instance), run_relaxed(instance),
               run_relaxed(instance, prune=True, tau_prune=1e-3),
               run_relaxed(instance, prune=True, tau_prune=1e-3, prune_hysteresis=False)]
    payload = {"instance": dataclasses.asdict(instance), "results": results}

    if args.out is not None:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {args.out}")

    if args.compare_to is not None:
        old = json.loads(Path(args.compare_to).read_text(encoding="utf-8"))
        diffs: list[str] = []
        for before, after in zip(old["results"], results):
            for key in before:
                if key == "theta":
                    if before[key] != after[key]:
                        worst = max(abs(float(p) - float(q)) for p, q in zip(before[key], after[key]))
                        diffs.append(f"{before['name']}.theta differs, max|delta|={worst:.3e}")
                    continue
                if before[key] != after[key]:
                    diffs.append(f"{before['name']}.{key}: {before[key]!r} -> {after[key]!r}")
        if diffs:
            print("DIFFERENCES vs baseline:")
            for line in diffs:
                print("  " + line)
            return 1
        print("IDENTICAL to baseline (every field of both algorithms)")
        return 0

    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
