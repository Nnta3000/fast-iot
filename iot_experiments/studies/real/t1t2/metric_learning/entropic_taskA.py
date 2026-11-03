"""
CN: **Task A —— 熵 OT 的泛化测试**（只测量，不实现受限 Sinkhorn 循环）。

    CN: 问题：在受限边集 `F` 与 LP 有竞争力的那些 `eps` 上，熵 OT 学到的 θ 是否比 cosine 更能找回正确配对？
        这与 `entropic_f2.py` 的区别是它**固定 eps**（不做 K 折选择），因为门槛检查的是指定的 `eps`。

    CN: 设置与 F2 其它臂**完全一致**：同一个池（400 对留出）、G2 几何、`matching_hit`、同一套标注子集选择。
        m=100 的标注子集用 F2 的 `default_rng(draw).choice(remaining, size=m)` 原样生成，因此与
        cosine / exact-iOT 的行配对。

    CN: 内层用 **Newton 半对偶**（交替 Sinkhorn 在 `eps=0.05` 停在 ~5e-6，达不到协议要求的 1e-6）。

    CN: **预先注册的判据**（在看到 D 之前写下，之后不改）：
        `D(eps) = hit(entropic, eps) - hit(cosine)`，单位为百分点，m=100。
        - 若 rel eps 0.1 **或** 0.05 上 `D > 0` ⇒ **GATE PASSES**；
        - 若两处都 `D <= 0` ⇒ **GATE FAILS**；
        - 若某个主 `eps` 无法达到（内层不收敛）⇒ 记 `UNREACHED`，**不**替换为别的 eps。

EN: **Task A -- the entropic generalisation test** (measurement only; no restricted-Sinkhorn loop).

    Question: at the epsilons where the restricted edge set `F` is competitive with the LP, does the theta
    learned by entropic OT find the right partner better than cosine? This differs from `entropic_f2.py` in
    that it **fixes epsilon** (no K-fold selection), because the gate is about specified epsilons.

    The setup is **identical** to F2's other arms: same pool (400 held out), G2 geometry, `matching_hit`, and
    the same labelled-subset selection. The m=100 subset is built exactly as F2 does it, so the rows pair with
    cosine and exact iOT.

    The inner solve is the **Newton semi-dual** (alternating Sinkhorn stalls at ~5e-6 at eps=0.05, which misses
    the protocol's 1e-6).

    **Pre-registered rule** (written before seeing D, not changed afterwards):
        `D(eps) = hit(entropic, eps) - hit(cosine)` in points, m=100.
        - `D > 0` at rel eps 0.1 **or** 0.05 => **GATE PASSES**;
        - `D <= 0` at both => **GATE FAILS**;
        - an unreachable primary epsilon is recorded as `UNREACHED` and **never** substituted.
"""

from __future__ import annotations

import dataclasses
import json
import resource
import sys
from pathlib import Path

import numpy as np
from jaxtyping import Float

_REPO_ROOT = Path(__file__).resolve().parents[5]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from iot import InverseOTProblem  # noqa: E402
from iot.core.features import DiagonalWeightFeatures  # noqa: E402
from iot.core.theta import SimplexTheta  # noqa: E402
from iot.entropic.theta import fit_entropic  # noqa: E402
from iot_experiments.harness.config import RunConfig, build_parser  # noqa: E402
from iot_experiments.harness.records import RunLog  # noqa: E402
from iot_experiments.harness.registry import apply_geometry, features, fit_geometry  # noqa: E402
from iot_experiments.harness.solver import diagonal_cost, matching_hit  # noqa: E402

PRIMARY_EPS: tuple[float, ...] = (0.1, 0.05)
"""CN: 门槛依据的主 eps 格。EN: the primary epsilons the gate rests on."""

REFERENCE_EPS: tuple[float, ...] = (0.25, 0.5)
"""CN: 仅作参考的 eps（不触发门槛）。EN: reference epsilons only; they never trigger the gate."""

HOLDOUT_PAIRS: int = 400

PROVENANCE: dict[str, object] = {
    "task": "A — entropic generalisation vs cosine at the LP-competitive epsilons",
    "setup": "F2 verbatim: dinov2-pool4 PCA, 400 held out, geometry G2 from the pool, matching_hit",
    "inner_solver": "Newton semi-dual (alternating Sinkhorn stalls at ~5e-6 at eps=0.05)",
    "no_certificate": "entropic produces no delta and no bracket; this measures held-out hit rate only",
    "eps": "ABSOLUTE and frozen within each fit: eps = rel_eps * std(cost at uniform theta), computed once",
}


def _rss_gb() -> float:
    """CN: 峰值 RSS（GB）。EN: peak RSS in GB."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6


def _f2_reference(k: int, m: int, draw: int) -> dict[str, float | None]:
    """
    CN: 从 F2 的已提交记录里读 cosine 与 exact-iOT 在同一 `(k, m, draw)` 的 hit。
        读不到就返回 `None`，由调用方决定是否重算。
    EN: read cosine and exact iOT hits for the same `(k, m, draw)` from F2's committed records. Missing values
        come back as `None`; the caller decides whether to recompute.
    """
    path = _REPO_ROOT / "iot_experiments" / "out" / "records" / "metric_learning" / "records.json"
    if not path.is_file():
        return {"cosine": None, "exact_iot": None}
    payload = json.loads(path.read_text(encoding="utf-8"))
    cosine: float | None = None
    exact: float | None = None
    for row in payload.get("records", []):
        if bool(row.get("excluded")) or int(row.get("k", -1)) != int(k) or int(row.get("m", -1)) != int(m):
            continue
        if int(row.get("draw", -1)) != int(draw):
            continue
        if row.get("arm") == "cosine":
            cosine = float(row["metrics"]["hit"])
        elif row.get("arm") == "iot":
            exact = float(row["metrics"]["hit"])
    return {"cosine": cosine, "exact_iot": exact}


def _fit_once(x: Float[np.ndarray, "m k"], y: Float[np.ndarray, "m k"], *, relative_epsilon: float,
              max_steps: int, learning_rate: float, inner_tolerance: float
              ) -> tuple[Float[np.ndarray, "k"] | None, dict[str, object]]:
    """
    CN: 在标注对上拟合一次熵 OT（Newton 内层）。内层不收敛即 `FAILED`，**绝不**用未收敛的方案。
    EN: one entropic fit on the labelled pairs (Newton inner solve). A non-converged inner plan is `FAILED`;
        a non-converged plan is **never** used.
    """
    import scipy.sparse as sparse  # noqa: PLC0415

    m = int(x.shape[0])
    k = int(x.shape[1])
    plan = sparse.csr_matrix((np.full(m, 1.0 / m), (np.arange(m), np.arange(m))), shape=(m, m))
    problem = InverseOTProblem(features=DiagonalWeightFeatures(x, y),
                               theta_set=SimplexTheta(dimension=k), plan=plan)
    try:
        result = fit_entropic(problem, relative_epsilon=float(relative_epsilon),
                              learning_rate=float(learning_rate), max_steps=int(max_steps),
                              gradient_tolerance=1e-8, inner_solver="newton",
                              inner_tolerance=float(inner_tolerance))
    except Exception as error:  # noqa: BLE001 - reported, never substituted
        return None, {"status": f"FAILED: {type(error).__name__}: {str(error)[:120]}"}
    theta = np.asarray(result.theta, dtype=np.float64)
    if not np.all(np.isfinite(theta)):
        return None, {"status": "FAILED: non-finite theta"}
    last = result.steps_record[-1] if result.steps_record else None
    # CN: 是否停在 Θ 的某个面上：坐标接近 0 的比例。
    # EN: whether theta sits on a face of Theta: the fraction of near-zero coordinates.
    at_bound = float(np.mean(theta <= 1e-12))
    return theta, {"status": "ok", "steps": int(result.steps), "loss": float(result.loss),
                   "converged": bool(result.converged), "stall": str(result.stall_reason),
                   "final_gradient": float(last.gradient_norm) if last is not None else float("nan"),
                   "inner_marginal_error": float(last.marginal_error) if last is not None else float("nan"),
                   "theta_at_bound_fraction": round(at_bound, 4),
                   "theta": [round(float(v), 8) for v in theta]}


@dataclasses.dataclass(frozen=True)
class TaskAConfig(RunConfig):
    """CN: 旋钮。EN: knobs."""

    section = "t1t2_entropic_taskA"
    study: str = "t1t2_entropic_taskA"
    k: int = 64
    m: int = 100
    draw: int = 0
    extra_label_draws: tuple[int, ...] = ()
    max_steps: int = 300
    learning_rate: float = 0.5
    inner_tolerance: float = 1e-9

    @classmethod
    def add_study_arguments(cls, parser) -> None:
        """CN: 只加本脚本特有的参数。EN: only this script's flags."""
        parser.add_argument("--k", type=int, default=64)
        parser.add_argument("--m", type=int, default=100)
        parser.add_argument("--extra-label-draws", type=int, nargs="*", default=[],
                            help="additional labelled-subset seeds; each reported separately")


def main(argv: list[str] | None = None) -> int:
    """CN: 入口。EN: entry point."""
    parser = build_parser(TaskAConfig)
    args = parser.parse_args(list(argv) if argv is not None else None)
    config = TaskAConfig.from_namespace(args, parser)
    run_dir = config.run_dir / f"k{int(config.k)}_m{int(config.m)}"
    run_dir.mkdir(parents=True, exist_ok=True)
    log = RunLog(run_dir / "run.log")
    log.line("TASK A: entropic generalisation vs cosine. Measurement only; no loop.")
    log.line(f"config: k={config.k} m={config.m} primary={PRIMARY_EPS} "
             f"reference={REFERENCE_EPS} extra_draws={config.extra_label_draws}")

    k = int(config.k)
    x, y = features("dinov2-pool4", dim=k)
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    pool_rng = np.random.default_rng(10_000)
    order = pool_rng.permutation(int(x.shape[0]))
    pool_subjects = np.sort(order[:HOLDOUT_PAIRS])
    remaining = np.sort(order[HOLDOUT_PAIRS:])
    stats = fit_geometry(x[pool_subjects], y[pool_subjects], "G2")
    x_pool, y_pool = apply_geometry(x[pool_subjects], y[pool_subjects], stats)
    for arr in (x_pool, y_pool):
        norms = np.linalg.norm(np.asarray(arr, dtype=np.float64), axis=1)
        if not np.allclose(norms, 1.0, atol=1e-8):
            raise SystemExit(f"pool space not L2-normalised: [{norms.min():.6f}, {norms.max():.6f}]")
    cosine_hit = float(matching_hit(diagonal_cost(x_pool, y_pool, np.full(k, 1.0 / k)), seed=0))
    reference = _f2_reference(k, int(config.m), 0)
    log.line(f"pool {len(pool_subjects)} pairs; recomputed cosine hit={cosine_hit:.4f}; "
             f"F2 records cosine={reference['cosine']} iot={reference['exact_iot']}")
    if reference["cosine"] is not None and abs(float(reference["cosine"]) - cosine_hit) > 1e-9:
        raise SystemExit(
            f"recomputed cosine hit {cosine_hit} != F2 record {reference['cosine']}; "
            "the pools differ and comparison would be invalid")
    exact_hit = reference["exact_iot"]
    d_exact = None if exact_hit is None else 100.0 * (exact_hit - cosine_hit)

    rows: list[dict[str, object]] = []
    for label_seed in (int(config.draw), *[int(v) for v in config.extra_label_draws]):
        labelled = np.sort(np.random.default_rng(label_seed).choice(remaining, size=int(config.m),
                                                                   replace=False))
        x_fit, y_fit = apply_geometry(x[labelled], y[labelled], stats)
        log.line(f"--- labelled subset seed={label_seed} (m={int(config.m)}) ---")
        for rel_eps in (*PRIMARY_EPS, *REFERENCE_EPS):
            theta, diag = _fit_once(x_fit, y_fit, relative_epsilon=float(rel_eps),
                                    max_steps=int(config.max_steps),
                                    learning_rate=float(config.learning_rate),
                                    inner_tolerance=float(config.inner_tolerance))
            row: dict[str, object] = {
                "label_seed": label_seed, "m": int(config.m), "k": k,
                "rel_eps": float(rel_eps), "eps_absolute": float(rel_eps) * float(
                    np.std(np.asarray(DiagonalWeightFeatures(x_fit, y_fit).costs(
                        np.full(k, 1.0 / k), np.repeat(np.arange(int(config.m)), int(config.m)),
                        np.tile(np.arange(int(config.m)), int(config.m))), dtype=np.float64))),
                "cell_class": "primary" if rel_eps in PRIMARY_EPS else "reference",
                "cos_hit": round(cosine_hit, 4),
            }
            if theta is None:
                row.update({"status": diag["status"], "hit": None, "d_hit_pts": None})
                log.line(f"  rel_eps={rel_eps}: {diag['status']}")
                rows.append(row)
                continue
            hit = float(matching_hit(diagonal_cost(x_pool, y_pool, theta), seed=0))
            row.update({"hit": round(hit, 4), "d_hit_pts": round(100.0 * (hit - cosine_hit), 1),
                        **{key: value for key, value in diag.items() if key != "theta"}})
            if exact_hit is not None:
                row["d_vs_exact_iot_pts"] = round(100.0 * (hit - exact_hit), 1)
            rows.append(row)
            log.line(f"  rel_eps={rel_eps:<5} ({row['cell_class']}): hit={hit:.4f} "
                     f"d_hit={row['d_hit_pts']:+.1f} pts vs cosine"
                     + (f", {row['d_vs_exact_iot_pts']:+.1f} vs exact iOT" if exact_hit else "")
                     + f" | steps={diag['steps']} conv={diag['converged']} "
                       f"inner_err={diag['inner_marginal_error']:.1e} at_bound={diag['theta_at_bound_fraction']}")

    # CN: **预先注册的判据**，在 m=100 的主子集（seed=0）上判定。
    # EN: the **pre-registered rule**, decided on the primary subset (seed 0).
    primary = [r for r in rows if r["label_seed"] == 0 and r["cell_class"] == "primary"]
    reached = [r for r in primary if r.get("d_hit_pts") is not None]
    if not reached:
        verdict = "INCONCLUSIVE (no primary epsilon reached)"
    elif any(float(r["d_hit_pts"]) > 0.0 for r in reached):
        verdict = "GATE PASSES"
    elif len(reached) == len(primary) and all(float(r["d_hit_pts"]) <= 0.0 for r in reached):
        verdict = "GATE FAILS"
    else:
        verdict = "INCONCLUSIVE (an unreached primary epsilon may have decided it)"
    log.line(f"VERDICT: {verdict}")
    payload = {"rows": rows, "verdict": verdict, "provenance": PROVENANCE,
               "cosine_hit": cosine_hit, "exact_iot_hit_m100": exact_hit,
               "d_exact_iot_pts": d_exact,
               "gate_rule": "D > 0 at rel eps 0.1 or 0.05 on the primary subset => PASSES",
               "config": {key: (list(v) if isinstance(v, tuple) else str(v) if isinstance(v, Path) else v)
                          for key, v in dataclasses.asdict(config).items()}}
    (run_dir / "taskA.json").write_text(json.dumps(payload, indent=2, default=str) + "\n",
                                        encoding="utf-8")
    log.line(f"wrote {run_dir / 'taskA.json'}; peak {_rss_gb():.2f} GB")
    log.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
