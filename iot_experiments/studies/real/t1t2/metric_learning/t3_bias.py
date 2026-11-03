"""
EN: **T3 bias study -- is `min_theta F(theta)` actually an estimator of `theta_star`?**

    Motivation (measured, 2026-11-03). On the full grid with `F(theta) = inf_{f,g} Psi(theta, f, g)`:

      - `F(lam*theta, lam*eps) = lam*F(theta, eps)` holds EXACTLY (0.0 residual, two values of lam), so the
        scaling direction is a genuine plateau of `Psi` -- the note that `Pi_e = exp(sigma_e / eps)` depends on
        `(theta/eps, f/eps, g/eps)` only is correct;
      - but at **fixed** `eps`, `theta_star` is NOT a stationary point: `||grad_theta F(theta_star)|| = 26.9`
        on the `n=40` instance at `rel_eps = 0.25`.

    Those two facts are consistent: the plateau is a property of the *joint* scaling of `(theta, eps)`, and it
    does not make `theta_star` optimal at fixed `eps`. So minimising `Psi` targets some `theta_hat(eps)`, and
    the question this study answers is how far `theta_hat(eps)` sits from `theta_star`, and whether the gap
    shrinks with `eps`.

    This MATTERS BEFORE ANY JOINT SOLVER IS BUILT. If `theta_hat(eps)` is far from `theta_star` at the eps
    values P2/P3 will use, then a Newton-CG on `min Psi_F` converges to the wrong estimator, and every
    theta-error column downstream measures the wrong quantity. If the gap is `O(eps)`, `Psi_F` is a legitimate
    biased estimator and the joint build proceeds as planned.

    Method: minimise `F(theta) = inf_{f,g} Psi(theta, f, g)` over the simplex by **SLSQP with the analytic
    gradient** `phi_bar - sum_e pi_e phi_e` (the same gradient `soft_master` uses). The inner `(f, g)` solve is
    `sinkhorn_sweeps_on_f` to a tight marginal tolerance. No Nelder-Mead, no softmax reparameterisation: both
    were in a throwaway probe that ran 25 minutes and produced no output.

    Reported per instance and `eps`: `theta_star`, `theta_hat`, `L1(theta_hat, theta_star)`, `F(theta_hat)`,
    `F(theta_star)`, `||grad_theta F(theta_star)||`, SLSQP iterations, inner sweeps, wall time.

CN: **T3 偏差研究 —— `min_theta F(theta)` 到底是不是 `theta_star` 的估计？**

    已测：缩放不变性 `F(λθ,λε) = λF(θ,ε)` **精确成立**（残差 0.0）；但**固定 ε** 时 `θ*` 不是驻点
    （`n=40`、`rel_eps=0.25` 时 `||∇_θF(θ*)|| = 26.9`）。两者并不矛盾：平台是 `(θ,ε)` 联合缩放的
    性质，不保证固定 ε 下 `θ*` 最优。

    本实验回答：`θ̂(ε) = argmin F` 离 `θ*` 多远，差距是否随 ε 缩小。方法：单纯形上 **SLSQP + 解析梯度**
    `φ̄ − Σπφ`，内层用 Sinkhorn 到紧容差。**必须在建联合求解器之前做**：若 `θ̂` 远离 `θ*`，则牛顿-CG
    收敛到的是错误的估计量，下游所有 θ 误差列都在量错东西。
"""

from __future__ import annotations

import dataclasses
import json
import sys
import time
from pathlib import Path

import numpy as np
from jaxtyping import Float

_REPO_ROOT = Path(__file__).resolve().parents[5]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from iot.core.edges import EdgeSet  # noqa: E402
from iot.core.parameters import cost_std  # noqa: E402
from iot.entropic.inner_f import sinkhorn_sweeps_on_f  # noqa: E402
from iot_experiments.harness.config import RunConfig, build_parser  # noqa: E402
from iot_experiments.harness.lib.planting import plant_instance  # noqa: E402
from iot_experiments.harness.preflight import check as preflight_check  # noqa: E402
from iot_experiments.harness.records import RunLog  # noqa: E402

SIZES: tuple[int, ...] = (40, 120)
REL_EPS_GRID: tuple[float, ...] = (0.5, 0.25, 0.1)
POINT_DIMENSION: int = 4

PROVENANCE: dict[str, object] = {
    "question": "does argmin_theta inf_{f,g} Psi(theta,f,g) recover theta_star, and does the gap shrink with eps?",
    "instance": "planting.plant_instance family=diagonal (Monge, exactly optimal pi_hat under theta_star)",
    "edge_set": "full grid F = U",
    "optimiser": "scipy SLSQP on the simplex with the analytic gradient phi_bar - sum_e pi_e phi_e",
    "inner": "sinkhorn_sweeps_on_f, marginal_tolerance 1e-12",
    "eps": "absolute and frozen: rel_eps * std(c(theta_ref = uniform))",
    "gate": "if L1(theta_hat, theta_star) does not shrink toward 0 with eps, min Psi_F is the wrong target",
}


@dataclasses.dataclass(frozen=True)
class BiasConfig(RunConfig):
    """EN: knobs. CN: 旋钮。"""

    section = "t3_bias"
    study = "t3_bias"
    point_dimension: int = POINT_DIMENSION
    inner_tolerance: float = 1e-12

    @classmethod
    def add_study_arguments(cls, parser) -> None:
        """EN: only this study's flags. CN: 只加本脚本特有的参数。"""
        parser.add_argument("--point-dimension", type=int, default=POINT_DIMENSION)
        parser.add_argument("--inner-tolerance", type=float, default=1e-12)


def _evaluate(problem, edges, epsilon: float, theta: Float[np.ndarray, "k"],
              ) -> tuple[float, Float[np.ndarray, "k"], float, int]:
    """
    EN: `F(theta) = inf_{f,g} Psi(theta,.)` and its theta-gradient; returns `(value, grad, error, sweeps)`.
    CN: 内层最优后的目标值与 θ 梯度，返回 `(值, 梯度, 边缘误差, 扫描数)`。
    """
    features = np.asarray(problem.features.features(edges.rows, edges.cols), dtype=np.float64)
    cost = np.asarray(features @ np.asarray(theta, dtype=np.float64), dtype=np.float64)
    source: Float[np.ndarray, "m"] = np.asarray(problem.resolved_source_mass, dtype=np.float64)
    target: Float[np.ndarray, "n"] = np.asarray(problem.resolved_target_mass, dtype=np.float64)
    inner = sinkhorn_sweeps_on_f(edges, cost, source, target, epsilon=float(epsilon),
                                 marginal_tolerance=1e-12, max_sweeps=400000)
    plan = np.asarray(inner.plan_on_f, dtype=np.float64)
    # EN: phi_bar is the observed feature average; the gradient is phi_bar minus the model's expectation.
    # CN: φ̄ 是观测特征均值，梯度为 φ̄ 减去模型期望。
    support_rows, support_cols, support_mass = problem.support()
    support_features = np.asarray(problem.features.features(support_rows, support_cols), dtype=np.float64)
    phi_bar: Float[np.ndarray, "k"] = (np.asarray(support_mass, dtype=np.float64)[:, None]
                                       * support_features).sum(axis=0)
    value = (float(np.asarray(theta) @ phi_bar)
             - float(source @ np.asarray(inner.potential_source, dtype=np.float64))
             - float(target @ np.asarray(inner.potential_target, dtype=np.float64))
             + float(epsilon) * float(plan.sum()))
    grad = np.asarray(phi_bar - features.T @ plan, dtype=np.float64)
    return value, grad, float(inner.marginal_error), int(inner.newton_steps)


def main(argv: list[str] | None = None) -> int:
    """EN: entry point. CN: 入口。"""
    parser = build_parser(BiasConfig)
    args = parser.parse_args(list(argv) if argv is not None else None)
    config = BiasConfig.from_namespace(args, parser)
    run_dir = config.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    log = RunLog(run_dir / "run.log")
    log.line("T3 bias study: is argmin_theta inf_{f,g} Psi an estimator of theta_star?")
    log.line(f"sizes={SIZES} rel_eps={REL_EPS_GRID} dimension={config.point_dimension}")

    preflight = preflight_check(run_tiny=False, need_gpu=False)
    preflight.write(run_dir / "preflight.json")
    log.line(f"preflight ok={preflight.ok} mem_available={preflight.mem_available_gb:.1f} GB "
             f"git={preflight.git_sha}")
    if not preflight.ok:
        log.line(f"preflight failures: {preflight.failures}")

    from scipy.optimize import minimize  # noqa: PLC0415

    rows: list[dict[str, object]] = []
    for size in SIZES:
        planted = plant_instance(family="diagonal", n=int(size),
                                 point_dimension=int(config.point_dimension), seed=0)
        problem = planted.problem
        m_eff = int(problem.features.n_source)
        n_eff = int(problem.features.n_target)
        k = int(planted.theta_star.size)
        theta_star: Float[np.ndarray, "k"] = np.asarray(planted.theta_star, dtype=np.float64)
        all_rows = np.repeat(np.arange(m_eff), n_eff)
        all_cols = np.tile(np.arange(n_eff), m_eff)
        edges = EdgeSet.from_arrays(all_rows, all_cols, n_target=n_eff)
        scale = cost_std(np.asarray(problem.features.costs(np.full(k, 1.0 / k), all_rows, all_cols),
                                    dtype=np.float64))
        constraints = problem.theta_set.linear_constraints()
        a_eq = np.asarray(constraints.a_eq, dtype=np.float64)
        b_eq = np.asarray(constraints.b_eq, dtype=np.float64)
        lower = np.asarray(constraints.lower, dtype=np.float64)
        bounds = [(None if not np.isfinite(lo) else float(lo), None) for lo in lower]
        log.line(f"=== n={size}: m_eff={m_eff} n_eff={n_eff} grid={m_eff * n_eff} K={k} "
                 f"theta_star={np.round(theta_star, 4).tolist()} cost_std={scale:.4e} ===")

        for rel_eps in REL_EPS_GRID:
            eps = float(rel_eps) * scale
            # EN: log BEFORE the work so a stall reads as "started and never returned".
            # CN: 先记开始，卡住时日志表现为"开始了但没回来"。
            log.line(f"  rel_eps={rel_eps}: start (eps={eps:.4e})")
            started = time.perf_counter()

            def objective(theta: np.ndarray) -> tuple[float, np.ndarray]:
                value, grad, _err, _sw = _evaluate(problem, edges, eps, theta)
                return value, grad

            f_star, grad_star, err_star, _sw_star = _evaluate(problem, edges, eps, theta_star)
            result = minimize(objective, theta_star.copy(), jac=True, method="SLSQP",
                              bounds=bounds, constraints=[{"type": "eq", "fun": lambda t: a_eq @ t - b_eq,
                                                           "jac": lambda t: a_eq}],
                              options={"maxiter": 200, "ftol": 1e-12})
            theta_hat = np.clip(np.asarray(result.x, dtype=np.float64), 0.0, None)
            theta_hat = theta_hat / theta_hat.sum()
            f_hat, _g_hat, err_hat, sweeps_hat = _evaluate(problem, edges, eps, theta_hat)
            elapsed = time.perf_counter() - started

            record = {
                "n": int(size), "m_eff": m_eff, "n_eff": n_eff, "K": k,
                "rel_eps": float(rel_eps), "eps_absolute": eps,
                "theta_star": [round(float(v), 8) for v in theta_star],
                "theta_hat": [round(float(v), 8) for v in theta_hat],
                "l1_theta_hat_to_star": round(float(np.abs(theta_hat - theta_star).sum()), 8),
                "f_theta_star": float(f_star), "f_theta_hat": float(f_hat),
                "f_improvement": float(f_star - f_hat),
                "grad_norm_at_theta_star": round(float(np.linalg.norm(grad_star)), 8),
                "slsqp_status": int(result.status), "slsqp_iterations": int(result.nit),
                "inner_marginal_error": err_hat, "inner_sweeps_at_hat": sweeps_hat,
                "seconds": round(elapsed, 2),
            }
            rows.append(record)
            log.line(f"  rel_eps={rel_eps}: L1(theta_hat,theta*)={record['l1_theta_hat_to_star']:.4f} "
                     f"F*={f_hat:.6e} (F(theta*)={f_star:.6e}, gain={record['f_improvement']:.3e}) "
                     f"|grad F(theta*)|={record['grad_norm_at_theta_star']:.2f} "
                     f"slsqp_iter={result.nit} err={err_hat:.1e} {elapsed:.1f}s")
            log.line(f"      theta_hat = {np.round(theta_hat, 4).tolist()}")

    payload = {"rows": rows, "provenance": PROVENANCE,
               "preflight": preflight.to_dict(),
               "config": {key: (list(v) if isinstance(v, tuple) else str(v) if isinstance(v, Path) else v)
                          for key, v in dataclasses.asdict(config).items()}}
    (run_dir / "bias.json").write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")
    log.line(f"wrote {run_dir / 'bias.json'}")
    log.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
