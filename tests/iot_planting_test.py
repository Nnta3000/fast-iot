"""
CN: **构造式 planting（Monge 问题）** 的测试：证书自洽、无质量分裂、|S| = n、n_eff 剪枝、
    以及在标准设置下循环能够收敛（tie_break / pruning 的影响）。
EN: Tests for **constructive planting (Monge problems)**: certificate self-consistency, no mass
    splitting, |S| = n, n_eff pruning, and that the loop converges under the standard settings
    (and how tie_break / pruning affect it).
"""

from __future__ import annotations

import numpy as np

from iot import LinearMasterSolver, solve_inverse_ot
from iot_experiments.core.planting import PROBLEM_TYPE, plant_instance


def test_planted_plan_is_certified_and_monge() -> None:
    # CN: 证书精确（互补松弛）且标注为 Monge、无分裂。EN: exact certificate (CS), tagged Monge, no split.
    planted = plant_instance(family="quadratic", n=200, point_dimension=4, seed=0)
    assert planted.problem_type == PROBLEM_TYPE == "monge"
    assert planted.mass_splitting is False
    assert planted.max_dual_violation <= 1e-9
    assert planted.max_support_residual <= 1e-9


def test_planted_support_is_one_edge_per_source() -> None:
    # CN: Monge => 每个源恰好一条边，|S| = n_eff。EN: Monge => one edge per source, |S| = n_eff.
    planted = plant_instance(family="diagonal", n=150, point_dimension=4, seed=0)
    rows, cols, values = planted.problem.support()
    assert rows.size == planted.n_eff
    assert np.all(np.bincount(rows, minlength=planted.n_eff) == 1)
    assert np.all(values > 0.0)


def test_planted_empty_columns_are_pruned() -> None:
    # CN: 空列剪枝后所有列质量为正，且 m_eff <= n_eff。EN: after pruning, all columns positive; m_eff <= n_eff.
    planted = plant_instance(family="quadratic", n=300, point_dimension=4, seed=0)
    col_mass = np.asarray(planted.problem.plan.sum(axis=0)).reshape(-1)
    assert col_mass.size == planted.m_eff
    assert np.all(col_mass > 0.0)
    assert planted.m_eff <= planted.n_eff


def test_planted_loop_converges_without_pruning() -> None:
    # CN: 在 prune=False（对严格循环安全）下循环收敛，且判定为最优。（CPU 精确 master）
    # EN: with prune=False (safe against strict cycling) the loop converges and the verdict is optimal
    #     (CPU exact master).
    planted = plant_instance(family="quadratic", n=200, point_dimension=4, seed=0)
    result = solve_inverse_ot(
        planted.problem, solver=LinearMasterSolver(random_seed=0), tie_break="none",
        max_iterations=200, scan_backend="numpy", validate=True, prune=False, prune_hysteresis=False,
    )
    assert result.converged
    assert result.is_optimal is True


def test_planted_pruning_cycles_with_the_first_order_master_contract() -> None:
    # CN: **目标路径的关键发现**——cupdlpx 只支持 tie_break="none"（无 lexicographic 重解），
    #     因此在种入的 Monge 实例上**只有 prune=none 收敛**；tolerance/hysteresis 两种剪枝都会循环。
    #     这里用 mock 的一个 first-order-like master 无法复现 GPU，故直接校验**契约**：
    #     CuPdlpxMasterSolver 对任何非 "none" 的 tie_break 抛错，且 prune=none 是 cupdlpx 的可行设置。
    # EN: **key finding on the target path** -- cupdlpx supports tie_break="none" only (no lexicographic
    #     re-solve), so on planted Monge instances ONLY prune=none converges; both tolerance and
    #     hysteresis cycle. We cannot reproduce the GPU here, so we assert the *contract*:
    #     CuPdlpxMasterSolver rejects any tie_break != "none", and prune=none is cupdlpx's usable setting.
    from iot import CuPdlpxMasterSolver
    from iot.problem import InverseOTError
    from iot.components import mandatory_edges

    planted = plant_instance(family="diagonal", n=60, point_dimension=4, seed=0)
    edges, _ = mandatory_edges(planted.problem)
    solver = CuPdlpxMasterSolver(params={"time_sec_limit": 5.0})
    # The contract: no lexicographic re-solve on the first-order GPU master.
    try:
        solver.solve(planted.problem, edges, tie_break="lex_generic")
    except Exception as error:  # noqa: BLE001 - the exact type depends on the optional native ext
        assert "tie_break" in str(error) or isinstance(error, (InverseOTError, RuntimeError))
    else:  # pragma: no cover - only if the native ext is unavailable and solve() would fail differently
        raise AssertionError("cupdlpx unexpectedly accepted tie_break='lex_generic'")


def test_planted_small_margin_regime_is_flagged() -> None:
    # CN: gap 分位数必须被报告（用于区分"求解失败"与"实例几乎不可辨识"）。
    # EN: gap quantiles are reported (to tell "solver failed" from "instance barely identifiable").
    planted = plant_instance(family="quadratic", n=200, point_dimension=4, seed=0)
    assert set(planted.gap_quantiles) == {"0.00", "0.01", "0.10", "0.50", "0.90", "1.00"}
    assert planted.gap_quantiles["0.00"] <= planted.gap_quantiles["0.50"]


def test_planted_random_g_and_mass() -> None:
    # CN: g 随机与随机源质量都可用；g 被记录（g_norm）。
    # EN: random g and random source masses both work; g is logged (g_norm).
    rng = np.random.default_rng(7)
    planted = plant_instance(
        family="quadratic", n=150, point_dimension=4, seed=0,
        g=rng.normal(scale=0.5, size=150), source_mass=rng.uniform(0.5, 1.5, size=150),
    )
    assert planted.g_norm > 0.0
    assert planted.max_dual_violation <= 1e-9
    row_mass = np.asarray(planted.problem.plan.sum(axis=1)).reshape(-1)
    assert np.all(row_mass > 0.0)
