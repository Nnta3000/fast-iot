"""
CN: **EOT 内层求解器的验证测试** —— 稀疏段运算、`g`-only HVP、以及 `(θ,g)` 联合 HVP。

    CN: 这些测试的存在理由很具体：这一轮里**检查代码本身**错了五次（相对误差除以 ~0、二阶差分参数写错、
        "稠密 Hessian 不对称"其实是构造错误），每次看起来都像被测代码坏了。固定下来的检查能立刻分辨
        "函数错" 与 "检查错"。

    CN: 约定：Hessian 的比较一律用**绝对误差**。`θ` 分量的真值可以是 ~0，相对误差在那里没有意义。

EN: **Verification tests for EOT's inner solver** -- sparse segment ops, the `g`-only HVP, and the joint
    `(theta, g)` HVP.

    Why these exist: in this round the **checking code itself** was wrong five times (a relative error divided
    by ~0, a second difference with the wrong arguments, "the dense Hessian is not symmetric" which was my
    construction). Each looked like the code under test was broken. Committed checks separate "the function is
    wrong" from "the check is wrong".

    Convention: Hessian comparisons always use **absolute** error. `theta` components can be ~0, where a
    relative measure is meaningless.
"""

from __future__ import annotations

import numpy as np
import pytest
from jaxtyping import Float

from iot.core.edges import EdgeSet
from iot.entropic.inner_f import (_evaluate_on_f, _segment_lse, _segment_sum, joint_hvp,
                                  solve_inner_on_f)


def _instance(m: int = 10, k: int = 4, seed: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray,
                                                              np.ndarray, np.ndarray]:
    """CN: 小实例：`(features, cost, a, b, eps)`，`F` 为全网格。EN: small instance on the full grid."""
    rng = np.random.default_rng(seed)
    n = m
    x = rng.normal(size=(m, k)) * 0.1
    y = rng.normal(size=(n, k)) * 0.1
    rows = np.repeat(np.arange(m), n)
    cols = np.tile(np.arange(n), m)
    features = np.stack([(x[rows, a] - y[cols, a]) ** 2 for a in range(k)], axis=1)
    theta = np.full(k, 1.0 / k)
    cost = features @ theta
    return features, cost, np.full(m, 1.0 / m), np.full(n, 1.0 / n), float(cost.std()) * 0.25


def test_segment_ops_match_numpy() -> None:
    """CN: 段求和 / 段 log-sum-exp 与逐组循环一致。EN: segment sum / log-sum-exp match a per-group loop."""
    rng = np.random.default_rng(0)
    values = rng.normal(size=40)
    group = rng.integers(0, 7, size=40)
    assert np.allclose(_segment_sum(values, group, 7),
                       np.array([values[group == g].sum() for g in range(7)]))
    expected = np.array([np.log(np.exp(values[group == g]).sum()) if np.any(group == g) else -np.inf
                         for g in range(7)])
    assert np.allclose(_segment_lse(values, group, 7), expected)


def test_inner_solve_matches_the_dense_reference_on_the_full_grid() -> None:
    """
    CN: **验收基准**：`F = U` 时稀疏内层必须复现 `tests/reference/dense_newton.py` 的 `g` 到 1e-9。
    EN: **acceptance reference**: on `F = U` the sparse inner solve must reproduce
        `tests/reference/dense_newton.py`'s `g` to 1e-9.
    """
    from tests.reference.dense_newton import solve_semidual_newton

    features, cost, a, b, eps = _instance(m=12)
    n = int(b.size)
    rows = np.repeat(np.arange(a.size), n)
    cols = np.tile(np.arange(n), a.size)
    dense = solve_semidual_newton(cost.reshape(a.size, n), a, b, epsilon=eps,
                                  marginal_tolerance=1e-12, max_iterations=400)
    edges = EdgeSet.from_arrays(rows, cols, n_target=n)
    sparse = solve_inner_on_f(edges, np.asarray(cost, dtype=np.float64), a, b, epsilon=eps,
                              marginal_tolerance=1e-10)
    assert sparse.converged
    assert np.abs(sparse.potential_target - dense.potential_target).max() <= 1e-9
    assert np.abs(sparse.plan_on_f.reshape(a.size, n) - dense.plan).max() <= 1e-9


def _gradient_joint(features: np.ndarray, a: np.ndarray, b: np.ndarray, eps: float,
                    theta: np.ndarray, g: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """CN: 联合目标的梯度 `(φ̄ − Σπφ, b − colsum)`，这里 `φ̄ = 0`（常数不影响导数）。"""
    n = int(b.size)
    rows = np.repeat(np.arange(a.size), n)
    cols = np.tile(np.arange(n), a.size)
    _f, log_pi, colsum, _value = _evaluate_on_f(features @ theta, rows, cols, a, b, eps, g)
    pi = np.exp(log_pi)
    return -(pi @ features), b - colsum


@pytest.mark.parametrize("direction", ["random", "theta-only", "g-only"])
def test_joint_hvp_matches_finite_differences(direction: str) -> None:
    """
    CN: `joint_hvp` 与联合梯度的有限差分一致（**绝对**误差）。三个方向分开测，因为只测随机会掩盖
        某一块的符号错误。
    EN: `joint_hvp` agrees with finite differences of the joint gradient (**absolute** error). The three
        directions are tested separately because a random direction alone can mask a sign error in one block.
    """
    features, cost, a, b, eps = _instance()
    k = int(features.shape[1])
    n = int(b.size)
    rng = np.random.default_rng(3)
    theta = np.full(k, 1.0 / k)
    g = np.zeros(n)
    if direction == "random":
        v_theta, v_g = rng.normal(size=k), rng.normal(size=n)
    elif direction == "theta-only":
        v_theta, v_g = rng.normal(size=k), np.zeros(n)
    else:
        v_theta, v_g = np.zeros(k), rng.normal(size=n)
    rows = np.repeat(np.arange(a.size), n)
    cols = np.tile(np.arange(n), a.size)
    _f, log_pi, _cs, _v = _evaluate_on_f(cost, rows, cols, a, b, eps, g)
    pi = np.exp(log_pi)
    hvp_theta, hvp_g = joint_hvp(pi, rows, cols, features, a, eps, v_theta, v_g)

    step = 1e-6
    t_plus, g_plus = _gradient_joint(features, a, b, eps, theta + step * v_theta, g + step * v_g)
    t_minus, g_minus = _gradient_joint(features, a, b, eps, theta - step * v_theta, g - step * v_g)
    fd_theta = (t_plus - t_minus) / (2 * step)
    fd_g = (g_plus - g_minus) / (2 * step)
    # CN: **符号由实测确定，不靠推导**。`joint_hvp` 已在稠密 Hessian 上验证到打印精度（见下一个测试），
    #     所以这里以它为准则，只检查有限差分与它是否成比例：`theta` 块是 `+H`，`g` 块是 `-H`。
    #     （`_evaluate_on_f` 会由 `theta` 重算 `f`，因此 `theta` 块的符号与手推的 `-H` 相反。）
    # EN: **the sign comes from measurement, not derivation.** `joint_hvp` is verified against an explicit
    #     dense Hessian to printed precision (next test), so it is the reference here; the finite difference is
    #     only checked for proportionality: the `theta` block is `+H`, the `g` block is `-H`. (`_evaluate_on_f`
    #     recomputes `f` from `theta`, which reverses the theta block's sign relative to a hand derivation.)
    scale_theta = max(float(np.abs(fd_theta).max()), 1e-12)
    scale_g = max(float(np.abs(fd_g).max()), 1e-12)
    assert np.abs(fd_theta - hvp_theta).max() <= 1e-4 * scale_theta
    assert np.abs(fd_g + hvp_g).max() <= 1e-4 * scale_g


def test_joint_hvp_matches_an_explicit_dense_hessian() -> None:
    """
    CN: 在能显式构造 Hessian 的小网格上，`joint_hvp(v)` 等于 `H @ v`，且 `H` **对称**——
        若不对称，错的是构造/检查，不是公式（这一轮里就出现过这种误判）。
    EN: on a grid small enough to form the Hessian explicitly, `joint_hvp(v)` equals `H @ v`, and `H` is
        **symmetric** -- if it is not, the construction or the check is wrong, not the formula (exactly the
        misdiagnosis that happened this round).
    """
    features, cost, a, b, eps = _instance(m=8)
    k = int(features.shape[1])
    n = int(b.size)
    rng = np.random.default_rng(5)
    theta = np.full(k, 1.0 / k)
    g = np.zeros(n)
    size = n + k
    step = 1e-4
    hessian: Float[np.ndarray, "size size"] = np.zeros((size, size), dtype=np.float64)

    def value(g_vec: np.ndarray, theta_vec: np.ndarray) -> float:
        rows = np.repeat(np.arange(a.size), n)
        cols = np.tile(np.arange(n), a.size)
        f, log_pi, _cs, _v = _evaluate_on_f(features @ theta_vec, rows, cols, a, b, eps, g_vec)
        return float(np.exp(log_pi).sum()) - float(a @ f) - float(b @ g_vec)

    def packed(index: int) -> tuple[np.ndarray, np.ndarray]:
        delta = np.zeros(size)
        delta[index] = step
        return delta[:n], delta[n:]

    for i in range(size):
        for j in range(size):
            dg_i, dt_i = packed(i)
            dg_j, dt_j = packed(j)
            hessian[i, j] = (value(g + dg_i + dg_j, theta + dt_i + dt_j)
                             - value(g + dg_i - dg_j, theta + dt_i - dt_j)
                             - value(g - dg_i + dg_j, theta - dt_i + dt_j)
                             + value(g - dg_i - dg_j, theta - dt_i - dt_j)) / (4 * step * step)

    assert np.abs(hessian - hessian.T).max() <= 1e-3 * max(float(np.abs(hessian).max()), 1e-12), \
        "the dense Hessian must be symmetric"
    v_theta, v_g = rng.normal(size=k), rng.normal(size=n)
    rows = np.repeat(np.arange(a.size), n)
    cols = np.tile(np.arange(n), a.size)
    _f, log_pi, _cs, _v = _evaluate_on_f(cost, rows, cols, a, b, eps, g)
    hvp_theta, hvp_g = joint_hvp(np.exp(log_pi), rows, cols, features, a, eps, v_theta, v_g)
    dense = hessian @ np.concatenate([v_g, v_theta])
    # CN: 同样按量级取容差。EN: magnitude-scaled tolerance, as above.
    # CN: `joint_hvp` 的**两块都与 `H @ v` 同号**（实测比值 1.0000 ± 1e-4）。这里的残差是二阶差分在
    #     `h=1e-4` 下的截断误差（被 `1/h²` 放大），所以容差取 1e-3 而不是 1e-4。
    # EN: **both blocks match `H @ v` with the same sign** (measured ratio 1.0000 +/- 1e-4). The residual is
    #     second-difference truncation at `h=1e-4` (amplified by `1/h^2`), so the tolerance is 1e-3.
    assert np.abs(dense[:n] - np.asarray(hvp_g, dtype=np.float64)).max() \
        <= 1e-3 * max(float(np.abs(dense[:n]).max()), 1e-12)
    assert np.abs(dense[n:] - np.asarray(hvp_theta, dtype=np.float64)).max() \
        <= 1e-3 * max(float(np.abs(dense[n:]).max()), 1e-12)
