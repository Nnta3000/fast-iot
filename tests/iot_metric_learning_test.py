"""
CN: 度量学习研究的断言 —— 先证明优化器能学，再相信真实数据上的任何数字。

    CN: 标准植入（固定，不做扫描）：K=64、m=300、**8 个特征**携带信号、噪声 σ=0.05。
        实测 oracle θ（θ=1/8 在 8 个信号特征上）hit = 0.983，cosine = 0.197，因此"学会了"可分辨。
        （σ=1.0 时连 oracle 都只有 0.007，那种植入无法区分好坏臂，已删除。）

    CN: 覆盖：
      (1) **步 0 = cosine**（B/C/D 都必须；盒投影对 uniform θ 是恒等）；
      (2) **植入恢复**：InfoNCE 三种盒都要接近 oracle，residual 要达到 oracle；
      (3) CCA 闭式，无步 0 检查，改用植入恢复；
      (4) 盒投影可行且精确。

    CN: 关于 InfoNCE 的**符号**：传给 InfoNCE 的必须是**相似度** `s = xᵀdiag(θ)y`（真配对上更大），
        而不是代价 `-s`。传 `-s` 会把目标反转——实测各臂因此把权重压在不一致的维度上（hit 0.003）。
        这条由下面的测试固定。

EN: Assertions for the metric-learning study -- prove the optimiser can learn before trusting any number on
    real data.

    **Standard planted case (fixed, no sweep):** K=64, m=300, **8 features** carry the signal, noise
    sigma=0.05. Measured: the oracle theta (1/8 on the 8 signal features) scores hit 0.983 and cosine 0.197,
    so "learned" is distinguishable. (At sigma=1.0 even the oracle managed only 0.007, so that plant could not
    tell a good arm from a bad one; it has been removed.)

    Covers: (1) **step 0 = cosine** for B/C/D, with the box projection the identity on uniform theta;
    (2) **planted recovery** for all three InfoNCE boxes and for the residual arm; (3) CCA is closed form so
    it gets a planted test instead; (4) the box projection is feasible and exact.

    On InfoNCE's **sign**: InfoNCE must receive the **similarity** `s = x^T diag(theta) y` (larger on true
    pairs), not the cost `-s`. Passing `-s` inverts the objective -- measured to put weight on dimensions that
    do *not* agree (hit 0.003). A test pins this.
"""

from __future__ import annotations

import numpy as np
from jaxtyping import Float

from iot_experiments.harness.solver import diagonal_cost, matching_hit
from iot_experiments.studies.metric_learning.run import (
    Box,
    fit_cca,
    fit_diagonal_infonce,
    fit_residual_bilinear,
    project,
    symmetric_infonce_torch,
)

K_STANDARD: int = 64
M_STANDARD: int = 300
SIGNAL_FEATURES: int = 8
NOISE: float = 0.05
ORACLE_FLOOR: float = 0.90
LEARNED_FLOOR: float = 0.75


def unit(a: Float[np.ndarray, "m k"]) -> Float[np.ndarray, "m k"]:
    """CN: 逐行 L2 归一化。EN: row-wise L2 normalisation."""
    norms = np.linalg.norm(a, axis=1, keepdims=True)
    return a / np.where(norms > 0, norms, 1.0)


def planted(*, seed: int = 0) -> tuple[Float[np.ndarray, "m k"], Float[np.ndarray, "m k"],
                                       Float[np.ndarray, "k"]]:
    """
    CN: 标准植入：返回 `(x, y, oracle_theta)`。8 个特征共享信号，其余独立噪声。
    EN: the standard plant: `(x, y, oracle_theta)` with 8 features sharing a signal and the rest independent
        noise.
    """
    rng = np.random.default_rng(seed)
    signal = np.zeros(K_STANDARD)
    signal[:SIGNAL_FEATURES] = 1.0
    # CN: **先归一化 x，再加信号**。顺序非常关键：若先乘信号再归一化，单位范数会被那 8 列主导，
    #     cosine 立刻就有 0.973，"学会了"便无从体现。先归一化后，信号是叠加在噪声上的**残差**，
    #     cosine 只有 0.197，学习才必须真正起作用（实测：before=0.973, after=0.197）。
    # EN: **normalise x first, then add the signal.** The order matters: multiplying by the signal before
    #     normalising lets those 8 columns dominate the unit norm, giving cosine 0.973 immediately, so
    #     "learned" would show nothing. Normalising first makes the signal a **residual** on top of noise,
    #     leaving cosine at 0.197 so learning has to do real work (measured: before 0.973, after 0.197).
    x: Float[np.ndarray, "m k"] = unit(rng.normal(size=(M_STANDARD, K_STANDARD)))
    y: Float[np.ndarray, "m k"] = unit(x * signal + rng.normal(scale=NOISE, size=(M_STANDARD, K_STANDARD)))
    oracle = np.zeros(K_STANDARD)
    oracle[:SIGNAL_FEATURES] = 1.0 / float(SIGNAL_FEATURES)
    return x, y, oracle


def test_planted_case_is_calibrated() -> None:
    """
    CN: 植入必须先可解：oracle 高、cosine 低，否则后面的"学会了"无法解释。
    EN: the plant must be solvable first: oracle high, cosine low, or "learned" cannot be interpreted.
    """
    x, y, oracle = planted()
    oracle_hit = matching_hit(diagonal_cost(x, y, oracle), seed=0)
    cosine_hit = matching_hit(diagonal_cost(x, y, np.ones(K_STANDARD) / K_STANDARD), seed=0)
    assert oracle_hit > ORACLE_FLOOR, f"oracle too low: {oracle_hit:.3f}"
    assert cosine_hit < 0.35, f"cosine too high: {cosine_hit:.3f}"


def test_step_zero_equals_cosine_for_all_arms() -> None:
    """
    CN: **步 0 = cosine**：InfoNCE 在 0 步时 θ = uniform（投影是恒等），residual 在 0 步时 V=0。
    EN: **step 0 = cosine**: InfoNCE at 0 steps gives uniform theta (projection is the identity) and the
        residual arm starts with V = 0.
    """
    x, y, _ = planted()
    uniform = np.ones(K_STANDARD) / K_STANDARD
    baseline = diagonal_cost(x, y, uniform)

    for box in (Box.iot(K_STANDARD), Box.wide(K_STANDARD), Box.free()):
        projected = project(uniform, box)
        assert np.allclose(projected, uniform, atol=1e-12), (box.name, projected.min(), projected.max())
        zero = fit_diagonal_infonce(x, y, box=box, lr=0.05, epochs=0, temperature=1.0)
        assert zero.status == "ok", zero.note
        assert zero.theta is not None
        assert np.allclose(zero.theta, uniform, atol=1e-12), (box.name, zero.theta[:3])
        assert np.allclose(zero.cost(x, y), baseline, atol=1e-12)

    for rank in (16, 64):
        zero_res = fit_residual_bilinear(x, y, rank=rank, lr=0.05, epochs=0, temperature=1.0)
        assert zero_res.status == "ok", zero_res.note
        # CN: residual 的代价是 `-score`；步 0 时 `V=0` ⇒ `score = xᵀy` ⇒ 代价与 cosine 单调等价。
        #     具体地 `-xᵀy` 与 cosine 的 `1 - xᵀy/... ` 在行归一化后只差常数，排序相同。
        # EN: the residual cost is `-score`; with V = 0 at step 0, `score = x^T y`, so the cost is a monotone
        #     equivalent of cosine on unit rows (it differs by a constant, so the ordering is identical).
        assert np.allclose(zero_res.cost(x, y), -x @ y.T, atol=1e-10), rank


def test_infonce_uses_similarity_not_cost() -> None:
    """
    CN: InfoNCE 必须收**相似度**：集中 θ 在信号特征上，损失必须**下降**（若收到 `-s` 则会上升）。
    EN: InfoNCE must receive the **similarity**: concentrating theta on the signal features must **lower** the
        loss (it would rise if `-s` were passed).
    """
    import torch

    x, y, oracle = planted()
    a = torch.tensor(x, dtype=torch.float64)
    b = torch.tensor(y, dtype=torch.float64)
    uniform = torch.tensor(np.ones(K_STANDARD) / K_STANDARD, dtype=torch.float64)
    signal = torch.tensor(oracle, dtype=torch.float64)

    def loss_at(theta: "torch.Tensor") -> float:
        similarity = (a * theta.unsqueeze(0)) @ b.t()
        return float(symmetric_infonce_torch(similarity, temperature=1.0))

    assert loss_at(signal) < loss_at(uniform), (
        f"concentrating on signal must lower InfoNCE: {float(loss_at(signal)):.4f} vs "
        f"{float(loss_at(uniform)):.4f}")


def test_planted_recovery_diagonal_arms() -> None:
    """
    CN: 三种盒的 InfoNCE 都必须学到信号（接近 oracle），并在信号特征上集中权重。
    EN: InfoNCE with all three boxes must learn the signal (approaching the oracle) and concentrate weight on
        the signal features.
    """
    x, y, oracle = planted()
    oracle_hit = matching_hit(diagonal_cost(x, y, oracle), seed=0)
    cosine_hit = matching_hit(diagonal_cost(x, y, np.ones(K_STANDARD) / K_STANDARD), seed=0)
    for box in (Box.iot(K_STANDARD), Box.wide(K_STANDARD), Box.free()):
        outcome = fit_diagonal_infonce(x, y, box=box, lr=0.05, epochs=100, temperature=1.0)
        assert outcome.status == "ok", (box.name, outcome.note)
        assert outcome.theta is not None
        theta = outcome.theta
        hit = matching_hit(diagonal_cost(x, y, theta), seed=0)
        assert hit > LEARNED_FLOOR, (box.name, hit, cosine_hit)
        assert hit > 3.0 * cosine_hit, (box.name, hit, cosine_hit)
        assert theta[:SIGNAL_FEATURES].mean() > 2.0 * theta[SIGNAL_FEATURES:].mean(), (
            box.name, theta[:SIGNAL_FEATURES].mean(), theta[SIGNAL_FEATURES:].mean())
        assert hit <= oracle_hit + 1e-9 or box.name != "iot", (box.name, hit, oracle_hit)


def test_planted_recovery_residual_arms() -> None:
    """CN: residual 臂必须达到 oracle（它没有盒限制）。EN: the residual arms must reach the oracle."""
    x, y, oracle = planted()
    oracle_hit = matching_hit(diagonal_cost(x, y, oracle), seed=0)
    for rank in (16, 64):
        outcome = fit_residual_bilinear(x, y, rank=rank, lr=0.05, epochs=100, temperature=1.0)
        assert outcome.status == "ok", outcome.note
        hit = matching_hit(outcome.cost(x, y), seed=0)
        assert hit >= oracle_hit - 0.05, (rank, hit, oracle_hit)


def test_cca_planted_recovery() -> None:
    """CN: CCA 闭式，用植入恢复代替步 0：线性相关的配对必须被恢复。"""
    rng = np.random.default_rng(4)
    k, m = 16, 200
    latent: Float[np.ndarray, "m 4"] = rng.normal(size=(m, 4))
    px: Float[np.ndarray, "4 k"] = rng.normal(size=(4, k))
    py: Float[np.ndarray, "4 k"] = rng.normal(size=(4, k))
    x: Float[np.ndarray, "m k"] = latent @ px + 0.05 * rng.normal(size=(m, k))
    y: Float[np.ndarray, "m k"] = latent @ py + 0.05 * rng.normal(size=(m, k))
    outcome = fit_cca(unit(x), unit(y), ridge=1e-3)
    assert outcome.status == "ok", outcome.note
    cost = outcome.cost(unit(x), unit(y))
    assert matching_hit(cost, seed=0) > 0.9, outcome.note


def test_box_projection_is_feasible_and_exact() -> None:
    """CN: 投影后 `sum=1` 且严格落在 `[lo, hi]`。EN: after projection, sum=1 and lo <= theta <= hi."""
    rng = np.random.default_rng(3)
    for k in (16, 64, 256):
        for box in (Box.iot(k), Box.wide(k), Box.free()):
            values: Float[np.ndarray, "k"] = rng.normal(size=k) * 3.0
            projected = project(values, box)
            assert abs(float(projected.sum()) - 1.0) < 1e-12, (k, box.name)
            assert projected.min() >= box.lo - 1e-12, (k, box.name)
            if np.isfinite(box.hi):
                assert projected.max() <= box.hi + 1e-12, (k, box.name)
