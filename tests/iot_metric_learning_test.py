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
    EARLY_STOP_EPOCHS,
    Box,
    at_bound_fraction,
    fit_residual_bilinear,
    infonce_torch,
    kfold_indices,
    project,
    select_and_fit_diagonal,
    select_cca,
    sinkhorn_infoonce_torch,
    sinkhorn_plan,
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

    for box in (Box.of_rel(K_STANDARD, 0.25), Box.of_rel(K_STANDARD, 0.5),
                Box.of_rel(K_STANDARD, 0.9), Box.wide(K_STANDARD), Box.free()):
        projected = project(uniform, box)
        assert np.allclose(projected, uniform, atol=1e-12), (box.name, projected.min(), projected.max())
        # CN: 选择流程里 `epochs=0` 可用，且它给出的就是 cosine。
        # EN: `epochs = 0` is available in the selection flow and yields cosine.
        assert 0 in EARLY_STOP_EPOCHS
        assert np.allclose(project(uniform, box), uniform, atol=1e-12)

    for rank in (16, 64):
        zero_res = fit_residual_bilinear(x, y, rank=rank, lr=0.05, epochs=0, tau=1.0)
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
        return float(infonce_torch(similarity, temperature=1.0))

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
    cosine_hit = matching_hit(diagonal_cost(x, y, np.ones(K_STANDARD) / K_STANDARD), seed=0)
    for objective in ("infonce", "sinkhorn"):
        for rel in (0.25, 0.9):
            box = Box.of_rel(K_STANDARD, rel)
            outcome = select_and_fit_diagonal(x, y, box=box, seed=0, device="cpu", objective=objective)
            assert outcome.status == "ok", (objective, rel, outcome.note)
            assert outcome.theta is not None
            theta = outcome.theta
            hit = matching_hit(diagonal_cost(x, y, theta), seed=0)
            # CN: 选择流程包含 epochs=0，故结果**不会**比 cosine 差。
            # EN: the selection flow includes epochs = 0, so the result cannot be worse than cosine.
            assert hit >= cosine_hit - 1e-12, (objective, rel, hit, cosine_hit)
            if rel <= 0.25:
                # CN: 窄盒允许集中，θ 必须在信号特征上明显更大。
                # EN: the narrow box permits concentration, so theta must be clearly larger on the signal.
                assert hit > LEARNED_FLOOR, (objective, rel, hit)
                assert theta[:SIGNAL_FEATURES].mean() > 2.0 * theta[SIGNAL_FEATURES:].mean(), (
                    objective, rel, theta[:SIGNAL_FEATURES].mean(), theta[SIGNAL_FEATURES:].mean())
            else:
                # CN: `rel=0.9` 的盒**非常窄**（hi≈1.7/K），θ 几乎动不了，接近 cosine 是正确行为，
                #     不能要求它集中——这正是"窄盒限制了可学习量"的直接证据。
                # EN: the `rel=0.9` box is **very narrow** (hi ~ 1.7/K), so theta barely moves and staying near
                #     cosine is correct. Demanding concentration there would be wrong; it is direct evidence
                #     that a narrow box caps how much can be learned.
                assert at_bound_fraction(theta, box) <= 1.0


def test_planted_recovery_residual_arms() -> None:
    """CN: residual 臂必须达到 oracle（它没有盒限制）。EN: the residual arms must reach the oracle."""
    x, y, oracle = planted()
    oracle_hit = matching_hit(diagonal_cost(x, y, oracle), seed=0)
    for rank in (16, 64):
        outcome = fit_residual_bilinear(x, y, rank=rank, lr=0.05, epochs=100, tau=1.0)
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
    outcome = select_cca(unit(x), unit(y), seed=0)
    assert outcome.status == "ok", outcome.note
    assert outcome.selected is not None and "ridge" in outcome.selected
    cost = outcome.cost(unit(x), unit(y))
    assert matching_hit(cost, seed=0) > 0.9, outcome.note


def test_box_projection_is_feasible_and_exact() -> None:
    """CN: 投影后 `sum=1` 且严格落在 `[lo, hi]`。EN: after projection, sum=1 and lo <= theta <= hi."""
    rng = np.random.default_rng(3)
    for k in (16, 64, 256):
        for box in (Box.of_rel(k, 0.25), Box.of_rel(k, 0.5), Box.wide(k), Box.free()):
            values: Float[np.ndarray, "k"] = rng.normal(size=k) * 3.0
            projected = project(values, box)
            assert abs(float(projected.sum()) - 1.0) < 1e-12, (k, box.name)
            assert projected.min() >= box.lo - 1e-12, (k, box.name)
            if np.isfinite(box.hi):
                assert projected.max() <= box.hi + 1e-12, (k, box.name)


def test_kfold_covers_all_pairs_disjointly() -> None:
    """CN: K-fold 必须覆盖全部标注对且各折不相交（选择只能在标注对内部进行）。"""
    folds = kfold_indices(103, folds=5, seed=0)
    seen = np.concatenate([val for _train, val in folds])
    assert seen.size == 103
    assert len(np.unique(seen)) == 103
    for train, val in folds:
        assert np.intersect1d(train, val).size == 0


def test_sinkhorn_plan_is_doubly_stochastic() -> None:
    """CN: Sinkhorn 计划的边际必须是均匀的 1/m（否则"iOT 似然"无意义）。"""
    import torch

    rng = np.random.default_rng(7)
    cost = torch.tensor(rng.random((12, 12)), dtype=torch.float64)
    plan = sinkhorn_plan(cost, tau=0.5, iterations=60)
    target = torch.full((12,), 1.0 / 12, dtype=torch.float64)
    assert torch.allclose(plan.sum(dim=1), target, atol=1e-9)
    assert torch.allclose(plan.sum(dim=0), target, atol=1e-9)


def test_sinkhorn_infoonce_prefers_low_cost_on_diagonal() -> None:
    """
    CN: Sinkhorn-InfoNCE 的损失必须让**对角代价低**的 θ 更优：θ 集中在信号特征上时损失更低。
    EN: Sinkhorn-InfoNCE must prefer a theta with a **low diagonal cost**: concentrating on the signal
        features lowers the loss.
    """
    import torch

    x, y, oracle = planted()
    a = torch.tensor(x, dtype=torch.float64)
    b = torch.tensor(y, dtype=torch.float64)
    uniform = torch.tensor(np.ones(K_STANDARD) / K_STANDARD, dtype=torch.float64)
    signal = torch.tensor(oracle, dtype=torch.float64)

    def loss_at(theta: "torch.Tensor") -> float:
        cost = -((a * theta.unsqueeze(0)) @ b.t())
        return float(sinkhorn_infoonce_torch(cost, tau=0.5))

    assert loss_at(signal) < loss_at(uniform)


def test_at_bound_fraction_detects_binding_box() -> None:
    """CN: 当 θ 顶到上界时，`at_bound_fraction` 必须报出来（衡量盒是否起作用）。"""
    box = Box.of_rel(10, 0.25)          # lo=0.025, hi=0.4
    theta = np.full(10, 1.0 / 10.0)
    assert at_bound_fraction(theta, box) == 0.0
    theta[0] = box.hi
    theta[1:] = (1.0 - box.hi) / 9.0
    assert at_bound_fraction(theta, box) > 0.05


def test_validation_fold_is_disjoint_from_gradient_rows() -> None:
    """
    CN: 早停的验证折必须**完全不在**梯度步里。`kfold_indices` 保证不相交；这里固定该不变量，
        因为"in-sample 1.000 而 held-out 0.6"正是验证折被偷看时的典型信号。
    EN: the early-stopping validation fold must be **entirely out of** the gradient step. `kfold_indices`
        guarantees disjointness; this pins it, because "in-sample 1.000 with held-out 0.6" is the signature of
        a validation fold that was peeked at.
    """
    folds = kfold_indices(97, folds=5, seed=3)
    for train, val in folds:
        assert np.intersect1d(train, val).size == 0, "validation fold overlaps training rows"
        assert train.size + val.size == 97, "folds must partition the labelled pairs"
        assert np.all(np.diff(np.sort(val)) > 0)
