"""
CN: **GPU 熵 OT（Sinkhorn）** —— 逆 OT 的第三个目标函数族的后端。

    CN: 与精确/松弛循环的关系：这里**没有 master、没有 delta、没有证书**。给定 θ，Sinkhorn 给出
        熵正则化方案 `P_ε(θ)`；外层对 θ 做投影梯度。见 `core/parameters.EntropicObjective`。

    CN: **ε 是相对量**：调用方给 `relative_epsilon`，这里乘以代价矩阵的标准差
        （`regularization = relative_epsilon * cost_std`）。理由与上游 HELLO-OT 一致：
        `Σθ = 1` 固定了代价尺度，绝对 ε 在 K=64 与 K=2080 上不可比。std 也写进结果，便于事后还原。

    CN: 实现是 log-domain（数值稳定），在 torch 上做行/列 `logsumexp`，因此**原生 GPU**。

EN: **GPU entropic OT (Sinkhorn)** -- the backend for the third objective family of inverse OT.

    Relation to the exact/relaxed loops: **no master, no delta, no certificate** here. Given theta, Sinkhorn
    yields the entropic plan `P_eps(theta)`; an outer loop does projected gradient on theta. See
    `core/parameters.EntropicObjective`.

    **`epsilon` is relative**: the caller passes `relative_epsilon`, which is multiplied by the standard
    deviation of the cost matrix (`regularization = relative_epsilon * cost_std`). Same reason as upstream
    HELLO-OT: `sum(theta) = 1` fixes the cost scale, so an absolute epsilon is not comparable across K=64
    and K=2080. The std is returned too, so the absolute epsilon can be reconstructed later.

    log-domain (numerically stable) row/column `logsumexp` on torch, hence **natively GPU**.
"""

from __future__ import annotations

import dataclasses

import numpy as np
from jaxtyping import Float


@dataclasses.dataclass(frozen=True)
class SinkhornResult:
    """
    CN: 一次 Sinkhorn 的读数。`potential_source/target` 是缩放后的对数势 `f, g`
        （`P_ij = exp((f_i + g_j - C_ij)/eps) * a_i * b_j`）。
    EN: one Sinkhorn reading. `potential_source/target` are the scaled log-potentials `f, g`
        (`P_ij = exp((f_i + g_j - C_ij)/eps) * a_i * b_j`).
    """

    plan: Float[np.ndarray, "m n"]
    potential_source: Float[np.ndarray, "m"]
    potential_target: Float[np.ndarray, "n"]
    regularization: float
    cost_std: float
    iterations: int
    converged: bool
    marginal_error: float


def _cost_std(cost: Float[np.ndarray, "m n"]) -> float:
    """
    CN: 代价矩阵的标准差，用**总体** std（`ddof=0`），与上游 `negative_dot_cost_std_float64` 同一约定。
        至少取一个正的下限，避免退化实例让 `regularization` 为 0。
    EN: the cost matrix's standard deviation, **population** std (`ddof=0`), the same convention as upstream's
        `negative_dot_cost_std_float64`. Floored at a small positive value so a degenerate instance cannot
        make `regularization` zero.
    """
    value = float(np.asarray(cost, dtype=np.float64).std())
    return max(value, 1e-12)


def sinkhorn_plan_torch(
    cost: Float[np.ndarray, "m n"],
    source_mass: Float[np.ndarray, "m"],
    target_mass: Float[np.ndarray, "n"],
    *,
    relative_epsilon: float,
    max_iterations: int = 5000,
    # CN: 1e-9 是**边缘质量**的绝对误差；ε 越小需要的迭代越多（实测 rel=1.0 → 9 步，0.2 → 196 步）。
    # EN: 1e-9 is the **absolute** marginal-mass error; smaller epsilon needs more iterations
    #     (measured: rel=1.0 -> 9 steps, 0.2 -> 196).
    tolerance: float = 1e-9,
    init: tuple[Float[np.ndarray, "m"], Float[np.ndarray, "n"]] | None = None,
    device: str = "auto",
    dtype: object = None,
) -> SinkhornResult:
    """
    CN: log-domain Sinkhorn（torch，GPU 可用）。`epsilon = relative_epsilon * std(cost)`。
        `init` 用于热启动（ε 退火时把上一步的势传进来）。
    EN: log-domain Sinkhorn (torch, GPU-capable). `epsilon = relative_epsilon * std(cost)`. `init` warm-starts
        the potentials (pass the previous epsilon's potentials when annealing).

    CN: 失败即抛：非有限代价、非正 ε、非正边缘质量、以及**不收敛**都会显式报错，绝不静默返回。
        迭代次数随 `relative_epsilon` 下降而急剧增加（实测 1.0→9、0.5→25、0.2→196、0.1→>5000 步），
        所以 `max_iterations` 默认取得较大；调用方压小 ε 时要相应加大它。
    EN: fail loud: non-finite costs, non-positive epsilon, non-positive marginals, and **non-convergence** all
        raise rather than returning silently. The iteration count grows sharply as `relative_epsilon` falls
        (measured: 1.0 -> 9, 0.5 -> 25, 0.2 -> 196, 0.1 -> >5000), so `max_iterations` defaults high; raise it
        when lowering epsilon.
    """
    import torch  # noqa: PLC0415 - torch is imported lazily so `import iot` stays light

    cost_matrix = np.asarray(cost, dtype=np.float64)
    source = np.asarray(source_mass, dtype=np.float64).reshape(-1)
    target = np.asarray(target_mass, dtype=np.float64).reshape(-1)
    if cost_matrix.shape != (source.size, target.size):
        raise ValueError(f"cost shape {cost_matrix.shape} != ({source.size}, {target.size})")
    if not np.all(np.isfinite(cost_matrix)):
        raise ValueError("cost contains non-finite entries")
    if float(relative_epsilon) <= 0.0:
        raise ValueError(f"relative_epsilon must be > 0, got {relative_epsilon!r}")
    if np.any(source <= 0.0) or np.any(target <= 0.0):
        raise ValueError("source_mass and target_mass must be strictly positive")
    if str(device) == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float64 if dtype is None else dtype

    std = _cost_std(cost_matrix)
    epsilon = float(relative_epsilon) * std

    C = torch.as_tensor(cost_matrix, device=str(device), dtype=torch_dtype)
    a = torch.as_tensor(source, device=str(device), dtype=torch_dtype)
    b = torch.as_tensor(target, device=str(device), dtype=torch_dtype)
    log_a, log_b = torch.log(a), torch.log(b)
    if init is None:
        f = torch.zeros(int(source.size), device=str(device), dtype=torch_dtype)
        g = torch.zeros(int(target.size), device=str(device), dtype=torch_dtype)
    else:
        f = torch.as_tensor(np.asarray(init[0], dtype=np.float64), device=str(device),
                            dtype=torch_dtype).clone()
        g = torch.as_tensor(np.asarray(init[1], dtype=np.float64), device=str(device),
                            dtype=torch_dtype).clone()

    inv = 1.0 / epsilon
    converged = False
    performed = 0
    marginal_error = float("inf")
    for step in range(1, int(max_iterations) + 1):
        # CN: 交替更新，但不以 `f` 的变化量作收敛判据：`f, g` 只在**差一个常数**的意义下唯一，
        #     因此 `max|Δf|` 可以长期停在非零水平。判据用**边缘质量误差**（物理量）。
        # EN: alternate, but do not test convergence on the change in `f`: `f, g` are unique only up to a
        #     common shift, so `max|df|` can sit at a non-zero level indefinitely. The test is the
        #     **marginal-mass error**, which is a physical quantity.
        f = epsilon * (log_a - torch.logsumexp((g[None, :] - C) * inv, dim=1))
        g = epsilon * (log_b - torch.logsumexp((f[:, None] - C) * inv, dim=0))
        performed = step
        plan_step = torch.exp((f[:, None] + g[None, :] - C) * inv)
        row_err = float((plan_step.sum(1) - a).abs().max())
        col_err = float((plan_step.sum(0) - b).abs().max())
        marginal_error = max(row_err, col_err)
        if marginal_error <= float(tolerance):
            converged = True
            break
    # CN: **不要**再乘 `a‖b`：`f = eps*(log a - LSE((g-C)/eps))` 已经把边缘质量吸收进 `f`，
    #     因此 `exp((f+g-C)/eps)` 的行和就是 `a`。再乘一次会把总量缩成 `1/(m n)`。
    #     （旧实现 `legacy/scratch/gpu_gradient.py` 有这个错误，被照抄过一次。）
    # EN: do **not** multiply by `a b` again: `f = eps*(log a - LSE((g-C)/eps))` already absorbs the marginal
    #     into `f`, so the rows of `exp((f+g-C)/eps)` sum to `a`. Multiplying again shrinks the total to
    #     `1/(m n)`. (The legacy `legacy/scratch/gpu_gradient.py` carried this error, and it was copied once.)
    plan = torch.exp((f[:, None] + g[None, :] - C) * inv)
    row = float((plan.sum(1) - a).abs().max())
    col = float((plan.sum(0) - b).abs().max())
    if not converged:
        raise RuntimeError(
            f"Sinkhorn did not converge in {performed} iterations "
            f"(relative_epsilon={relative_epsilon:g}, regularization={epsilon:.3e}, "
            f"marginal error={marginal_error:.3e}); raise max_iterations or loosen relative_epsilon")
    return SinkhornResult(plan=np.asarray(plan.detach().cpu().numpy(), dtype=np.float64),
                          potential_source=np.asarray(f.detach().cpu().numpy(), dtype=np.float64),
                          potential_target=np.asarray(g.detach().cpu().numpy(), dtype=np.float64),
                          regularization=float(epsilon), cost_std=float(std),
                          iterations=int(performed), converged=True, marginal_error=float(max(row, col)))


__all__ = ["SinkhornResult", "sinkhorn_plan_torch"]
