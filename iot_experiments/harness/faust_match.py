"""
EN: Shared matching primitives for the FAUST correspondence study -- cost matrices, the exact-OT decoder, the
    geodesic-error metric, the supervised cost fitting, and the paired bootstrap.

    This module exists because four drivers had grown their own copy of the same four functions
    (`heatmaps`, `gates`, `arms` and `finalists` each re-implemented `_cost`/`_match`/`_errors`), and two of
    them imported private helpers out of another driver (`abstention` and `curriculum` reached into
    `finalists` for `_cost_numpy` and `fit_softmax`). A driver is not a library: everything here is
    study-independent policy that more than one experiment needs.

    Nothing in this module decides what to measure -- it only computes a cost, decodes a map, scores it, or
    fits a supervised cost. The study-specific choices (pairing, splits, selection, record building) stay in
    `studies/iot/real/faust/`.

    CN: FAUST 对应研究的**共享匹配原语**：代价矩阵、精确 OT 解码、测地误差、监督代价拟合、配对 bootstrap。
        存在的原因：四个驱动各自复制了同一批函数，另有两个驱动从别的驱动里 import 私有函数。驱动不是库，
        凡是被一个以上实验需要的、与具体研究无关的策略都放这里；"测什么"仍留在各研究模块里。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import ot
import torch
from jaxtyping import Float, Int

__all__ = [
    "bootstrap_contrast",
    "cost_matrix",
    "cost_matrix_torch",
    "cost_tensor",
    "duality_gap",
    "fit_softmax",
    "free_vram_gb",
    "geodesic_errors",
    "map_metrics",
    "match",
    "near_optimal_walk",
    "sinkhorn_log",
]


def cost_matrix(a: Float[np.ndarray, "m k"], b: Float[np.ndarray, "n k"],
                theta: Float[np.ndarray, "k"]) -> Float[np.ndarray, "m n"]:
    """
    EN: `c_ij = sum_k theta_k (a_ik - b_jk)^2` via the bilinear identity, so only `n x n` is materialised:

        `|sqrt(theta) * a_i|^2 + |sqrt(theta) * b_j|^2 - 2 <sqrt(theta) * a_i, sqrt(theta) * b_j>`.

        Negative `theta` is clipped to zero; the result is clipped at zero and made contiguous so downstream
        potpourri/POT calls never see a stride surprise.

    CN: 用双线性恒等式算 `c_ij`，只物化 `n×n`；负 `theta` 截到 0，结果截到 0 并连续化。
    """
    root = np.sqrt(np.maximum(np.asarray(theta, dtype=np.float64), 0.0))
    ax = np.asarray(a, dtype=np.float64) * root
    bx = np.asarray(b, dtype=np.float64) * root
    out = (ax * ax).sum(1)[:, None] + (bx * bx).sum(1)[None, :] - 2.0 * (ax @ bx.T)
    return np.ascontiguousarray(np.maximum(out, 0.0), dtype=np.float64)


def cost_tensor(a: Float[np.ndarray, "n k"], b: Float[np.ndarray, "n k"]
                ) -> Float[np.ndarray, "n n k"]:
    """
    EN: `Phi_ij,k = (a_ik - b_jk)^2` as an explicit tensor. Only for places that genuinely need all `K` slices;
        the fitting path never uses it (it is the `n x n x K` allocation that cost 4.9 GB at `n = 6890`).
        It transiently holds two copies, because `diff` stays alive while `diff * diff` is computed.

    CN: 显式 `(n, n, K)` 张量，只给真正需要全部 K 层的地方用；拟合路径不用它（n=6890 时那份 4.9 GB 分配）。
    """
    diff = a[:, None, :] - b[None, :, :]
    return np.asarray(diff * diff, dtype=np.float64)


def match(cost: Float[np.ndarray, "n n"], *, num_itermax: int = 10_000_000) -> Int[np.ndarray, "n"]:
    """
    EN: exact EMD with uniform marginals (POT network simplex), reduced to a map by taking each source row's
        largest plan entry.
    CN: 均匀边缘下的精确 EMD（POT 网络单纯形），按每行方案质量最大的目标归约成映射。
    """
    n = int(cost.shape[0])
    plan = ot.emd(np.full(n, 1.0 / n), np.full(n, 1.0 / n), np.asarray(cost, dtype=np.float64),
                  numItermax=int(num_itermax))
    return np.argmax(plan, axis=1).astype(np.int64)


def geodesic_errors(pred: Int[np.ndarray, "n"], geo: Float[np.ndarray, "n n"], norm: float,
                    mirror: Int[np.ndarray, "n"] | None = None) -> dict[str, float]:
    """
    EN: geodesic distance from each predicted vertex to the true one, normalised by `norm` (sqrt of the
        target's surface area), plus the miss rates at 1 % and 5 %. With `mirror`, also the left/right flip
        rate -- the fraction of *errors* where the prediction is closer to the mirrored vertex, restricted to
        vertices that are actually wrong and whose mirror is a different vertex (so exact hits are not
        miscounted).

    CN: 预测顶点到真值顶点的测地距离（按 `norm` 归一化）与 1%/5% 未命中率；给出 `mirror` 时再算左右翻转率。
    """
    rows = np.arange(int(pred.size))
    pred_idx = np.asarray(pred, dtype=np.int64)
    dist = np.asarray(geo)[rows, pred_idx] / max(norm, 1e-12)
    out = {"mean": float(dist.mean()), "median": float(np.median(dist)),
           "miss_1pct": float(np.mean(dist > 0.01)), "miss_5pct": float(np.mean(dist > 0.05))}
    if mirror is not None:
        mir = np.asarray(mirror, dtype=np.int64)
        d_mirror = np.asarray(geo)[rows, mir] / max(norm, 1e-12)
        err = dist > 0.01
        distinct = mir != rows
        mask = err & distinct
        flipped = (d_mirror < dist) & mask
        out["flip_rate"] = float(np.mean(flipped))
        out["flip_share_of_errors"] = float(flipped.sum() / mask.sum()) if mask.any() else float("nan")
        out["mirror_error_mean"] = float(d_mirror.mean())
    return out


def map_metrics(a: Float[np.ndarray, "n k"], b: Float[np.ndarray, "n k"],
                theta: Float[np.ndarray, "k"], geo: Float[np.ndarray, "n n"], norm: float,
                mirror: Int[np.ndarray, "n"] | None = None) -> dict[str, float]:
    """
    EN: the one-line experiment: build the cost under `theta`, decode with exact OT, score against the
        geodesic ground truth. Takes the precomputed `geo` because callers evaluate many thetas on the same
        pair and must not recompute the n-source Dijkstra each time.
    CN: 一行实验：按 `theta` 建代价、精确 OT 解码、对测地真值打分。`geo` 由调用方预计算并复用。
    """
    return geodesic_errors(match(cost_matrix(a, b, theta)), geo, norm, mirror)


def cost_matrix_torch(a: torch.Tensor, b: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    """
    EN: the same bilinear cost on torch tensors, differentiable in `theta` -- this is what makes the
        supervised arms trainable.
    CN: 同一双线性代价的 torch 版本，对 `theta` 可微，监督臂因此可训练。
    """
    root = torch.sqrt(torch.clamp(theta, min=1e-12))
    x = a * root
    y = b * root
    return (x * x).sum(1)[:, None] + (y * y).sum(1)[None, :] - 2.0 * (x @ y.t())


def sinkhorn_log(log_kernel: torch.Tensor, iterations: int) -> torch.Tensor:
    """
    EN: log-domain Sinkhorn with uniform marginals, returning the log-plan. Every step is a `logsumexp`, so it
        stays finite at small `tau` where the plan itself would underflow.
    CN: 对数域 Sinkhorn（均匀边缘），返回 log 方案；小 `tau` 下方案本身会下溢，而对数域每步是 `logsumexp`。
    """
    log_kernel = log_kernel.clone()
    for _ in range(int(iterations)):
        log_kernel = log_kernel - torch.logsumexp(log_kernel, dim=1)[:, None]
        log_kernel = log_kernel - torch.logsumexp(log_kernel, dim=0)[None, :]
    return log_kernel


def fit_softmax(a: Float[np.ndarray, "n k"], b: Float[np.ndarray, "n k"], *, tau: float,
                two_sided: bool, steps: int, lr: float, sinkhorn_iterations: int, device: str = "cpu"
                ) -> Float[np.ndarray, "k"]:
    """
    EN: fit `theta = softmax(u)` by minimising a cross-entropy over the true partners.

        `two_sided=False` is the one-sided row loss `mean_i CE(-C_ij/tau, i)`; `two_sided=True` applies the
        same loss to the Sinkhorn-normalised matrix, whose normalisation is wrapped in a **single** checkpoint
        segment: unrolled, 10 iterations retain ~30 `n x n` tensors (5.7 GB at `n = 6890`), which is what took
        the GPU down twice; checkpointed, the backward pass recomputes them from one saved tensor.

    CN: 用真值配对上的交叉熵拟合 `theta = softmax(u)`。`two_sided=False` 是单边行损失，`True` 在
        Sinkhorn 归一化的矩阵上用同一损失；归一化用**单段** checkpoint 包裹（展开会保留约 30 个 `n×n`
        张量，n=6890 时 5.7 GB，正是两次显存溢出的原因）。
    """
    dtype = torch.float32 if device == "cuda" else torch.float64
    xa = torch.tensor(np.asarray(a, dtype=np.float64), dtype=dtype, device=device)
    xb = torch.tensor(np.asarray(b, dtype=np.float64), dtype=dtype, device=device)
    n, k = int(xa.shape[0]), int(xa.shape[1])
    u = torch.zeros(k, dtype=dtype, device=device, requires_grad=True)
    optimiser = torch.optim.Adam([u], lr=float(lr))
    target = torch.arange(n, device=device)
    for _ in range(int(steps)):
        theta = torch.softmax(u, dim=0)
        log_kernel = -cost_matrix_torch(xa, xb, theta) / float(tau)
        if two_sided:
            log_plan = torch.utils.checkpoint.checkpoint(
                sinkhorn_log, log_kernel, int(sinkhorn_iterations), use_reentrant=False)
            loss = torch.nn.functional.nll_loss(torch.log_softmax(log_plan, dim=1), target)
        else:
            loss = torch.nn.functional.cross_entropy(log_kernel, target)
        optimiser.zero_grad(set_to_none=True)
        loss.backward()
        optimiser.step()
    with torch.no_grad():
        theta = torch.softmax(u, dim=0).detach().to("cpu", torch.float64).numpy()
    return np.asarray(theta, dtype=np.float64)


def duality_gap(theta: Float[np.ndarray, "k"], a: Float[np.ndarray, "m k"], b: Float[np.ndarray, "n k"]
                ) -> tuple[float, int]:
    """
    EN: `(R(theta), wrong)` with no labels: `R = <c(theta), pi_hat> - OT(c(theta))` measures how far the
        observed plan is from being optimal, and `wrong` counts the vertices whose decoded target differs from
        the diagonal. `R = 0` and `wrong = 0` mean the cost certifies the plan.

        Both are returned because they disagree in practice: the fit's `R` is ~1 % of `R_uniform` while its
        decoded map is wrong at thousands of vertices, so a regret tolerance is much weaker than it looks.

    CN: 无标签的 `(R(θ), wrong)`：`R` 是观测方案离最优的距离，`wrong` 是解码后在训练对上错配的顶点数。
    """
    cost = cost_matrix(a, b, theta)
    n = int(cost.shape[0])
    mass = np.full(n, 1.0 / n)
    plan = ot.emd(mass, mass, cost, numItermax=10_000_000)
    predicted = np.argmax(plan, axis=1).astype(np.int64)
    wrong = int(np.sum(predicted != np.arange(n)))
    return float(np.trace(cost) / n - np.sum(cost * plan)), wrong


def near_optimal_walk(reference: Float[np.ndarray, "k"], a: Float[np.ndarray, "m k"],
                      b: Float[np.ndarray, "n k"], *, tolerance: float, proposals: int,
                      step: float = 0.08, seed: int = 0) -> list[Float[np.ndarray, "k"]]:
    """
    EN: sample the set `{theta : R(theta) <= tolerance}` by a normalised Gaussian random walk from
        `reference`, keeping only feasible proposals. The step grows on acceptance and shrinks on rejection,
        so the walk explores the shell instead of sticking to one point. Returns the reference plus every
        accepted point (reference first, so callers can treat index 0 as the shipped answer).

        A random walk is needed because `reference` sits on the boundary: an isotropic draw from the simplex is
        rejected essentially always, and a fixed step is either entirely inside (no exploration) or entirely
        outside (no points).

    CN: 从 `reference` 出发的归一化高斯随机游走，只保留 `R <= tolerance` 的可行点；步长自适应。参考点位于边界
        上，因此不能靠均匀采样。
    """
    rng = np.random.default_rng(int(seed))
    admitted: list[Float[np.ndarray, "k"]] = [reference]
    step = float(step)
    made = 0
    while made < int(proposals):
        made += 1
        candidate = np.maximum(reference + step * rng.normal(size=int(reference.size)), 0.0)
        total = float(candidate.sum())
        if total <= 0.0:
            step *= 0.7
            continue
        candidate = candidate / total
        gap, _wrong = duality_gap(candidate, a, b)
        if abs(gap) <= tolerance:
            admitted.append(candidate)
            step = min(step * 1.15, 0.5)
        else:
            step *= 0.85
    return admitted


def free_vram_gb() -> float:
    """EN: free VRAM in GB (0.0 without CUDA). CN: 空闲显存（GB），无 CUDA 时为 0。"""
    if not torch.cuda.is_available():
        return 0.0
    free, _total = torch.cuda.mem_get_info()
    return float(free) / 2**30


def bootstrap_contrast(per_pair: Mapping[str, Mapping[str, float]], *, base: str,
                       others: Sequence[str], draws: int = 5000, seed: int = 0
                       ) -> dict[str, dict[str, float]]:
    """
    EN: paired bootstrap of `base - other` over **pairs** (never vertices), one entry per `other`. Returns
        `mean`, `ci_low`, `ci_high`, `wins` (pairs where `base` is lower) and `pairs`. A missing arm on a pair
        drops that pair from that comparison, so the two arms are always compared on the same pairs.

    CN: 按**对**做 `base - other` 的配对 bootstrap（绝不对顶点），返回均值、CI、胜场与对数。
    """
    rng = np.random.default_rng(int(seed))
    out: dict[str, dict[str, float]] = {}
    for other in others:
        diff = np.array([values[base] - values[other] for values in per_pair.values()
                         if base in values and other in values], dtype=np.float64)
        if diff.size == 0:
            continue
        boot = np.array([rng.choice(diff, size=diff.size, replace=True).mean() for _ in range(int(draws))])
        out[other] = {"mean": float(diff.mean()), "ci_low": float(np.percentile(boot, 2.5)),
                      "ci_high": float(np.percentile(boot, 97.5)), "wins": float(int((diff < 0).sum())),
                      "pairs": float(diff.size)}
    return out
