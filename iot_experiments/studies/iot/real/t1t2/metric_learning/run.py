"""
CN: **度量学习基线研究** —— 问题：**放松版 iOT**（对角 θ、盒约束）是否胜过普通的度量学习，
    还是只与 cosine 打平？

    CN: **术语**：该 LP 是**遗憾最小化器**（regret minimiser），不是 max-margin。它最小化
        `R(θ) = <θ, g(π̂)> - R_uniform`，输出 `R'/R_uniform`；不要称其为 max-margin。

    CN: **固定的留出池**：所有 m 共用**同一个** 400 对的留出池（由 `--holdout-draw` 一次抽定），
        标注对从**其余受试者**里抽。这样不同 m 的读数在同一池上可比，而此前每个 m 的池都不一样。

    CN: **协议**：DINOv2 pool4 + PCA，K ∈ {64,128,256}；几何 `G2`（统计量取自未标注图像）；
        m ∈ {50,100,300}，3 个 draw；留出池固定 400 对；标注对与池**受试者不相交**；
        拟合空间 == 评分空间（断言）。

    CN: **所有超参数都在标注对内部选择，绝不用池**：
        * `rel`（iOT 的盒宽）∈ {0.25, 0.5, 0.75, 0.9, wide}，用标注对的 K-fold 选；
        * InfoNCE / residual：`lr` ∈ {0.01, 0.05}、`tau` ∈ {1.0, 0.1}，**step 0 也参与选择**，
          这样早停不会返回比步 0 更差的检查点；
        * CCA：`ridge` ∈ {1e-4, 1e-3, 1e-2, 1e-1}，标注对 K-fold 选择并报告所选取值。

    CN: **新臂 Sinkhorn-InfoNCE**：logits = `-C_θ/τ`，几轮可微 Sinkhorn（均匀边际），
        损失 = Sinkhorn 计划对恒等置换的交叉熵；对角 θ，同一盒；torch 自动求导；早停同上。

EN: **Metric-learning baselines** -- does relaxed iOT beat ordinary metric learning, or only tie cosine?

    **Terminology:** the LP is a **regret minimiser**, not a max-margin problem. It minimises
    `R(theta) = <theta, g(pi_hat)> - R_uniform` and reports `R'/R_uniform`.

    **Fixed held-out pool:** every m shares **one** 400-pair pool drawn once (`--holdout-draw`); labelled
    pairs are drawn from the **remaining** subjects. Readings at different m are therefore comparable on the
    same pool, whereas previously each m had its own.

    **Protocol:** DINOv2 pool4 + PCA, K in {64,128,256}; geometry `G2` (statistics from unlabelled images);
    m in {50,100,300} over 3 draws; labelled subjects disjoint from the pool; fit space asserted equal to the
    scoring space.

    **All hyperparameters are chosen inside the labelled pairs, never on the pool:** `rel` (iOT box width) in
    {0.25, 0.5, 0.75, 0.9, wide} by K-fold on the labelled pairs; InfoNCE/residual `lr` in {0.01, 0.05} and
    `tau` in {1.0, 0.1} with **step 0 as a candidate**, so early stopping cannot return a checkpoint worse than
    step 0; CCA `ridge` in {1e-4, 1e-3, 1e-2, 1e-1} by K-fold, with the chosen value reported.

    **New arm, Sinkhorn-InfoNCE:** logits `= -C_theta/tau`, a few differentiable Sinkhorn iterations (uniform
    marginals), loss = cross-entropy of the Sinkhorn plan against the identity, diagonal theta in the same box,
    torch autograd, early stopping as above.
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from jaxtyping import Bool, Float, Int

_REPO_ROOT = Path(__file__).resolve().parents[4]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from iot_experiments.harness.config import RunConfig, build_parser  # noqa: E402
from iot_experiments.harness.records import (  # noqa: E402
    RunLog,
    RunRecord,
    format_table,
    summarise,
    write_records,
)
from iot_experiments.harness.registry import apply_geometry, features, fit_geometry  # noqa: E402
from iot_experiments.harness.solver import (  # noqa: E402
    diagonal_cost,
    fit_full_lp,
    matching_hit,
    rank_metrics,
)

PROVENANCE: dict[str, object] = {
    "feature": "DINOv2 vit_small_patch14_pool4 + PCA (basis on all T1+T2 images, label-free)",
    "geometry": "G2, statistics from the held-out pool",
    "pool": "ONE fixed 400-pair held-out pool shared by every m (labelled subjects disjoint)",
    "selection": "rel / lr / tau / ridge chosen by K-fold on the LABELLED pairs only, step 0 included",
    "chance_hit": "1 / 400 = 0.0025",
    "verdict_rule": "matching hit higher on all 3 draws by >= 3 points and R@1 agrees in direction",
    "gradients": "torch autograd (no hand-derived Jacobians)",
    "terminology": "the relaxed LP is a regret minimiser, NOT max-margin",
}

MIN_GAIN: float = 0.03
HOLDOUT_PAIRS: int = 400
REL_GRID: tuple[float, ...] = (0.25, 0.5, 0.75, 0.9)
LR_GRID: tuple[float, ...] = (0.01, 0.05)
TAU_GRID: tuple[float, ...] = (1.0, 0.1)
RIDGE_GRID: tuple[float, ...] = (1e-4, 1e-3, 1e-2, 1e-1)
FOLDS: int = 5
EARLY_STOP_EPOCHS: tuple[int, ...] = (0, 1, 2, 5, 10, 20, 40, 60, 100)


class MetricLearningError(RuntimeError):
    """CN: 研究的前置条件失败。EN: a precondition of the study failed."""


@dataclass(frozen=True)
class Box:
    """
    CN: 对角 θ 的可行域，由 `rel` 参数化：`lo = rel/K`，`hi = 1/(rel*K)`。
        **`rel` 与盒宽是反比关系**：`rel = 1` 时 `lo = hi = 1/K`，即只剩均匀点（最窄）；`rel` 越小，
        盒越**宽**。`rel = 0.05` 是**最宽**的一档（K=64 时窗口 [7.8e-4, 0.3125]，hi/lo = 400），
        `rel = 0.9` 最窄（hi/lo = 1.2）。
    EN: the feasible set for diagonal theta, parameterised by `rel`: `lo = rel/K`, `hi = 1/(rel*K)`.
        **`rel` is inverse to box width**: `rel = 1` collapses the box to the single point `theta = 1/K`
        (narrowest); *smaller* `rel` is **wider**. `rel = 0.05` is the **widest** setting (K=64 window
        [7.8e-4, 0.3125], hi/lo = 400); `rel = 0.9` is the narrowest (hi/lo = 1.2).
    """

    lo: float
    hi: float
    name: str
    rel: float

    @classmethod
    def of_rel(cls, k: int, rel: float) -> "Box":
        """CN: 由 `rel` 构造盒。EN: build a box from `rel`."""
        return cls(lo=float(rel) / float(k), hi=1.0 / (float(rel) * float(k)), name=f"rel{rel:g}",
                   rel=float(rel))

    @classmethod
    def wide(cls, k: int) -> "Box":
        """CN: 宽盒 `rel=0.05`。EN: wide box at `rel = 0.05`."""
        return cls.of_rel(k, 0.05)

    @classmethod
    def free(cls) -> "Box":
        """CN: 自由臂 `θ >= 0`。EN: free arm, `theta >= 0`."""
        return cls(lo=0.0, hi=float("inf"), name="free", rel=0.0)


def project(theta: Float[np.ndarray, "k"], box: Box) -> Float[np.ndarray, "k"]:
    """
    CN: 精确欧氏投影到 `{sum=1, lo<=θ<=hi}`（对偶二分）。**对 uniform θ 必须是恒等**，由测试断言。
    EN: exact Euclidean projection onto `{sum=1, lo<=theta<=hi}` via dual bisection; it must be the
        **identity on uniform theta**, asserted by a test.
    """
    values = np.asarray(theta, dtype=np.float64).reshape(-1)
    lo, hi = float(box.lo), float(box.hi)
    if not np.isfinite(hi):
        hi = float(values.max()) + 1.0
    if lo * values.size > 1.0 + 1e-12 or hi * values.size < 1.0 - 1e-12:
        raise MetricLearningError(f"infeasible box {box.name}: K*lo={lo * values.size:.6f}")
    left = float(min(values.min(), lo) - hi - 1.0)
    right = float(max(values.max(), hi) - lo + 1.0)
    for _ in range(200):
        mid = 0.5 * (left + right)
        if float(np.clip(values - mid, lo, hi).sum()) > 1.0:
            left = mid
        else:
            right = mid
    out = np.clip(values - 0.5 * (left + right), lo, hi)
    for _ in range(100):
        gap = 1.0 - float(out.sum())
        if abs(gap) <= 1e-14:
            break
        room = (hi - out) if gap > 0 else (out - lo)
        usable = float(room[room > 0].sum())
        if usable <= 0.0:
            break
        out = np.clip(out + gap * (room / usable), lo, hi)
    return out


def at_bound_fraction(theta: Float[np.ndarray, "k"], box: Box, *, tol: float = 1e-9) -> float:
    """
    CN: θ 落在盒边界上的比例（`lo` 或 `hi` 的 `tol` 邻域内）。衡量盒是否**真正起作用**。
    EN: fraction of theta sitting on a box bound (within `tol` of `lo` or `hi`), i.e. whether the box
        **actually binds**.
    """
    values = np.asarray(theta, dtype=np.float64)
    on_lo = np.abs(values - float(box.lo)) <= tol
    on_hi = np.isfinite(box.hi) & (np.abs(values - float(box.hi)) <= tol)
    return float(np.mean(on_lo | on_hi))


def kfold_indices(m: int, *, folds: int, seed: int) -> list[tuple[Int[np.ndarray, "a"], Int[np.ndarray, "b"]]]:
    """
    CN: 在 `m` 个标注对上做 `folds` 折划分（受试者无关，直接按对索引），返回 `(train, val)` 列表。
    EN: `folds`-fold split over the `m` labelled pairs (indices, not subjects), returning `(train, val)`.
    """
    order = np.random.default_rng(int(seed)).permutation(int(m))
    out: list[tuple[Int[np.ndarray, "a"], Int[np.ndarray, "b"]]] = []
    for fold in range(int(folds)):
        val = np.sort(order[fold::int(folds)])
        train = np.sort(np.setdiff1d(order, val))
        out.append((train, val))
    return out


def pair_hit(cost: Float[np.ndarray, "n n"]) -> float:
    """
    CN: **验证折上的配对命中率**（恒等置换），不涉及池、不涉及 EMD，纯粹是"θ 是否把真配对排前面"。
    EN: the pairing hit on a **validation fold** (identity permutation): no pool, no EMD, purely "does theta
        rank the true partner first".
    """
    values = np.asarray(cost, dtype=np.float64)
    return float(np.mean(np.argmin(values, axis=1) == np.arange(values.shape[0])))


@dataclass(frozen=True)
class FitOutcome:
    """
    CN: 一个臂的拟合结果。`status != "ok"` ⇒ FAILED/发散，**必须排除**，不许用 cosine 顶替。
        `selected` 记录在验证折上选中的超参数，报告必须给出。
    EN: one arm's fit. `status != "ok"` means FAILED/diverged and the record **must be excluded**, never
        replaced by cosine. `selected` records the hyperparameters chosen on the validation folds and must be
        reported.
    """

    status: str
    theta: Float[np.ndarray, "k"] | None
    maps: tuple[Float[np.ndarray, "k r"], Float[np.ndarray, "k r"]] | None
    n_params: int
    seconds: float
    epochs: int
    note: str = ""
    cca: tuple[Float[np.ndarray, "k r"], Float[np.ndarray, "k r"]] | None = None
    selected: dict[str, object] | None = None

    def cost(self, x: Float[np.ndarray, "n k"], y: Float[np.ndarray, "n k"]) -> Float[np.ndarray, "n n"]:
        """CN: 评分空间上的代价（越小越像）。EN: cost in the scoring space (lower is a better match)."""
        if self.status != "ok":
            raise MetricLearningError(f"arm did not fit: {self.status}: {self.note}")
        xa = np.asarray(x, dtype=np.float64)
        ya = np.asarray(y, dtype=np.float64)
        if self.cca is not None:
            wx, wy = self.cca
            return -((xa @ wx) @ (ya @ wy).T)
        if self.theta is not None:
            return diagonal_cost(xa, ya, self.theta)
        if self.maps is not None:
            left, right = self.maps
            return -(xa @ ya.T + (xa @ left) @ (ya @ right).T)
        raise MetricLearningError("no parameters to score with")


def _device(name: str) -> "object":
    """CN: `auto` 时优先 CUDA。EN: prefer CUDA when `auto`."""
    import torch  # noqa: PLC0415

    if name == "cuda" or (name == "auto" and torch.cuda.is_available()):
        return torch.device("cuda")
    return torch.device("cpu")


def _project_torch(theta: "object", lo: "object", hi: "object") -> "object":
    """CN: torch 版盒∩单纯形投影（对偶二分，数值部分 detach）。EN: torch box ∩ simplex projection."""
    import torch  # noqa: PLC0415

    left = (theta.min() - hi - 1.0).clone()
    right = (theta.max() - lo + 1.0).clone()
    for _ in range(80):
        mid = 0.5 * (left + right)
        if float(torch.clamp(theta.detach() - mid, min=lo, max=hi).sum().detach()) > 1.0:
            left = mid
        else:
            right = mid
    return torch.clamp(theta - 0.5 * (left + right), min=lo, max=hi)


def infonce_torch(similarity: "object", *, temperature: float) -> "object":
    """
    CN: 对称 InfoNCE，**输入是相似度**（真配对上更大），先除以 τ 与 `mean|z|`。
    EN: symmetric InfoNCE; the input is the **similarity** (larger on true pairs), divided by tau and by
        `mean|z|`.
    """
    import torch  # noqa: PLC0415

    z = similarity / float(temperature)
    s = z / torch.clamp(z.abs().mean(), min=1e-12)
    m = int(s.shape[0])
    target = torch.arange(m, device=s.device)
    return 0.5 * (torch.nn.functional.cross_entropy(s, target)
                  + torch.nn.functional.cross_entropy(s.t(), target))


def sinkhorn_plan(cost: "object", *, tau: float, iterations: int = 8) -> "object":
    """
    CN: 均匀边际下的可微 Sinkhorn（log 域），返回传输计划 `P`（行和为 1/m）。
        logits = `-cost/τ`，每轮做行/列 logsumexp 归一。
    EN: differentiable log-domain Sinkhorn under uniform marginals, returning the plan `P` (rows sum to 1/m).
        Logits are `-cost/tau`, normalised by row/column logsumexp each iteration.
    """
    import torch  # noqa: PLC0415

    m = int(cost.shape[0])
    log_k = -cost / float(tau)
    log_u = torch.zeros(m, dtype=cost.dtype, device=cost.device)
    log_v = torch.zeros(m, dtype=cost.dtype, device=cost.device)
    log_marginal = -torch.log(torch.tensor(float(m), dtype=cost.dtype, device=cost.device))
    for _ in range(int(iterations)):
        log_u = log_marginal - torch.logsumexp(log_k + log_v.unsqueeze(0), dim=1)
        log_v = log_marginal - torch.logsumexp(log_k + log_u.unsqueeze(1), dim=0)
    return torch.exp(log_k + log_u.unsqueeze(1) + log_v.unsqueeze(0))


def sinkhorn_infoonce_torch(cost: "object", *, tau: float, iterations: int = 8) -> "object":
    """
    CN: **Sinkhorn-InfoNCE（熵正则 iOT 似然）**：损失 = Sinkhorn 计划对恒等置换的交叉熵，
        即 `-sum_i log P_ii`（P 已按行归一）。计划来自 `-C/τ`，因此**代价越小越可能被选中**。
    EN: **Sinkhorn-InfoNCE (entropic iOT likelihood)**: the loss is the cross-entropy of the Sinkhorn plan
        against the identity, i.e. `-sum_i log P_ii` after row normalisation. The plan comes from `-C/tau`, so a
        **lower cost is more likely to be matched**.
    """
    import torch  # noqa: PLC0415

    plan = sinkhorn_plan(cost, tau=float(tau), iterations=int(iterations))
    rows = plan / torch.clamp(plan.sum(dim=1, keepdim=True), min=1e-30)
    m = int(plan.shape[0])
    target = torch.arange(m, device=plan.device)
    return -torch.log(torch.clamp(rows[target, target], min=1e-30)).mean()


def fit_cca(x: Float[np.ndarray, "m k"], y: Float[np.ndarray, "m k"], *, ridge: float,
            n_components: int = 0) -> FitOutcome:
    """
    CN: 岭正则 CCA（闭式）。白化为单位方差 `W = Σ^{-1/2}U/√m`，按相关 > 0.1 截断噪声方向。
        `ridge` 由外层 K-fold 选择（见 `main`），这里只负责一次拟合。
    EN: ridge CCA in closed form. Whitening is unit-variance `W = S^{-1/2} U / sqrt(m)`, with components
        above correlation 0.1 kept. `ridge` is chosen by the outer K-fold; this function does one fit.
    """
    start = time.perf_counter()
    a = np.asarray(x, dtype=np.float64)
    b = np.asarray(y, dtype=np.float64)
    m, k = int(a.shape[0]), int(a.shape[1])
    sxx = (a.T @ a) / float(m) + float(ridge) * np.eye(k)
    syy = (b.T @ b) / float(m) + float(ridge) * np.eye(k)
    sxy = (a.T @ b) / float(m)

    def inv_sqrt(matrix: Float[np.ndarray, "k k"]) -> Float[np.ndarray, "k k"]:
        vals, vecs = np.linalg.eigh(matrix)
        vals = np.maximum(vals, 1e-12)
        return (vecs / np.sqrt(vals)[None, :]) @ vecs.T

    sx, sy = inv_sqrt(sxx), inv_sqrt(syy)
    u, singular, vt = np.linalg.svd(sx @ sxy @ sy)
    keep = int(n_components) if n_components > 0 else max(1, int(np.sum(singular > 0.1)))
    keep = min(keep, int(singular.size))
    root = float(np.sqrt(float(m)))
    wx = (sx @ u[:, :keep]) / root
    wy = (sy @ vt.T[:, :keep]) / root
    if not (np.all(np.isfinite(wx)) and np.all(np.isfinite(wy))):
        return FitOutcome(status="FAILED_nan", theta=None, maps=None, n_params=int(2 * k * keep),
                          seconds=time.perf_counter() - start, epochs=0, note="non-finite CCA maps")
    return FitOutcome(status="ok", theta=None, maps=None, n_params=int(2 * k * keep),
                      seconds=time.perf_counter() - start, epochs=0,
                      note=f"ridge={ridge:g} keep={keep} canon_top={float(singular[0]):.3f}",
                      cca=(wx, wy))


def select_cca(x: Float[np.ndarray, "m k"], y: Float[np.ndarray, "m k"], *, seed: int) -> FitOutcome:
    """
    CN: 用标注对的 K-fold 选 `ridge`（按验证折配对命中率；同分取更小的 ridge），再在全标注对上重拟合。
    EN: choose `ridge` by K-fold on the labelled pairs (validation-fold pairing hit; ties go to the smaller
        ridge) and refit on all labelled pairs.
    """
    best: tuple[float, float] | None = None
    m = int(np.asarray(x).shape[0])
    for ridge in RIDGE_GRID:
        scores = []
        for train, val in kfold_indices(m, folds=FOLDS, seed=seed):
            fit = fit_cca(x[train], y[train], ridge=ridge)
            if fit.status != "ok":
                scores.append(float("nan"))
                continue
            scores.append(pair_hit(fit.cost(x[val], y[val])))
        mean = float(np.nanmean(scores)) if np.isfinite(scores).any() else float("nan")
        if not np.isfinite(mean):
            continue
        if best is None or mean > best[1] + 1e-12:
            best = (ridge, mean)
    ridge = best[0] if best is not None else RIDGE_GRID[0]
    fit = fit_cca(x, y, ridge=ridge)
    return FitOutcome(status=fit.status, theta=fit.theta, maps=fit.maps, n_params=fit.n_params,
                      seconds=fit.seconds, epochs=0, note=fit.note, cca=fit.cca,
                      selected={"ridge": ridge})


def _make_diagonal_state(x: Float[np.ndarray, "a k"], y: Float[np.ndarray, "a k"], *, box: Box, lr: float,
                         device: str) -> dict[str, object]:
    """
    CN: 建一个可**增量**训练的优化状态（参数 `w`、Adam 与常量张量），供多个 epoch 检查点复用。
    EN: build an **incrementally trainable** optimiser state (parameter `w`, Adam, constant tensors) reusable
        across epoch checkpoints.
    """
    import torch  # noqa: PLC0415

    dev = _device(device)
    a = torch.tensor(np.asarray(x, dtype=np.float64), dtype=torch.float64, device=dev)
    b = torch.tensor(np.asarray(y, dtype=np.float64), dtype=torch.float64, device=dev)
    w = torch.zeros(int(a.shape[1]), dtype=torch.float64, device=dev, requires_grad=True)
    opt = torch.optim.Adam([w], lr=float(lr))
    lo = torch.tensor(float(box.lo), dtype=torch.float64, device=dev)
    hi = torch.tensor(float(box.hi) if np.isfinite(box.hi) else 1e3, dtype=torch.float64, device=dev)
    return {"a": a, "b": b, "w": w, "opt": opt, "lo": lo, "hi": hi, "box": box}


def _train_diagonal_steps(state: dict[str, object], *, box: Box, tau: float, steps: int,
                          objective: str) -> None:
    """
    CN: 在给定状态上再训练 `steps` 步（目标 `infonce` 或 `sinkhorn`）；数据已在状态里。
    EN: advance the state by `steps` more steps with the `infonce` or `sinkhorn` objective; the data lives in
        the state.
    """
    import torch  # noqa: PLC0415

    a = state["a"]
    b = state["b"]
    w = state["w"]
    opt = state["opt"]
    lo = state["lo"]
    hi = state["hi"]
    for _ in range(int(steps)):
        opt.zero_grad(set_to_none=True)
        theta = torch.softmax(w, dim=0)
        projected = _project_torch(theta, lo, hi)
        theta = theta + (projected - theta).detach()
        similarity = (a * theta.unsqueeze(0)) @ b.t()
        loss = (infonce_torch(similarity, temperature=float(tau)) if objective == "infonce"
                else sinkhorn_infoonce_torch(-similarity, tau=float(tau)))
        loss.backward()
        torch.nn.utils.clip_grad_norm_([w], max_norm=10.0)
        opt.step()


def _state_theta(state: dict[str, object], box: Box) -> Float[np.ndarray, "k"]:
    """CN: 读取当前 θ（已投影）。EN: read the current (projected) theta."""
    import torch  # noqa: PLC0415

    with torch.no_grad():
        theta = _project_torch(torch.softmax(state["w"], dim=0), state["lo"], state["hi"])
    return theta.detach().cpu().numpy().copy()


def _train_diagonal(x: Float[np.ndarray, "a k"], y: Float[np.ndarray, "a k"], *, box: Box, lr: float,
                    tau: float, epochs: int, device: str, objective: str) -> Float[np.ndarray, "k"]:
    """
    CN: 从零训练 `epochs` 步并返回 θ。EN: train `epochs` steps from scratch and return theta.
    """
    state = _make_diagonal_state(x, y, box=box, lr=lr, device=device)
    _train_diagonal_steps(state, box=box, tau=tau, steps=int(epochs), objective=objective)
    return _state_theta(state, box)


def select_and_fit_diagonal(x: Float[np.ndarray, "m k"], y: Float[np.ndarray, "m k"], *, box: Box,
                            seed: int, device: str, objective: str) -> FitOutcome:
    """
    CN: 在标注对上用 K-fold 选择 `(lr, tau, epochs)`，再在全标注对上重拟合。

        CN: **`epochs=0` 也是候选**，因此早停**不可能**返回比步 0（= cosine）更差的检查点。
        选择准则是验证折的配对命中率；同分取更小的 `lr`、更大的 `tau`、更少的 `epochs`（偏好更简单）。

    EN: choose `(lr, tau, epochs)` by K-fold on the labelled pairs, then refit on all of them.

        **`epochs = 0` is a candidate**, so early stopping **cannot** return a checkpoint worse than step 0
        (which equals cosine). The criterion is the validation-fold pairing hit; ties prefer the smaller `lr`,
        the larger `tau` and fewer epochs (a simpler fit).
    """
    start = time.perf_counter()
    m = int(np.asarray(x).shape[0])
    folds = kfold_indices(m, folds=FOLDS, seed=seed)
    # CN: 每次训练只在 train 折上进行，**增量**训练到各个 epoch 检查点，避免重复从零训练。
    # EN: each run trains on the train folds only and trains **incrementally** to each epoch checkpoint, so
    #     nothing is retrained from scratch.
    baseline = project(np.ones(int(x.shape[1])) / float(x.shape[1]), box)
    curve: dict[tuple[float, float, int], list[float]] = {}
    for lr in LR_GRID:
        for tau in TAU_GRID:
            per_epoch: dict[int, list[float]] = {e: [] for e in EARLY_STOP_EPOCHS}
            for train, val in folds:
                # CN: **断言**验证折与梯度步使用的行**不相交**。这是"早停不偷看验证折"的硬保证，
                #     不能只靠代码阅读；若相交则在此立即失败。
                # EN: **assert** the validation fold is disjoint from the rows the gradient step uses. This is
                #     the hard guarantee that early stopping does not peek; a violation fails here rather than
                #     producing a suspicious 1.000 in-sample score.
                if np.intersect1d(train, val).size:
                    raise MetricLearningError("validation fold overlaps the gradient-step rows")
                state = _make_diagonal_state(x[train], y[train], box=box, lr=lr, device=device)
                trained = 0
                for epoch in EARLY_STOP_EPOCHS:
                    steps = int(epoch) - trained
                    if steps > 0:
                        _train_diagonal_steps(state, box=box, tau=tau, steps=steps, objective=objective)
                        trained = int(epoch)
                    current = _state_theta(state, box) if trained > 0 else baseline
                    per_epoch[int(epoch)].append(pair_hit(diagonal_cost(x[val], y[val], current)))
            for epoch, scores in per_epoch.items():
                curve[(lr, tau, epoch)] = scores
    best: tuple[float, float, int, float] | None = None
    for (lr, tau, epoch), scores in sorted(curve.items(), key=lambda kv: (kv[0][0], -kv[0][1], kv[0][2])):
        mean = float(np.nanmean(scores)) if np.isfinite(scores).any() else float("nan")
        if not np.isfinite(mean):
            continue
        if best is None or mean > best[3] + 1e-12:
            best = (lr, tau, epoch, mean)
    if best is None:
        return FitOutcome(status="FAILED_selection", theta=None, maps=None, n_params=int(x.shape[1]),
                          seconds=time.perf_counter() - start, epochs=0, note="no finite validation score")
    lr, tau, epochs, val_score = best
    if epochs == 0:
        theta = project(np.ones(int(x.shape[1])) / float(x.shape[1]), box)
    else:
        theta = _train_diagonal(x, y, box=box, lr=lr, tau=tau, epochs=epochs, device=device,
                                objective=objective)
    return FitOutcome(status="ok", theta=theta, maps=None, n_params=int(x.shape[1]),
                      seconds=time.perf_counter() - start, epochs=int(epochs),
                      note=f"{objective} lr={lr:g} tau={tau:g} epochs={epochs} box={box.name}",
                      selected={"lr": lr, "tau": tau, "epochs": int(epochs),
                                "val_hit": round(val_score, 4)})


def fit_residual_bilinear(x: Float[np.ndarray, "m k"], y: Float[np.ndarray, "m k"], *, rank: int,
                          lr: float, epochs: int, tau: float, device: str = "auto") -> FitOutcome:
    """
    CN: **残差双线性**：`s = xᵀ(I + UVᵀ)y`，`V=0` ⇒ 步 0 严格等于 cosine。梯度由 torch 自动求导。
    EN: **residual bilinear**: `s = x^T (I + U V^T) y` with `V = 0`, so step 0 is exactly cosine. Gradients
        from torch autograd.
    """
    import torch  # noqa: PLC0415

    start = time.perf_counter()
    dev = _device(device)
    a = torch.tensor(np.asarray(x, dtype=np.float64), dtype=torch.float64, device=dev)
    b = torch.tensor(np.asarray(y, dtype=np.float64), dtype=torch.float64, device=dev)
    k = int(a.shape[1])
    r = int(min(rank, k))
    gen = torch.Generator(device="cpu").manual_seed(0)
    u = (torch.randn(k, r, generator=gen, dtype=torch.float64) * 1e-3).to(dev).requires_grad_(True)
    v = torch.zeros(k, r, dtype=torch.float64, device=dev, requires_grad=True)
    opt = torch.optim.Adam([u, v], lr=float(lr))
    for _ in range(int(epochs)):
        opt.zero_grad(set_to_none=True)
        score = a @ b.t() + (a @ u) @ (b @ v).t()
        loss = infonce_torch(score, temperature=float(tau))
        loss.backward()
        torch.nn.utils.clip_grad_norm_([u, v], max_norm=10.0)
        opt.step()
    with torch.no_grad():
        if not (bool(torch.isfinite(u).all()) and bool(torch.isfinite(v).all())):
            return FitOutcome(status="FAILED_diverged", theta=None, maps=None, n_params=int(2 * k * r),
                              seconds=time.perf_counter() - start, epochs=int(epochs),
                              note="weights diverged")
        maps = (u.detach().cpu().numpy().copy(), v.detach().cpu().numpy().copy())
    return FitOutcome(status="ok", theta=None, maps=maps, n_params=int(2 * k * r),
                      seconds=time.perf_counter() - start, epochs=int(epochs),
                      note=f"lr={lr:g} tau={tau:g} r={r} residual")


def select_residual(x: Float[np.ndarray, "m k"], y: Float[np.ndarray, "m k"], *, rank: int, seed: int,
                    device: str) -> FitOutcome:
    """
    CN: 同对角臂的策略：K-fold 选 `(lr, tau, epochs)`，**`epochs=0` 是候选**，故不会比步 0 差。
    EN: the same policy as the diagonal arms: K-fold choice of `(lr, tau, epochs)` with **`epochs = 0` a
        candidate**, so it can never be worse than step 0.
    """
    start = time.perf_counter()
    m = int(np.asarray(x).shape[0])
    folds = kfold_indices(m, folds=FOLDS, seed=seed)
    best: tuple[float, float, int, float] | None = None
    for lr in LR_GRID:
        for tau in TAU_GRID:
            for epoch in EARLY_STOP_EPOCHS:
                scores: list[float] = []
                for train, val in folds:
                    # CN: 同上，残差臂也必须保证验证折不在梯度步里。
                    # EN: as above, the residual arm must also keep the validation fold out of the gradient.
                    if np.intersect1d(train, val).size:
                        raise MetricLearningError("validation fold overlaps the gradient-step rows")
                    if epoch == 0:
                        baseline = diagonal_cost(x[val], y[val],
                                                 np.ones(int(x.shape[1])) / float(x.shape[1]))
                        scores.append(pair_hit(baseline))
                        continue
                    fit = fit_residual_bilinear(x[train], y[train], rank=rank, lr=lr, epochs=int(epoch),
                                                tau=tau, device=device)
                    if fit.status != "ok":
                        scores.append(float("nan"))
                        continue
                    scores.append(pair_hit(fit.cost(x[val], y[val])))
                mean = float(np.nanmean(scores)) if np.isfinite(scores).any() else float("nan")
                if not np.isfinite(mean):
                    continue
                if best is None or mean > best[3] + 1e-12:
                    best = (lr, tau, int(epoch), mean)
    if best is None:
        return FitOutcome(status="FAILED_selection", theta=None, maps=None, n_params=int(2 * x.shape[1]),
                          seconds=time.perf_counter() - start, epochs=0, note="no finite validation score")
    lr, tau, epoch, val_score = best
    if epoch == 0:
        return FitOutcome(status="ok", theta=None, maps=(np.zeros((int(x.shape[1]), 1)),
                                                         np.zeros((int(x.shape[1]), 1))),
                          n_params=int(2 * x.shape[1]), seconds=time.perf_counter() - start, epochs=0,
                          note="residual selected epochs=0 (no better than cosine)",
                          selected={"lr": lr, "tau": tau, "epochs": 0, "val_hit": round(val_score, 4)})
    fit = fit_residual_bilinear(x, y, rank=rank, lr=lr, epochs=int(epoch), tau=tau, device=device)
    return FitOutcome(status=fit.status, theta=fit.theta, maps=fit.maps, n_params=fit.n_params,
                      seconds=time.perf_counter() - start, epochs=int(epoch), note=fit.note,
                      selected={"lr": lr, "tau": tau, "epochs": int(epoch), "val_hit": round(val_score, 4)})


def select_iot(x: Float[np.ndarray, "m k"], y: Float[np.ndarray, "m k"], *, solver: str,
               solve_limit_s: float, seed: int) -> FitOutcome:
    """
    CN: iOT 的 `rel` 用标注对 K-fold 选择：每次在 train 折上解一次遗憾最小化 LP，在 val 折上算配对命中率。
        选择后在全标注对上重拟合。**注意：这是遗憾最小化器，不是 max-margin。**
    EN: choose iOT's `rel` by K-fold on the labelled pairs: solve the regret-minimiser LP on the train folds
        and score the validation fold, then refit on all labelled pairs. This is a **regret minimiser**, not a
        max-margin problem.
    """
    start = time.perf_counter()
    m = int(np.asarray(x).shape[0])
    folds = kfold_indices(m, folds=FOLDS, seed=seed)
    candidates = list(REL_GRID) + [0.05]
    best: tuple[float, float] | None = None
    for rel in candidates:
        scores: list[float] = []
        for train, val in folds:
            fit = fit_full_lp(x[train], y[train], solver=solver, solve_limit_s=solve_limit_s,
                              box_relative=float(rel))
            if fit.status != "ok":
                scores.append(float("nan"))
                continue
            scores.append(pair_hit(diagonal_cost(x[val], y[val], fit.theta)))
        mean = float(np.nanmean(scores)) if np.isfinite(scores).any() else float("nan")
        if not np.isfinite(mean):
            continue
        # CN: 同分取更宽的盒（rel 更小），因为它更接近"不用盒约束"的默认。
        # EN: ties go to the wider box (smaller rel), closer to the unconstrained default.
        if best is None or mean > best[1] + 1e-12:
            best = (float(rel), mean)
    rel = best[0] if best is not None else 0.25
    fit = fit_full_lp(x, y, solver=solver, solve_limit_s=solve_limit_s, box_relative=float(rel))
    box = Box.of_rel(int(x.shape[1]), float(rel))
    bound = at_bound_fraction(fit.theta, box) if fit.status == "ok" else float("nan")
    return FitOutcome(status=fit.status, theta=fit.theta if fit.status == "ok" else None, maps=None,
                      n_params=int(x.shape[1]), seconds=time.perf_counter() - start, epochs=0,
                      note=f"regret-minimiser rel={rel:g} R'/R_uni={fit.R_ratio:.3f} "
                           f"at_bound={bound:.2f}",
                      selected={"rel": rel, "R_ratio": fit.R_ratio, "R_prime": fit.R_prime,
                                "R_uniform": fit.R_uniform, "at_bound": bound,
                                "val_hit": round(best[1], 4) if best else None})


def evaluate(cost: Float[np.ndarray, "n n"], *, draw: int, fraction: float = 0.5) -> dict[str, float]:
    """
    CN: 与 `studies/pool` 一致：matching hit、R@1、margin-P@`fraction`。
    EN: as `studies/pool`: matching hit, R@1, margin-P at `fraction` coverage.
    """
    import ot  # noqa: PLC0415

    values = np.asarray(cost, dtype=np.float64)
    hit = matching_hit(values, seed=int(draw))
    ranks = rank_metrics(values)
    row_sorted = np.sort(values, axis=1)
    col_sorted = np.sort(values, axis=0)
    margin = np.minimum(row_sorted[:, 1] - row_sorted[:, 0], col_sorted[1, :] - col_sorted[0, :])
    n = int(values.shape[0])
    weights = np.ones(n, dtype=np.float64) / float(n)
    perm: Int[np.ndarray, "n"] = np.random.default_rng(int(draw)).permutation(n)
    plan = ot.emd(weights, weights, values[:, perm], numItermax=10_000_000)
    hits: Bool[np.ndarray, "n"] = np.argmax(plan, axis=1) == np.argsort(perm)
    order: Int[np.ndarray, "n"] = np.argsort(-margin, kind="stable")
    k = max(1, int(round(float(fraction) * n)))
    return {"hit": hit, "R@1": float(ranks["R@1"]),
            f"P@{int(round(fraction * 100))}": float(np.mean(np.asarray(hits, dtype=np.float64)[order[:k]]))}


def verdict(higher: list[float], base: list[float], *, min_gain: float = MIN_GAIN) -> str:
    """
    CN: **预注册判定**：每个 draw 都高 ≥ `min_gain` ⇒ `WIN`；全部不高于 ⇒ `LOSS`；否则 `TIE`。
    EN: **pre-registered verdict**: higher on every draw by >= `min_gain` is `WIN`; never higher is `LOSS`;
        otherwise `TIE`.
    """
    if len(higher) != len(base) or not higher:
        return "-"
    diffs = [float(a) - float(b) for a, b in zip(higher, base)]
    if all(d >= float(min_gain) for d in diffs):
        return "WIN"
    if all(d <= 0.0 for d in diffs):
        return "LOSS"
    return "TIE"


@dataclass(frozen=True)
class MetricLearningConfig(RunConfig):
    """CN: 研究自己的旋钮。EN: the study's own knobs."""

    section = "metric_learning"
    study: str = "metric_learning"
    feature: str = "dinov2-pool4"
    ks: tuple[int, ...] = (64, 128, 256)
    ms: tuple[int, ...] = (50, 100, 300)
    holdout_pairs: int = HOLDOUT_PAIRS
    holdout_draw: int = 0
    fit_device: str = "auto"

    @classmethod
    def add_study_arguments(cls, parser: argparse.ArgumentParser) -> None:
        """CN: 只加本研究特有的参数。EN: only this study's flags."""
        parser.add_argument("--feature", default="dinov2-pool4", choices=["dinov2-pool4"])
        parser.add_argument("--ks", type=int, nargs="+", default=[64, 128, 256])
        parser.add_argument("--ms", type=int, nargs="+", default=[50, 100, 300])
        parser.add_argument("--holdout-pairs", type=int, default=HOLDOUT_PAIRS,
                            help="size of the ONE fixed held-out pool shared by every m")
        parser.add_argument("--holdout-draw", type=int, default=0,
                            help="seed that fixes the held-out pool (constant across m)")
        parser.add_argument("--fit-device", default="auto", choices=["auto", "cpu", "cuda"])


ARM_ORDER: tuple[str, ...] = ("cosine", "iot", "cca", "infonce", "sinkhorn", "residual16", "residual64")


def main(argv: list[str] | None = None) -> int:
    """CN: 入口。EN: entry point."""
    parser = build_parser(MetricLearningConfig)
    args = parser.parse_args(list(argv) if argv is not None else None)
    config = MetricLearningConfig.from_namespace(args, parser)

    records: list[RunRecord] = []
    t_all = time.perf_counter()
    log = RunLog(config.run_dir / "run.log")
    log.line(f"config: ks={config.ks} ms={config.ms} draws={config.draws} "
             f"holdout_pairs={config.holdout_pairs} holdout_draw={config.holdout_draw} "
             f"solver={config.solver} device={config.fit_device}")
    for k in config.ks:
        x, y = features(str(config.feature), dim=int(k))
        n_subjects = int(x.shape[0])
        # CN: **一个固定的留出池**，所有 m 共用；标注对从其余受试者中抽。
        # EN: **one fixed held-out pool** shared by every m; labelled pairs come from the remaining subjects.
        pool_rng = np.random.default_rng(10_000 + int(config.holdout_draw))
        pool_order = pool_rng.permutation(n_subjects)
        pool_subjects = np.sort(pool_order[: int(config.holdout_pairs)])
        remaining = np.sort(pool_order[int(config.holdout_pairs):])
        stats = fit_geometry(x[pool_subjects], y[pool_subjects], "G2")
        x_pool, y_pool = apply_geometry(x[pool_subjects], y[pool_subjects], stats)
        for arr in (x_pool, y_pool):
            norms = np.linalg.norm(np.asarray(arr, dtype=np.float64), axis=1)
            if not np.allclose(norms, 1.0, atol=1e-8):
                raise SystemExit(f"pool space not L2-normalised: [{norms.min():.6f}, {norms.max():.6f}]")
        uniform = np.ones(int(k)) / float(k)
        cosine_pool = diagonal_cost(x_pool, y_pool, uniform)

        for m in config.ms:
            for draw in config.draws:
                label_rng = np.random.default_rng(int(draw))
                labelled = np.sort(label_rng.choice(remaining, size=int(m), replace=False))
                x_fit, y_fit = apply_geometry(x[labelled], y[labelled], stats)
                norms = np.linalg.norm(np.asarray(x_fit, dtype=np.float64), axis=1)
                if not np.allclose(norms, 1.0, atol=1e-8):
                    raise SystemExit(f"fit space not L2-normalised: [{norms.min():.6f}, {norms.max():.6f}]")
                if np.intersect1d(labelled, pool_subjects).size:
                    raise SystemExit("labelled and pool subjects overlap")

                base = evaluate(cosine_pool, draw=draw)
                records.append(RunRecord(study=str(config.study), geometry="G2", arm="cosine",
                                         k=int(k), m=int(m), draw=int(draw), metrics=base,
                                         config=config.to_dict(), extra={"n_params": 0}))
                log.line(f"[K={k} m={m} d={draw}] cosine  hit={base['hit']:.4f} R@1={base['R@1']:.4f}")

                outcome = select_iot(x_fit, y_fit, solver=str(config.solver),
                                     solve_limit_s=float(config.solve_limit_s), seed=int(draw))
                sel = outcome.selected or {}
                log.line(f"[K={k} m={m} d={draw}] iot     rel={sel.get('rel')} "
                      f"R'/R_uni={sel.get('R_ratio')} at_bound={sel.get('at_bound')} "
                      f"val_hit={sel.get('val_hit')}")
                if outcome.status != "ok":
                    records.append(RunRecord(study=str(config.study), geometry="G2", arm="iot",
                                             k=int(k), m=int(m), draw=int(draw), metrics={},
                                             config=config.to_dict(), excluded=True,
                                             status=outcome.status))
                else:
                    met = evaluate(outcome.cost(x_pool, y_pool), draw=draw)
                    ins = evaluate(outcome.cost(x_fit, y_fit), draw=draw)
                    records.append(RunRecord(study=str(config.study), geometry="G2", arm="iot",
                                             k=int(k), m=int(m), draw=int(draw), metrics=met,
                                             config=config.to_dict(),
                                             extra={"n_params": outcome.n_params,
                                                    "seconds": round(outcome.seconds, 2),
                                                    "in_sample_hit": ins["hit"], **sel}))
                    log.line(f"[K={k} m={m} d={draw}] iot     hit={met['hit']:.4f} R@1={met['R@1']:.4f} "
                          f"in={ins['hit']:.4f} {outcome.seconds:.1f}s")

                plans: dict[str, tuple[Box, str]] = {
                    "cca": (Box.of_rel(int(k), 0.25), "cca"),
                    "infonce": (Box.of_rel(int(k), 0.25), "infonce"),
                    "sinkhorn": (Box.of_rel(int(k), 0.25), "sinkhorn"),
                }
                for arm, (box, objective) in plans.items():
                    if objective == "cca":
                        fit = select_cca(x_fit, y_fit, seed=int(draw))
                    else:
                        fit = select_and_fit_diagonal(x_fit, y_fit, box=box, seed=int(draw),
                                                      device=str(config.fit_device), objective=objective)
                    sel = fit.selected or {}
                    if fit.status != "ok":
                        records.append(RunRecord(study=str(config.study), geometry="G2", arm=arm,
                                                 k=int(k), m=int(m), draw=int(draw), metrics={},
                                                 config=config.to_dict(), excluded=True,
                                                 status=fit.status, extra={"note": fit.note}))
                        log.line(f"[K={k} m={m} d={draw}] {arm:<8} FAILED {fit.status} {sel}")
                        continue
                    met = evaluate(fit.cost(x_pool, y_pool), draw=draw)
                    ins = evaluate(fit.cost(x_fit, y_fit), draw=draw)
                    records.append(RunRecord(study=str(config.study), geometry="G2", arm=arm,
                                             k=int(k), m=int(m), draw=int(draw), metrics=met,
                                             config=config.to_dict(),
                                             extra={"n_params": fit.n_params,
                                                    "seconds": round(fit.seconds, 2),
                                                    "in_sample_hit": ins["hit"], **sel}))
                    log.line(f"[K={k} m={m} d={draw}] {arm:<8} hit={met['hit']:.4f} R@1={met['R@1']:.4f} "
                          f"in={ins['hit']:.4f} sel={sel} {fit.seconds:.1f}s")

                for rank in (16, 64):
                    arm = f"residual{rank}"
                    fit = select_residual(x_fit, y_fit, rank=rank, seed=int(draw),
                                          device=str(config.fit_device))
                    sel = fit.selected or {}
                    if fit.status != "ok":
                        records.append(RunRecord(study=str(config.study), geometry="G2", arm=arm,
                                                 k=int(k), m=int(m), draw=int(draw), metrics={},
                                                 config=config.to_dict(), excluded=True,
                                                 status=fit.status, extra={"note": fit.note}))
                        log.line(f"[K={k} m={m} d={draw}] {arm:<10} FAILED {fit.status}")
                        continue
                    met = evaluate(fit.cost(x_pool, y_pool), draw=draw)
                    ins = evaluate(fit.cost(x_fit, y_fit), draw=draw)
                    records.append(RunRecord(study=str(config.study), geometry="G2", arm=arm,
                                             k=int(k), m=int(m), draw=int(draw), metrics=met,
                                             config=config.to_dict(),
                                             extra={"n_params": fit.n_params,
                                                    "seconds": round(fit.seconds, 2),
                                                    "in_sample_hit": ins["hit"], **sel}))
                    log.line(f"[K={k} m={m} d={draw}] {arm:<10} hit={met['hit']:.4f} R@1={met['R@1']:.4f} "
                          f"in={ins['hit']:.4f} sel={sel} {fit.seconds:.1f}s")

    # CN: **断言 cosine 与 m / draw 无关**。cosine 是**无标签**的：它只看固定的 400 对留出池，
    #     因此对同一个 K 必须给出**完全相同**的分数。若这里变化，说明留出池、几何统计量或
    #     打乱被 m/draw 污染了——在读数之前必须先查清（此前我误把臂之间的差异当成 cosine 的波动）。
    # EN: **assert cosine is independent of m and draw.** Cosine is **label-free**: it sees only the fixed
    #     400-pair pool, so for a given K it must be **exactly** the same number. Any variation means the
    #     pool, the geometry statistics or the shuffle has been contaminated by m or draw -- which must be
    #     diagnosed before reading the table (I previously mistook arm-to-arm spread for cosine drift).
    for k in config.ks:
        cosine_hits = [r.metrics["hit"] for r in records if r.arm == "cosine" and r.k == k]
        cosine_r1 = [r.metrics["R@1"] for r in records if r.arm == "cosine" and r.k == k]
        if cosine_hits:
            spread = max(cosine_hits) - min(cosine_hits)
            if spread > 1e-12 or (max(cosine_r1) - min(cosine_r1)) > 1e-12:
                log.line(f"COSINE VARIED at K={k}: hit {min(cosine_hits):.6f}..{max(cosine_hits):.6f}, "
                         f"R@1 {min(cosine_r1):.6f}..{max(cosine_r1):.6f} -- pool/geometry contamination")
                raise SystemExit(f"cosine must be label-free and constant per K; it varied at K={k}")

    # CN: 每个臂**选到 epoch 0**（即恰好等于 cosine）的比例——这是结果，不能被当成无信息。
    # EN: per arm, the fraction of fits whose selected checkpoint is **epoch 0** (i.e. exactly cosine). That
    #     is a result, not a non-informative tie.
    for arm in ARM_ORDER:
        if arm == "cosine":
            continue
        arm_rows = [r for r in records if r.arm == arm and not r.excluded]
        if not arm_rows:
            continue
        epoch0 = sum(1 for r in arm_rows if float(r.extra.get("epochs", -1) or -1) == 0)
        log.line(f"epoch-0 selected: {arm:<10} {epoch0}/{len(arm_rows)} = {epoch0 / len(arm_rows):.2f}")

    rows = summarise(records, group=("k", "m", "arm"),
                     metrics=("hit", "R@1", "P@50", "in_sample_hit"))
    table = format_table(rows, ("k", "m", "arm", "n_draws", "hit_mean", "hit_std",
                                "R@1_mean", "P@50_mean", "in_sample_hit_mean"))
    log.line("\n" + table)
    log.line("# pre-registered verdict WITHIN a cell (vs cosine)")
    for k in config.ks:
        for m in config.ms:
            base_rows = [r for r in records if r.arm == "cosine" and r.k == k and r.m == m]
            base = [r.metrics["hit"] for r in base_rows]
            base_r1 = [r.metrics["R@1"] for r in base_rows]
            for arm in ARM_ORDER:
                if arm == "cosine":
                    continue
                arm_rows = [r for r in records if r.arm == arm and r.k == k and r.m == m and not r.excluded]
                if not base or len(arm_rows) != len(base):
                    continue
                h = [r.metrics["hit"] for r in arm_rows]
                v = verdict(h, base)
                r1 = [r.metrics["R@1"] for r in arm_rows]
                agree = np.sign(np.mean(r1) - np.mean(base_r1)) == np.sign(np.mean(h) - np.mean(base))
                log.line(f"  K={k:>3} m={m:>3} {arm:<10} {v:<5} R@1-agrees={agree}")
    write_records(config.run_dir / "records.json", records, provenance=PROVENANCE)
    log.line(f"wrote {config.run_dir / 'records.json'}; wall {time.perf_counter() - t_all:.0f}s")
    log.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
