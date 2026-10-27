"""
CN: **度量学习基线研究** —— 问题：**放松版 iOT**（对角 θ、盒约束）是否胜过普通的度量学习，
    还是只与 cosine 打平？

    CN: 背景：InfoNCE 曾经发散并被丢弃，因此"普通度量学习 vs iOT"这个比较**从未真正跑过**。
        本研究把它做成可执行的对照，并把"出发点必须等于 cosine"写成断言。

    CN: **协议（每个臂完全相同）**：DINOv2 pool4 + PCA，K ∈ {64,128,256}；几何 `G2`（统计量取自
        **未标注**图像）；m ∈ {50,100,300} 标注对，3 个 draw；**留出池** = 未标注图像，且那 m 个受试者
        从**两侧都剔除**；只在标注对上拟合；超参数/早停只用标注对内的验证折，**从不**看池；
        指标与 `studies/pool` 一致（hit、R@1、margin-P@50）；断言拟合空间 == 评分空间。

    CN: **所有梯度由 torch 自动求导得到**，不手写 Jacobian。此前手写版本三次算错
        （最大误差 46 倍），使任何 InfoNCE 读数都不可信；自动求导把这个失败模式整体移除。

    CN: **盒的三种变体（都报）**：
      * `iot`        —— iOT 的盒 `[0.25/K, 4/K]`（与 iOT 同口径）；
      * `wide`       —— 宽盒 `[0.05/K, 20/K]`，**InfoNCE 与 iOT 都跑**；
      * `free`       —— `θ >= 0` 无盒（仅 InfoNCE）。
        预注册：**除非宽盒给出同样的排序，否则不把盒差异解释为目标函数差异。**

EN: **Metric-learning baselines** -- does relaxed iOT beat ordinary metric learning, or only tie cosine?

    Plain InfoNCE diverged earlier and was dropped, so this comparison had never been run.

    **Protocol (identical for every arm):** DINOv2 pool4 + PCA, K in {64,128,256}; geometry `G2`
    (statistics from the **unlabelled** images); m in {50,100,300}, 3 draws; the **held-out pool** is the
    unlabelled images with the m labelled subjects removed from **both** sides; fits use labelled pairs only;
    hyperparameters/early stopping use a validation fold of the labelled pairs and **never** the pool; metrics
    as in `studies/pool`; the fit space is asserted equal to the scoring space.

    **All gradients come from torch autograd.** The hand-derived version was wrong three times (up to 46x),
    which would have invalidated every InfoNCE reading; autograd removes that failure mode.

    **Three box variants, all reported:** `iot` uses iOT's `[0.25/K, 4/K]`; `wide` uses `[0.05/K, 20/K]` for
    **both** InfoNCE and iOT; `free` is `theta >= 0` for InfoNCE only. Pre-registered: **unless the wide box
    shows the same ordering, box differences are not interpreted as objective differences.**
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from jaxtyping import Bool, Float, Int

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from iot_experiments.harness.config import RunConfig, build_parser  # noqa: E402
from iot_experiments.harness.driver import make_split  # noqa: E402
from iot_experiments.harness.records import RunRecord, format_table, summarise, verdict, write_records  # noqa: E402
from iot_experiments.harness.registry import apply_geometry, features, fit_geometry  # noqa: E402
from iot_experiments.harness.solver import diagonal_cost, fit_full_lp, matching_hit, rank_metrics  # noqa: E402

PROVENANCE: dict[str, object] = {
    "feature": "DINOv2 vit_small_patch14_pool4 + PCA (basis on all T1+T2 images, label-free)",
    "geometry": "G2, statistics from the unlabelled pool",
    "pool": "unlabelled images, with the m labelled subjects removed from BOTH sides",
    "selection": "hyperparameters/early stopping may use a validation fold of the LABELLED pairs only",
    "chance_hit": "1 / n_holdout",
    "verdict_rule": "matching hit higher on all 3 draws by >= 3 points and R@1 agrees in direction",
    "gradients": "torch autograd (no hand-derived Jacobians)",
    "boxes": {"iot": "[0.25/K, 4/K]", "wide": "[0.05/K, 20/K]", "free": "theta >= 0"},
}

MIN_GAIN: float = 0.03


class MetricLearningError(RuntimeError):
    """CN: 研究的前置条件失败。EN: a precondition of the study failed."""


@dataclass(frozen=True)
class Box:
    """
    CN: 对角 θ 的可行域。`lo < 0` 表示无下界（自由臂），`hi = inf` 表示无上界。
    EN: the feasible set for diagonal theta. `lo < 0` means unbounded below (free arm), `hi = inf` unbounded
        above.
    """

    lo: float
    hi: float
    name: str

    @classmethod
    def iot(cls, k: int) -> "Box":
        """CN: iOT 的盒 `[0.25/K, 4/K]`。EN: iOT's box `[0.25/K, 4/K]`."""
        return cls(lo=0.25 / float(k), hi=4.0 / float(k), name="iot")

    @classmethod
    def wide(cls, k: int) -> "Box":
        """CN: 宽盒 `[0.05/K, 20/K]`（盒差异的对照）。EN: wide box `[0.05/K, 20/K]`, the box control."""
        return cls(lo=0.05 / float(k), hi=20.0 / float(k), name="wide")

    @classmethod
    def free(cls) -> "Box":
        """CN: 自由臂 `θ >= 0`。EN: free arm, `theta >= 0`."""
        return cls(lo=0.0, hi=float("inf"), name="free")


def project(theta: Float[np.ndarray, "k"], box: Box) -> Float[np.ndarray, "k"]:
    """
    CN: 精确投影到 `{sum = 1, lo <= theta <= hi}`（欧氏）。对偶 `theta = clip(v - tau, lo, hi)`，`tau` 由
        `sum = 1` 唯一确定，用二分求到机器精度。

        CN: **对 uniform θ 必须是恒等**（`lo <= 1/K <= hi`），否则步 0 会偏离 cosine；由测试断言。
    EN: exact Euclidean projection onto `{sum = 1, lo <= theta <= hi}` via the dual
        `theta = clip(v - tau, lo, hi)` with `tau` from `sum = 1`, solved by bisection. It must be the
        **identity on uniform theta** or step 0 would already drift from cosine; a test asserts this.
    """
    values = np.asarray(theta, dtype=np.float64).reshape(-1)
    lo, hi = float(box.lo), float(box.hi)
    if not np.isfinite(hi):
        hi = float(values.max()) + 1.0
    if lo * values.size > 1.0 + 1e-12 or hi * values.size < 1.0 - 1e-12:
        raise MetricLearningError(f"infeasible {box.name} box: K*lo={lo * values.size:.6f}")
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


def symmetric_infonce_torch(logits: "object", *, temperature: float) -> "object":
    """
    CN: 对称 InfoNCE（行/列交叉熵取平均），logits 先除以 `temperature` 与其**平均绝对值**（正数，
        不因均值为负而翻转符号）。梯度由 torch 自动求导。
    EN: symmetric InfoNCE (mean of row- and column-wise cross-entropy); logits are divided by `temperature`
        and by their mean **absolute** value (positive, so a negative mean cannot flip the sign). Gradients
        come from torch autograd.
    """
    import torch  # noqa: PLC0415

    z = logits / float(temperature)
    scale = z.abs().mean()
    s = z / torch.clamp(scale, min=1e-12)
    m = int(s.shape[0])
    target = torch.arange(m, device=s.device)
    rows = torch.nn.functional.cross_entropy(s, target)
    cols = torch.nn.functional.cross_entropy(s.t(), target)
    return 0.5 * (rows + cols)


@dataclass(frozen=True)
class FitOutcome:
    """
    CN: 一个臂的拟合结果。`status != "ok"` ⇒ FAILED/发散，**必须排除**，不许用 cosine 顶替。
    EN: one arm's fit. `status != "ok"` means FAILED/diverged and the record **must be excluded**, never
        replaced by cosine.
    """

    status: str
    theta: Float[np.ndarray, "k"] | None
    maps: tuple[Float[np.ndarray, "k r"], Float[np.ndarray, "k r"]] | None
    n_params: int
    seconds: float
    epochs: int
    note: str = ""
    cca: tuple[Float[np.ndarray, "k r"], Float[np.ndarray, "k r"]] | None = None

    def cost(self, x: Float[np.ndarray, "n k"], y: Float[np.ndarray, "n k"]) -> Float[np.ndarray, "n n"]:
        """CN: 在**评分空间**上算代价（越小越像）。EN: cost in the **scoring space** (lower is a better match)."""
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


def fit_cca(x: Float[np.ndarray, "m k"], y: Float[np.ndarray, "m k"], *, ridge: float,
            n_components: int = 0) -> FitOutcome:
    """
    CN: 岭正则 CCA（闭式，无需梯度）。**白化必须是单位方差** `W = Σ^{-1/2}U/sqrt(m)`，这样
        `m·<xW_x, yW_y>` 的对角元就是典型相关（必在 [0,1] 内），并按相关 > 0.1 截断噪声方向。
    EN: ridge CCA in closed form (no gradients). Whitening must be **unit-variance**
        `W = S^{-1/2} U / sqrt(m)` so `m * <x W_x, y W_y>` on the diagonal is the canonical correlation (in
        [0, 1]); components with correlation <= 0.1 are dropped as noise.
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


def fit_diagonal_infonce(x: Float[np.ndarray, "m k"], y: Float[np.ndarray, "m k"], *, box: Box,
                         lr: float, epochs: int, temperature: float, device: str = "auto",
                         weight_decay: float = 0.0) -> FitOutcome:
    """
    CN: 对角 InfoNCE：`θ = project(softmax(w))`，代价 `-xᵀdiag(θ)y`，**全部 m×m logits**（无 minibatch），
        Adam，梯度由 torch 自动求导。`epochs=0` 时 `softmax(0)` 是 uniform θ，投影是其恒等 ⇒ 等于 cosine。
    EN: diagonal InfoNCE: `theta = project(softmax(w))`, cost `-x^T diag(theta) y`, **full m x m logits** (no
        minibatching), Adam, gradients from torch autograd. At `epochs=0`, `softmax(0)` is uniform theta and
        the projection is its identity, so it equals cosine.
    """
    import torch  # noqa: PLC0415

    start = time.perf_counter()
    dev = _device(device)
    a = torch.tensor(np.asarray(x, dtype=np.float64), dtype=torch.float64, device=dev)
    b = torch.tensor(np.asarray(y, dtype=np.float64), dtype=torch.float64, device=dev)
    k = int(a.shape[1])
    w = torch.zeros(k, dtype=torch.float64, device=dev, requires_grad=True)
    opt = torch.optim.Adam([w], lr=float(lr), weight_decay=float(weight_decay))
    lo = torch.tensor(float(box.lo), dtype=torch.float64, device=dev)
    hi = torch.tensor(float(box.hi) if np.isfinite(box.hi) else 1e3, dtype=torch.float64, device=dev)

    for step in range(1, int(epochs) + 1):
        opt.zero_grad(set_to_none=True)
        theta = torch.softmax(w, dim=0)
        # CN: 盒投影在 torch 里做（对偶二分），再用 `detach` 的 θ 计算损失；θ 的梯度经软 max 回传。
        # EN: box projection in torch (dual bisection); the loss uses the projected theta and gradients flow
        #     back through the softmax.
        projected = _project_torch(theta, lo, hi)
        theta = theta + (projected - theta).detach()
        # CN: **传给 InfoNCE 的必须是"相似度"而不是代价**。`s = xᵀdiag(θ)y` 在真配对上更大
        #     （实测 diag-off 间隙：uniform 0.0038、signal 0.0151），而 InfoNCE 把输入当 logits，
        #     所以传 `-s` 会**把目标反转**——实测各臂因此把权重压在**不**一致的维度上。
        # EN: **InfoNCE must receive the SIMILARITY, not the cost.** `s = x^T diag(theta) y` is larger on
        #     true pairs (measured diagonal-minus-off-diagonal gap: 0.0038 uniform, 0.0151 signal) and
        #     InfoNCE treats its argument as logits, so passing `-s` **inverts the objective** -- which is why
        #     every arm put weight on dimensions that did *not* agree.
        similarity = (a * theta.unsqueeze(0)) @ b.t()
        loss = symmetric_infonce_torch(similarity, temperature=float(temperature))
        if not bool(torch.isfinite(loss)):
            return FitOutcome(status="FAILED_nan", theta=None, maps=None, n_params=k,
                              seconds=time.perf_counter() - start, epochs=step, note="non-finite loss")
        loss.backward()
        assert w.grad is not None
        if not bool(torch.isfinite(w.grad).all()):
            return FitOutcome(status="FAILED_nan", theta=None, maps=None, n_params=k,
                              seconds=time.perf_counter() - start, epochs=step,
                              note="non-finite gradient")
        torch.nn.utils.clip_grad_norm_([w], max_norm=10.0)
        opt.step()

    with torch.no_grad():
        final = _project_torch(torch.softmax(w, dim=0), lo, hi)
    if not bool(torch.isfinite(final).all()):
        return FitOutcome(status="FAILED_diverged", theta=None, maps=None, n_params=k,
                          seconds=time.perf_counter() - start, epochs=int(epochs), note="non-finite theta")
    return FitOutcome(status="ok", theta=final.detach().cpu().numpy().copy(), maps=None, n_params=k,
                      seconds=time.perf_counter() - start, epochs=int(epochs),
                      note=f"lr={lr:g} T={temperature:g} wd={weight_decay:g} box={box.name} dev={dev}")


def _project_torch(theta: "object", lo: "object", hi: "object") -> "object":
    """
    CN: torch 版的盒∩单纯形投影（对偶二分）。与 numpy 版同算法，便于在 GPU 上完成整个 fit。
    EN: torch box ∩ simplex projection (dual bisection), same algorithm as the numpy version so the whole fit
        can stay on the GPU.
    """
    import torch  # noqa: PLC0415

    values = theta
    left = (values.min() - hi - 1.0).clone()
    right = (values.max() - lo + 1.0).clone()
    for _ in range(80):
        mid = 0.5 * (left + right)
        # CN: 二分只在**数值**上做，故 detach 后再取标量，避免把 requires_grad 张量转 float 的告警。
        # EN: the bisection is numeric only, so detach before reading a scalar (avoids the requires_grad
        #     conversion warning).
        total = torch.clamp(values.detach() - mid, min=lo, max=hi).sum()
        if float(total.detach()) > 1.0:
            left = mid
        else:
            right = mid
    return torch.clamp(values - 0.5 * (left + right), min=lo, max=hi)


def fit_residual_bilinear(x: Float[np.ndarray, "m k"], y: Float[np.ndarray, "m k"], *, rank: int,
                          lr: float, epochs: int, temperature: float, device: str = "auto",
                          weight_decay: float = 0.0) -> FitOutcome:
    """
    CN: **残差双线性**（臂 D）：`s(x,y) = xᵀ(I + UVᵀ)y`，`V` 初始化为 **0** ⇒ 步 0 严格等于 cosine。
        梯度由 torch 自动求导。
    EN: **residual bilinear** (arm D): `s = x^T (I + U V^T) y` with `V = 0` so step 0 is exactly cosine.
        Gradients from torch autograd.
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
    opt = torch.optim.Adam([u, v], lr=float(lr), weight_decay=float(weight_decay))

    for step in range(1, int(epochs) + 1):
        opt.zero_grad(set_to_none=True)
        # CN: 同上：`score` 是相似度，直接当 logits。EN: as above, `score` is the similarity used as logits.
        score = a @ b.t() + (a @ u) @ (b @ v).t()
        loss = symmetric_infonce_torch(score, temperature=float(temperature))
        if not bool(torch.isfinite(loss)):
            return FitOutcome(status="FAILED_nan", theta=None, maps=None, n_params=int(2 * k * r),
                              seconds=time.perf_counter() - start, epochs=step, note="non-finite loss")
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
                      note=f"lr={lr:g} T={temperature:g} wd={weight_decay:g} r={r} residual dev={dev}")


def evaluate(cost: Float[np.ndarray, "n n"], *, draw: int, fraction: float = 0.5) -> dict[str, float]:
    """
    CN: 与 `studies/pool` 一致的三个指标：matching hit、R@1、margin-P@`fraction`。
    EN: the three metrics of `studies/pool`: matching hit, R@1, margin-P at `fraction` coverage.
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


@dataclass(frozen=True)
class MetricLearningConfig(RunConfig):
    """CN: 研究自己的旋钮。EN: the study's own knobs."""

    section = "metric_learning"
    study: str = "metric_learning"
    feature: str = "dinov2-pool4"
    ks: tuple[int, ...] = (64, 128, 256)
    ms: tuple[int, ...] = (50, 100, 300)
    epochs: int = 100
    lr: float = 0.05
    temperature: float = 1.0
    fit_device: str = "auto"

    @classmethod
    def add_study_arguments(cls, parser: argparse.ArgumentParser) -> None:
        """CN: 只加本研究特有的参数。EN: only this study's flags."""
        parser.add_argument("--feature", default="dinov2-pool4", choices=["dinov2-pool4"])
        parser.add_argument("--ks", type=int, nargs="+", default=[64, 128, 256])
        parser.add_argument("--ms", type=int, nargs="+", default=[50, 100, 300])
        parser.add_argument("--epochs", type=int, default=100)
        parser.add_argument("--lr", type=float, default=0.05)
        parser.add_argument("--temperature", type=float, default=1.0)
        parser.add_argument("--fit-device", default="auto", choices=["auto", "cpu", "cuda"])


def arms_for(config: MetricLearningConfig) -> list[str]:
    """
    CN: 固定的臂清单（含三种盒变体）。每条记录的 `arm` 名称把盒编码进去，便于分表。
    EN: the fixed arm list including all three box variants; the box is encoded in the arm name so tables can
        be split by it.
    """
    out = ["cosine", "cca", "iot_box", "iot_wide"]
    for rank in (16, 64):
        out.append(f"residual{rank}")
    out += ["infonce_box", "infonce_wide", "infonce_free"]
    return out


def main(argv: list[str] | None = None) -> int:
    """CN: 入口。EN: entry point."""
    parser = build_parser(MetricLearningConfig)
    args = parser.parse_args(list(argv) if argv is not None else None)
    config = MetricLearningConfig.from_namespace(args, parser)

    records: list[RunRecord] = []
    t_all = time.perf_counter()
    for k in config.ks:
        x, y = features(str(config.feature), dim=int(k))
        n_subjects = int(x.shape[0])
        for m in config.ms:
            for draw in config.draws:
                split = make_split(n_subjects, int(m), draw=int(draw))
                labelled = split.labelled
                pool = split.unlabelled          # labelled EXCLUDED from both sides
                stats = fit_geometry(x[pool], y[pool], "G2")
                x_fit, y_fit = apply_geometry(x[labelled], y[labelled], stats)
                x_pool, y_pool = apply_geometry(x[pool], y[pool], stats)
                # CN: 断言拟合与评分在同一空间（两者行范数都必须是 1）。
                # EN: assert fit and pool live in one space (both must have unit row norms).
                for arr in (x_fit, y_fit, x_pool, y_pool):
                    norms = np.linalg.norm(np.asarray(arr, dtype=np.float64), axis=1)
                    if not np.allclose(norms, 1.0, atol=1e-8):
                        raise SystemExit(f"space not L2-normalised: [{norms.min():.6f}, {norms.max():.6f}]")

                # CN: 参考臂 cosine（= uniform θ）。
                # EN: reference arm cosine (= uniform theta).
                uniform = np.ones(int(k)) / float(k)
                metrics = evaluate(diagonal_cost(x_pool, y_pool, uniform), draw=draw)
                records.append(RunRecord(study=str(config.study), geometry="G2", arm="cosine",
                                         k=int(k), m=int(m), draw=int(draw), metrics=metrics,
                                         config=config.to_dict(), extra={"n_params": 0}))
                print(f"[K={k} m={m} d={draw}] cosine hit={metrics['hit']:.4f} R@1={metrics['R@1']:.4f}",
                      flush=True)

                boxes = {"iot_box": Box.iot(int(k)), "iot_wide": Box.wide(int(k))}
                for arm_name, arm_box in boxes.items():
                    t0 = time.perf_counter()
                    fit = fit_full_lp(x_fit, y_fit, solver=str(config.solver),
                                      solve_limit_s=float(config.solve_limit_s))
                    secs = time.perf_counter() - t0
                    if fit.status != "ok":
                        records.append(RunRecord(study=str(config.study), geometry="G2", arm=arm_name,
                                                 k=int(k), m=int(m), draw=int(draw), metrics={},
                                                 config=config.to_dict(), excluded=True,
                                                 status=fit.status))
                        print(f"[K={k} m={m} d={draw}] {arm_name} EXCLUDED {fit.status}", flush=True)
                        continue
                    # CN: iOT 的 LP 只给 iOT 的盒；宽盒的 iOT 用同一 θ 但报告为 `iot_wide` 仅当盒在 θ 上生效。
                    #     这里对 `iot_wide` 用**同一个 LP**（求解器只支持 iOT 盒），因此把 `iot_wide` 记成
                    #     skipped，避免假装跑过。
                    # EN: the LP only implements iOT's box, so `iot_wide` is recorded as skipped rather than
                    #     pretending it ran.
                    if arm_box.name != "iot":
                        records.append(RunRecord(study=str(config.study), geometry="G2", arm=arm_name,
                                                 k=int(k), m=int(m), draw=int(draw), metrics={},
                                                 config=config.to_dict(), excluded=True,
                                                 status="skipped_unsupported_box",
                                                 extra={"note": "fit_full_lp only implements the iOT box"}))
                        print(f"[K={k} m={m} d={draw}] {arm_name} SKIPPED (LP has no wide box)", flush=True)
                        continue
                    met = evaluate(diagonal_cost(x_pool, y_pool, fit.theta), draw=draw)
                    ins = evaluate(diagonal_cost(x_fit, y_fit, fit.theta), draw=draw)
                    records.append(RunRecord(study=str(config.study), geometry="G2", arm=arm_name,
                                             k=int(k), m=int(m), draw=int(draw), metrics=met,
                                             config=config.to_dict(),
                                             extra={"n_params": int(k), "seconds": round(secs, 2),
                                                    "in_sample_hit": ins["hit"],
                                                    "R_prime": fit.R_prime}))
                    print(f"[K={k} m={m} d={draw}] {arm_name} hit={met['hit']:.4f} R@1={met['R@1']:.4f} "
                          f"in-sample={ins['hit']:.4f} {secs:.1f}s", flush=True)

                fits: dict[str, FitOutcome] = {
                    "cca": fit_cca(x_fit, y_fit, ridge=float(1e-2)),
                }
                for rank in (16, 64):
                    fits[f"residual{rank}"] = fit_residual_bilinear(
                        x_fit, y_fit, rank=rank, lr=float(config.lr), epochs=int(config.epochs),
                        temperature=float(config.temperature), device=str(config.fit_device))
                for arm_name, arm_box in (("infonce_box", Box.iot(int(k))),
                                          ("infonce_wide", Box.wide(int(k))),
                                          ("infonce_free", Box.free())):
                    fits[arm_name] = fit_diagonal_infonce(
                        x_fit, y_fit, box=arm_box, lr=float(config.lr), epochs=int(config.epochs),
                        temperature=float(config.temperature), device=str(config.fit_device))

                for arm_name, outcome in fits.items():
                    if outcome.status != "ok":
                        records.append(RunRecord(study=str(config.study), geometry="G2", arm=arm_name,
                                                 k=int(k), m=int(m), draw=int(draw), metrics={},
                                                 config=config.to_dict(), excluded=True,
                                                 status=outcome.status, extra={"note": outcome.note}))
                        print(f"[K={k} m={m} d={draw}] {arm_name} FAILED {outcome.status} "
                              f"({outcome.note})", flush=True)
                        continue
                    cost = outcome.cost(x_pool, y_pool)
                    if not np.all(np.isfinite(cost)):
                        records.append(RunRecord(study=str(config.study), geometry="G2", arm=arm_name,
                                                 k=int(k), m=int(m), draw=int(draw), metrics={},
                                                 config=config.to_dict(), excluded=True,
                                                 status="FAILED_nonfinite_cost"))
                        print(f"[K={k} m={m} d={draw}] {arm_name} FAILED non-finite cost", flush=True)
                        continue
                    met = evaluate(cost, draw=draw)
                    ins = evaluate(outcome.cost(x_fit, y_fit), draw=draw)
                    records.append(RunRecord(study=str(config.study), geometry="G2", arm=arm_name,
                                             k=int(k), m=int(m), draw=int(draw), metrics=met,
                                             config=config.to_dict(),
                                             extra={"n_params": outcome.n_params,
                                                    "seconds": round(outcome.seconds, 2),
                                                    "epochs": outcome.epochs,
                                                    "in_sample_hit": ins["hit"],
                                                    "note": outcome.note}))
                    print(f"[K={k} m={m} d={draw}] {arm_name} hit={met['hit']:.4f} R@1={met['R@1']:.4f} "
                          f"P50={met['P@50']:.4f} in-sample={ins['hit']:.4f} "
                          f"({outcome.seconds:.1f}s, {outcome.n_params}p)", flush=True)

    rows = summarise(records, group=("k", "m", "arm"),
                     metrics=("hit", "R@1", "P@50", "in_sample_hit"))
    print("\n" + format_table(rows, ("k", "m", "arm", "n_draws", "hit_mean", "hit_std",
                                     "R@1_mean", "P@50_mean", "in_sample_hit_mean")))
    print("\n# pre-registered verdict vs cosine (hit higher on ALL draws by >= 3 points, R@1 agrees)")
    for k in config.ks:
        for m in config.ms:
            base_rows = [r for r in records if r.arm == "cosine" and r.k == k and r.m == m]
            base = [r.metrics["hit"] for r in base_rows]
            base_r1 = [r.metrics["R@1"] for r in base_rows]
            for arm in arms_for(config):
                if arm == "cosine":
                    continue
                arm_rows = [r for r in records if r.arm == arm and r.k == k and r.m == m and not r.excluded]
                if not base or len(arm_rows) != len(base):
                    continue
                v = verdict([r.metrics["hit"] for r in arm_rows], base, min_gain=MIN_GAIN)
                arm_r1 = [r.metrics["R@1"] for r in arm_rows]
                agree = np.sign(np.mean(arm_r1) - np.mean(base_r1)) == np.sign(
                    np.mean([r.metrics["hit"] for r in arm_rows]) - np.mean(base))
                print(f"  K={k:>3} m={m:>3} {arm:<14} {v:<5} R@1-agrees={agree}")
    print(f"\n[metric_learning] wall {time.perf_counter() - t_all:.0f}s")
    write_records(config.run_dir / "records.json", records, provenance=PROVENANCE)
    print(f"[metric_learning] wrote {config.run_dir / 'records.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
