"""
CN: **E2 精确配对的规模** —— 在给定 n 下比较 Hungarian / POT EMD / 我们的求解器，以及 NMI 代价矩阵的可行性。

    CN: 输入是**特征向量**（不需要图像）：`X (n, k)` 与 `Y (n, k)`，代价是白化余弦或任意 `n×n` 矩阵。
        因此 E2 可以在真实归档到位之前跑，用来确定"多大 n 还能跑、谁是瓶颈"。

    CN: 每个臂报：挂钟时间、峰值内存（进程级 RSS 增量）、可行性与最优性残差。

      * `hungarian` —— `scipy.optimize.linear_sum_assignment`，精确、O(n³)，单一置换；
      * `emd` —— POT 的网络单纯形，精确、单一置换（n=n 且均匀边际）；
      * `hello` —— 我们的求解器（HELLO-OT 的精确 OT，代价按需计算，不落地 `n×n`）；
      * `nmi_matrix` —— **只**构造 NMI 代价矩阵（O(n²·pixels)），用来确定它在哪个 n 上不再可行。

    CN: `hello` 与 `nmi_matrix` 需要 `FeatureMap`/像素输入；当它们不可用时**明确报 `skipped`**，
        绝不静默替换成别的求解器（见项目规则：没有静默回退）。

EN: **E2 scale of exact matching** -- compare Hungarian / POT EMD / our solver at a given n, and check
    whether the NMI cost matrix is still feasible.

    The input is **feature vectors** (no images), so E2 runs before the archive arrives and answers "how
    large an n can we still run, and where is the bottleneck".

    Per arm: wall time, peak memory (process RSS delta), and feasibility/optimality residuals. `hungarian`
    and `emd` return a single permutation; `hello` is our exact OT with the cost computed on the fly (no
    `n x n` materialisation); `nmi_matrix` **only** builds the NMI cost matrix, `O(n**2 * pixels)`, to find
    where it stops being feasible. When an arm's input is unavailable it reports **`skipped`**; it never
    silently substitutes another solver.
"""

from __future__ import annotations

import resource
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
from jaxtyping import Float

from .nmi_helpers import mutual_information_detail


class E2Error(RuntimeError):
    """CN: E2 的输入不合法。EN: invalid input to E2."""


@dataclass(frozen=True)
class ArmTiming:
    """
    CN: 一个臂在一个 n 上的读数。`status` ∈ {`ok`, `skipped`, `timeout`, `failed`}；
        非 `ok` 的行**不进入**任何汇总。
    EN: one arm's reading at one n. `status` is one of `ok`, `skipped`, `timeout`, `failed`; non-`ok` rows
        never enter an aggregate.
    """

    arm: str
    n: int
    seconds: float
    peak_rss_mb: float
    status: str
    objective: float | None = None
    residual: float | None = None
    note: str = ""

    def to_dict(self) -> dict[str, object]:
        """CN: JSON 可序列化。EN: JSON-serialisable."""
        return {"arm": self.arm, "n": self.n, "seconds": self.seconds,
                "peak_rss_mb": self.peak_rss_mb, "status": self.status,
                "objective": self.objective, "residual": self.residual, "note": self.note}


def _rss_mb() -> float:
    """
    CN: 当前进程的峰值 RSS（MB）。`ru_maxrss` 在 Linux 上是 KB。
    EN: the process's peak RSS in MB (`ru_maxrss` is in KB on Linux).
    """
    return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / 1024.0


def _timed(arm: str, n: int, fn: Callable[[], tuple[float | None, float | None]]) -> ArmTiming:
    """
    CN: 计时包装：记录挂钟与**峰值 RSS 增量**，并捕获超时/失败成为状态而不是异常。
    EN: timing wrapper: records wall time and the **peak-RSS delta**, turning timeout/failure into a status
        rather than an exception.
    """
    before: float = _rss_mb()
    start: float = time.perf_counter()
    try:
        objective, residual = fn()
    except TimeoutError as exc:
        return ArmTiming(arm=arm, n=n, seconds=time.perf_counter() - start,
                         peak_rss_mb=_rss_mb() - before, status="timeout", note=str(exc))
    except Exception as exc:  # noqa: BLE001 - CN: 失败必须变成状态，不能中断整轮扫描。EN: failure is a status.
        return ArmTiming(arm=arm, n=n, seconds=time.perf_counter() - start,
                         peak_rss_mb=_rss_mb() - before, status="failed",
                         note=f"{type(exc).__name__}: {exc}")
    elapsed: float = time.perf_counter() - start
    return ArmTiming(arm=arm, n=n, seconds=elapsed, peak_rss_mb=max(_rss_mb() - before, 0.0),
                     status="ok", objective=objective, residual=residual)


def cosine_cost(x: Float[np.ndarray, "n k"], y: Float[np.ndarray, "n k"]) -> Float[np.ndarray, "n n"]:
    """
    CN: 白化余弦距离 `1 - <x̂, ŷ>`（行已 L2 归一化）；这是 E2 的默认代价。
    EN: whitened cosine distance `1 - <x̂, ŷ>` for L2-normalised rows; E2's default cost.
    """
    a: Float[np.ndarray, "n k"] = np.asarray(x, dtype=np.float64)
    b: Float[np.ndarray, "n k"] = np.asarray(y, dtype=np.float64)
    # CN: keepdims 后形状是 (n, 1)，因此注解必须写成 "n 1"（真实形状，不是期望形状）。
    # EN: with keepdims the shape is (n, 1), so the annotation must say "n 1" -- the real shape, not the
    #     intended one.
    na: Float[np.ndarray, "n 1"] = np.linalg.norm(a, axis=1, keepdims=True)
    nb: Float[np.ndarray, "n 1"] = np.linalg.norm(b, axis=1, keepdims=True)
    return 1.0 - (a / np.where(na > 0, na, 1.0)) @ (b / np.where(nb > 0, nb, 1.0)).T


def run_hungarian(cost: Float[np.ndarray, "n n"]) -> tuple[float | None, float | None]:
    """
    CN: 精确指派，返回 `(目标值, 可行性残差)`。残差 = 置换代价之和与目标值之差（应为 0）。
    EN: exact assignment; returns `(objective, feasibility residual)`, where the residual is the sum of the
        permutation's costs minus the objective (zero for a true permutation).
    """
    from scipy.optimize import linear_sum_assignment  # noqa: PLC0415

    matrix: Float[np.ndarray, "n n"] = np.asarray(cost, dtype=np.float64)
    rows, cols = linear_sum_assignment(matrix)
    if not np.array_equal(rows, np.arange(rows.size)):
        raise E2Error("linear_sum_assignment did not return rows in order")
    total: float = float(matrix[rows, cols].sum())
    residual: float = abs(total - float(matrix[rows, cols].sum()))
    return total, residual


def run_emd(cost: Float[np.ndarray, "n n"], *, num_iter_max: int = 20_000_000
            ) -> tuple[float | None, float | None]:
    """
    CN: POT 网络单纯形；均匀边际下 n=n 的耦合是置换。残差 = 计划代价 - 目标值。
    EN: POT network simplex; with n = n and uniform marginals the coupling is a permutation. Residual is
        plan cost minus the objective.
    """
    import ot  # noqa: PLC0415

    matrix: Float[np.ndarray, "n n"] = np.asarray(cost, dtype=np.float64)
    n: int = int(matrix.shape[0])
    weights: Float[np.ndarray, "n"] = np.ones(n, dtype=np.float64) / float(n)
    plan: Float[np.ndarray, "n n"] = ot.emd(weights, weights, matrix, numItermax=int(num_iter_max))
    # CN: **单位必须与 Hungarian 一致**。POT 在 1/n 边际下的计划元素是 1/n，故 (plan*C).sum() 是
    #     **平均**代价；乘 n 才是与 Hungarian 可比的**总和**（实测 n=50 时相差 50 倍）。
    # EN: **match Hungarian's units.** POT's plan entries are 1/n under 1/n marginals, so (plan*C).sum() is a
    #     **mean** cost; multiply by n for the **total** that is comparable with Hungarian (measured 50x apart
    #     at n=50).
    objective: float = float((plan * matrix).sum()) * float(n)
    # CN: 可行性：行/列和必须回到均匀边际。
    # EN: feasibility: row and column sums must return to the uniform marginals.
    residual: float = max(float(np.abs(plan.sum(axis=1) - weights).max()),
                          float(np.abs(plan.sum(axis=0) - weights).max()))
    return objective, residual


def run_nmi_matrix(images_a: Float[np.ndarray, "n pixels"], images_b: Float[np.ndarray, "n pixels"],
                   *, bins: int = 32) -> tuple[float | None, float | None]:
    """
    CN: **只**构造 NMI 代价矩阵，用来测量 O(n²·pixels) 何时不可行。返回 `(总互信息, None)`。
        CN: 这是"矩阵构造"的时间，不含求解。
    EN: **only** build the NMI cost matrix, to measure when `O(n**2 * pixels)` stops being feasible.
        Returns `(total mutual information, None)`; this times matrix construction, not solving.
    """
    a: Float[np.ndarray, "n pixels"] = np.asarray(images_a, dtype=np.float64)
    b: Float[np.ndarray, "n pixels"] = np.asarray(images_b, dtype=np.float64)
    n: int = int(a.shape[0])
    total: float = 0.0
    for i in range(n):
        for j in range(n):
            total += mutual_information_detail(a[i], b[j], bins=int(bins)).nats
    return total, None


def samples_per_cell(pixels: int, *, bins: int = 32) -> float:
    """
    CN: 每对样本的**样本/格子**——NMI 代价的饥饿诊断量（低于 ~5 应视为饥饿）。
    EN: **samples per cell** for one pair -- the starvation diagnostic for an NMI cost (below about 5 is
        starved).
    """
    return float(pixels) / float(int(bins) ** 2)


def sweep(ns: Sequence[int], *, make_features: Callable[[int], tuple[Float[np.ndarray, "n k"],
                                                                    Float[np.ndarray, "n k"]]],
          arms: Sequence[str] = ("hungarian", "emd"),
          time_limit_s: float = 600.0,
          pixels: int | None = None,
          bins: int = 32) -> list[ArmTiming]:
    """
    CN: 在 `ns` 上扫描各臂。`time_limit_s` 是**单臂单 n** 的预算；超过则记 `timeout` 并跳过该臂的更大 n。
        `pixels` 给出时才会跑 `nmi_matrix`（需要像素级输入）。
    EN: sweep the arms over `ns`. `time_limit_s` is the budget for **one arm at one n**; on overrun the arm
        is recorded as `timeout` and skipped for larger n. `nmi_matrix` runs only when `pixels` is given.
    """
    out: list[ArmTiming] = []
    exhausted: set[str] = set()
    for n in ns:
        x, y = make_features(int(n))
        cost: Float[np.ndarray, "n n"] | None = None
        for arm in arms:
            if arm in exhausted:
                out.append(ArmTiming(arm=arm, n=int(n), seconds=0.0, peak_rss_mb=0.0,
                                     status="skipped", note="a smaller n already exceeded the budget"))
                continue
            if arm in ("hungarian", "emd"):
                if cost is None:
                    cost = cosine_cost(x, y)
                fn: Callable[[], tuple[float | None, float | None]] = (
                    (lambda c=cost: run_hungarian(c)) if arm == "hungarian"
                    else (lambda c=cost: run_emd(c)))
                timing = _timed(arm, int(n), fn)
            elif arm == "nmi_matrix":
                if pixels is None:
                    out.append(ArmTiming(arm="nmi_matrix", n=int(n), seconds=0.0, peak_rss_mb=0.0,
                                         status="skipped", note="no pixel input supplied"))
                    continue
                timing = _timed(arm, int(n),
                                lambda px=pixels: run_nmi_matrix(x[:, :int(px)], y[:, :int(px)],
                                                                 bins=int(bins)))
            else:
                out.append(ArmTiming(arm=arm, n=int(n), seconds=0.0, peak_rss_mb=0.0,
                                     status="skipped", note=f"arm {arm!r} not available in this harness"))
                continue
            if timing.seconds > float(time_limit_s) or timing.status == "timeout":
                timing = ArmTiming(arm=timing.arm, n=timing.n, seconds=timing.seconds,
                                   peak_rss_mb=timing.peak_rss_mb, status="timeout",
                                   note=f"exceeded {time_limit_s:.0f}s budget")
                exhausted.add(arm)
            out.append(timing)
    return out


def report_lines(rows: Sequence[ArmTiming], *, bins: int = 32, pixels: int | None = None
                 ) -> list[str]:
    """
    CN: 把读数排成文本行（直接进报告）。含每行的状态、时间、峰值内存与残差。
    EN: render the readings as text lines for the report, with status, time, peak memory and residual.
    """
    header = (f"{'n':>6}  {'arm':<12} {'status':<8} {'seconds':>9} {'peak MB':>9} "
              f"{'objective':>12} {'residual':>10}")
    lines = [header, "-" * len(header)]
    for row in sorted(rows, key=lambda r: (r.n, r.arm)):
        obj = "-" if row.objective is None else f"{row.objective:.6g}"
        res = "-" if row.residual is None else f"{row.residual:.2e}"
        lines.append(f"{row.n:>6}  {row.arm:<12} {row.status:<8} {row.seconds:>9.2f} "
                     f"{row.peak_rss_mb:>9.1f} {obj:>12} {res:>10}")
    if pixels is not None:
        lines.append(f"# NMI samples/cell at {pixels} pixels, {bins} bins: "
                     f"{samples_per_cell(pixels, bins=bins):.2f} "
                     f"(below ~5 is starved)")
    return lines


__all__ = [
    "ArmTiming",
    "E2Error",
    "cosine_cost",
    "report_lines",
    "run_emd",
    "run_hungarian",
    "run_nmi_matrix",
    "samples_per_cell",
    "sweep",
]
