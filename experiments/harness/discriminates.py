"""
CN: **配对度量的判别力检查** —— 任何配对指标在**被使用之前**必须先通过这个检查。

    CN: 背景（本项目实测）：在 T1/T2 上，SSIM 区分"真配对"与"跨受试者配对"只有 **0.00 SD**
        （0.4973 vs 0.4975），而 SSIM 的 argmax 找回真实伙伴的比例等于随机。也就是说，那个指标
        连"该配对是否更好"都无法回答，用它做表格只会得到噪声。这个模块把该检查变成**代码**，
        使同类错误无法在下一个数据集上重演。

    CN: 检查口径（三分组，全部来自同一批图像）：
        * `true`   —— 真实配对（同受试者 / 同患者，注册对齐）；
        * `same`   —— **同中心**的跨受试者配对（控制中心/扫描仪差异，最难的对照）；
        * `random` —— 任意跨受试者配对（随机下界）。
        对每个指标报三组的 均值、标准差，以及**效应量**：`(mean_true - mean_control) / std_control`。
        只有 `same` 组的效应量超过阈值（默认 0.5 SD，中等效应）才算"有判别力"。

    CN: 为什么必须用 `same` 而不是只用 `random`：如果信号只来自"同一个中心/同一台扫描仪"，
        那么 `true` vs `random` 会显得很好，而方法实际上学的是**中心身份**而不是配对兼容性。
        本项目此前正是被这种捷径骗过（像素几乎相同 ⇒ 隐藏配对饱和）。

EN: **Discrimination check for pairing metrics** -- every candidate pairing metric must pass this before it
    is used.

    Why (measured here): on T1/T2, SSIM separates true pairs from cross-subject pairs by **0.00 SD**
    (0.4973 vs 0.4975), and SSIM's argmax recovers the true partner at chance. Such a metric cannot answer
    "is this pairing better", so tabulating it yields noise. This module turns that check into **code** so
    the same mistake cannot recur on the next dataset.

    Three groups over one batch of images:
    * `true`   -- the real pair (same subject / same patient, registered);
    * `same`   -- a cross-subject pair **from the same centre** (controls scanner/centre differences; the
                  hard control);
    * `random` -- any cross-subject pair (the random floor).

    For each metric report the mean, standard deviation and the **effect size**
    `(mean_true - mean_control) / std_control` for both controls. A metric is usable only if the `same`
    effect size clears a threshold (default 0.5 SD, a medium effect).

    Using `same` matters: if the signal is only "this is the same centre/scanner", `true` vs `random` looks
    impressive while the method is learning **centre identity** rather than pairing compatibility.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, TypeVar

import numpy as np
from jaxtyping import Int

# CN: 样本类型由调用方决定：可以是数组（特征向量），也可以是**字典**（E1 的切片同时需要
#     MR/CT/掩膜/z）。这里用 `TypeVar` 而不是 `Any`，使签名保留"两侧样本类型相同"这一信息。
# EN: the sample type is the caller's choice: an array (a feature vector) or a **dict** (E1 slices need
#     MR/CT/mask/z at once). A `TypeVar` is used instead of `Any` so the signature still records that both
#     sides share one sample type.
SampleT = TypeVar("SampleT")

# CN: 指标签名：`(a, b) -> float`，**越大越好**。
# EN: a metric maps a pair of samples `(a, b)` to a float; **higher is better**.
Metric = Callable[[Any, Any], float]


class DiscriminatesError(RuntimeError):
    """CN: 判别力检查的输入不合法。EN: invalid input to the discrimination check."""


@dataclass(frozen=True)
class MetricReport:
    """
    CN: 一个指标的判别力报告。`usable` 表示**两个对照**都通过阈值。
    EN: one metric's discriminations report. `usable` means **both** controls clear the threshold.
    """

    metric: str
    means: dict[str, float]
    stds: dict[str, float]
    effects: dict[str, float]
    usable: bool
    threshold: float


@dataclass(frozen=True)
class CheckResult:
    """CN: 多个指标的完整结果。EN: the full result over several metrics."""

    reports: Sequence[MetricReport]
    n_draws: int
    groups: Sequence[str] = field(default_factory=lambda: ("true", "same", "random"))

    def usable_metrics(self) -> list[str]:
        """CN: 通过检查的指标名。EN: names of the metrics that passed."""
        return [r.metric for r in self.reports if r.usable]

    def rejection_summary(self) -> list[str]:
        """CN: 未通过的原因（供报告直接引用）。EN: reasons for rejection, ready to quote in a report."""
        out: list[str] = []
        for rep in self.reports:
            if rep.usable:
                continue
            worst = min(rep.effects.items(), key=lambda kv: kv[1])
            out.append(f"{rep.metric}: effect vs {worst[0]} = {worst[1]:+.2f} SD "
                       f"< {rep.threshold:.2f} (true mean {rep.means['true']:.4f})")
        return out


def _effect(mean_true: float, mean_control: float, std_control: float) -> float:
    """CN: 效应量（以对照组标准差为单位）。标准差为 0 时退化为 ±inf/0。"""
    if not np.isfinite(std_control) or std_control <= 0.0:
        delta = mean_true - mean_control
        return float("inf") if delta > 0 else (float("-inf") if delta < 0 else 0.0)
    return float((mean_true - mean_control) / std_control)


def discrimination_check(
    metrics: dict[str, Metric],
    samples_a: Sequence[SampleT],
    samples_b: Sequence[SampleT],
    *,
    centres: Int[np.ndarray, "n"] | None = None,
    draws: int = 3,
    seed: int = 0,
    threshold: float = 0.5,
) -> CheckResult:
    """
    CN: 对每个指标跑三分组检查。

        * `samples_a[i]` 与 `samples_b[i]` 是**真实配对**（同受试者/同患者）；
        * `centres[i]` 是受试者 `i` 的中心；`same` 组取**同中心**的另一个受试者，
          `random` 组取任意另一个受试者；
        * 每组的**配对**（哪个下标配哪个）由 `seed` 与 `draw` 固定，两个指标看到**完全相同**的配对，
          因此指标之间的比较是配对的、公平的。

    CN: 返回每个指标的效应量；效应量按 `(true - control)/std(control)` 计算，越高越有判别力。

    EN: run the three-group check for each metric.
    Every metric sees **exactly the same pairings** (fixed by `seed` and `draw`), so metrics are compared
    fairly on matched pairs. Effect size is `(true - control) / std(control)`; higher is more discriminating.
    """
    if len(samples_a) != len(samples_b):
        raise DiscriminatesError(
            f"samples_a and samples_b must be aligned: {len(samples_a)} vs {len(samples_b)}")
    n: int = len(samples_a)
    if n < 4:
        raise DiscriminatesError(f"need at least 4 samples to form the controls, got {n}")
    if centres is not None and len(centres) != n:
        raise DiscriminatesError(f"centres must have length {n}, got {len(centres)}")

    reports: list[MetricReport] = []
    for name, metric in metrics.items():
        true_vals: list[float] = []
        same_vals: list[float] = []
        random_vals: list[float] = []
        for draw in range(int(draws)):
            rng = np.random.default_rng(int(seed) * 1000 + int(draw))
            # CN: 每行 i 取一个"别的受试者" j；`same` 组再要求同一中心。
            # EN: each row i takes another subject j; the `same` group additionally requires the same
            #     centre.
            for i in range(n):
                true_vals.append(float(metric(samples_a[i], samples_b[i])))
                candidates = np.array([j for j in range(n) if j != i], dtype=np.int64)
                j_any: int = int(rng.choice(candidates))
                random_vals.append(float(metric(samples_a[i], samples_b[j_any])))
                if centres is not None:
                    same_pool = np.array([j for j in candidates if centres[j] == centres[i]],
                                         dtype=np.int64)
                    if same_pool.size == 0:
                        continue
                    j_same: int = int(rng.choice(same_pool))
                    same_vals.append(float(metric(samples_a[i], samples_b[j_same])))

        # CN: 用 `nanmean`/`nanstd`，因为个别指标可能返回 inf/nan（例如 PSNR 的上限处理）。
        # EN: `nanmean`/`nanstd` because an individual metric may return inf/nan (e.g. a capped PSNR).
        arrays = {"true": np.asarray(true_vals, dtype=np.float64),
                  "random": np.asarray(random_vals, dtype=np.float64)}
        if same_vals:
            arrays["same"] = np.asarray(same_vals, dtype=np.float64)
        means = {k: float(np.nanmean(v)) for k, v in arrays.items()}
        stds = {k: float(np.nanstd(v)) for k, v in arrays.items()}
        effects = {k: _effect(means["true"], means[k], stds[k]) for k in arrays if k != "true"}
        usable: bool = bool(effects) and all(e >= float(threshold) for e in effects.values())
        reports.append(MetricReport(metric=str(name), means=means, stds=stds, effects=effects,
                                    usable=usable, threshold=float(threshold)))

    groups = ["true", "same", "random"] if centres is not None else ["true", "random"]
    return CheckResult(reports=reports, n_draws=int(draws), groups=groups)


def format_check(result: CheckResult) -> str:
    """
    CN: 把检查结果排成等宽表（直接进报告）。列：指标、真/同中心/随机的均值、两个效应量、判定。
    EN: render the result as a fixed-width table for the report: metric, the three means, both effect
        sizes and the verdict.
    """
    cols = ["metric", "mean_true", "mean_same", "mean_random", "eff_same", "eff_random", "usable"]

    def cell(rep: MetricReport, col: str) -> str:
        if col == "metric":
            return rep.metric
        if col == "usable":
            return "yes" if rep.usable else "NO"
        if col.startswith("mean_"):
            value = rep.means.get(col[5:])
            return "-" if value is None else f"{value:.4f}"
        value = rep.effects.get(col[4:])
        return "-" if value is None else f"{value:+.2f}"

    widths = [max(len(c), *(len(cell(r, c)) for r in result.reports)) for c in cols]
    lines = ["  ".join(c.rjust(w) for c, w in zip(cols, widths)),
             "  ".join("-" * w for w in widths)]
    for rep in result.reports:
        lines.append("  ".join(cell(rep, c).rjust(w) for c, w in zip(cols, widths)))
    return "\n".join(lines)


__all__ = [
    "CheckResult",
    "DiscriminatesError",
    "Metric",
    "MetricReport",
    "SampleT",
    "discrimination_check",
    "format_check",
]
