"""
CN: **E1 配对度量** —— 在 SynthRAD 的切片上定义可测量的配对度量，并用 `discriminates` 判定哪些可用。

    CN: 对一对 (MR 切片 a, CT 切片 b) 定义：

      | 度量 | 定义 | 方向 |
      |---|---|---|
      | `mask_iou` | 两片身体掩膜的交并比 | 越大越像同一层 |
      | `z_gap` | `|a.z_normalised - b.z_normalised|` | **越小**越好 ⇒ 报 `1 - z_gap` |
      | `nmi` | 128×128、32 箱的像素互信息（nats） | 越大越像同一对 |
      | `ssim` | 与 a 的**真实 CT** 的结构相似度 | 越大越好（需要真值） |
      | `psnr` | 与真实 CT 的 PSNR（dB） | 越大越好（需要真值） |

    CN: **先测量再使用**：`ssim`/`psnr` 依赖真实配对，因此**不能**作为配对代价（那会泄漏真值），
        只作为**评估**指标；它们同样必须通过判别力检查，否则用它们评估也会得到噪声。

    CN: 检查的三组（同一批切片）：
        * `true`   —— 同一患者的对应切片（同 z 序号，注册对齐）；
        * `same`   —— **同中心**的另一患者的切片（最难对照，能暴露"只是同扫描仪"）；
        * `random` —— 任意另一患者的切片。

    CN: 预期（写下来再跑，避免事后解释）：`mask_iou` 与 `1-z_gap` 应该**通不过** `same` 对照——
        因为任何骨盆/脑切片的身体轮廓与层位都差不多，它们描述的是"这是个骨盆中间层"，不是"这是同一个患者"。
        若确实如此，E3 的门就必须用 `nmi`/`ssim` 这类**内容**度量，而不能用形状/层位。

EN: **E1 pairing metrics** -- define measurable pairing metrics on SynthRAD slices and use `discriminates`
    to decide which are usable.

    For an (MR slice a, CT slice b) pair: `mask_iou` (body-mask intersection-over-union), `z_gap`
    (|Δz_normalised|, reported as `1 - z_gap` so higher is better), `nmi` (128x128, 32-bin pixel mutual
    information), and `ssim`/`psnr` against **a's own true CT**. `ssim`/`psnr` need the ground truth, so they
    are **evaluation** metrics only -- never a pairing cost, which would leak the answer -- and they must pass
    the same discrimination check or they cannot evaluate anything.

    Three groups over one batch: `true` (same patient, registered), `same` (another patient **from the same
    centre**, the hard control), `random` (any other patient).

    Pre-registered expectation: `mask_iou` and `1 - z_gap` should **fail** the `same` control, because any
    pelvis/brain slice has a similar body contour and a similar mid-volume level; they describe "this is a
    mid pelvis slice", not "this is the same patient". If so, the E3 gate must rest on content metrics
    (`nmi`, `ssim`), not on shape or position.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from jaxtyping import Bool, Float

from .discriminates import CheckResult, Metric, discrimination_check

NMI_SIZE: int = 128
NMI_BINS: int = 32


class E1Error(RuntimeError):
    """CN: E1 度量的输入不合法。EN: invalid input to an E1 metric."""


def _square(image: Float[np.ndarray, "pixels"], *, size: int) -> Float[np.ndarray, "size size"]:
    """
    CN: 把展平切片变回方阵并按 `size` 缩放（双线性）。E1 的 NMI 固定 128×128，与 E2 的特征尺度分开。
    EN: unflatten a slice and resize to `size` (bilinear). E1's NMI is fixed at 128x128, independent of the
        feature scale used in E2.
    """
    from scipy.ndimage import zoom  # noqa: PLC0415

    side: int = int(round(float(np.sqrt(image.size))))
    if side * side != image.size:
        raise E1Error(f"slice length {image.size} is not a perfect square")
    arr: Float[np.ndarray, "side side"] = np.asarray(image, dtype=np.float64).reshape(side, side)
    if side == int(size):
        return arr
    factor: float = float(size) / float(side)
    out: Float[np.ndarray, "size size"] = zoom(arr, factor, order=1)
    if out.shape != (size, size):
        fixed = np.zeros((size, size), dtype=np.float64)
        h, w = min(size, out.shape[0]), min(size, out.shape[1])
        fixed[:h, :w] = out[:h, :w]
        out = fixed
    return out


def mask_iou(mask_a: Bool[np.ndarray, "pixels"], mask_b: Bool[np.ndarray, "pixels"]) -> float:
    """
    CN: 身体掩膜交并比；两者皆空时返回 1（都"没有身体"是一致的）。
    EN: body-mask IoU; both empty returns 1 (agreeing that there is no body).
    """
    a: Bool[np.ndarray, "pixels"] = np.asarray(mask_a, dtype=bool)
    b: Bool[np.ndarray, "pixels"] = np.asarray(mask_b, dtype=bool)
    if a.shape != b.shape:
        raise E1Error(f"mask shapes differ: {a.shape} vs {b.shape}")
    union: int = int(np.logical_or(a, b).sum())
    if union == 0:
        return 1.0
    return float(np.logical_and(a, b).sum()) / float(union)


def nmi(a: Float[np.ndarray, "pixels"], b: Float[np.ndarray, "pixels"], *,
        size: int = NMI_SIZE, bins: int = NMI_BINS) -> float:
    """
    CN: 两幅切片之间的互信息（nats），等频分箱。**每对只有 `size²` 个样本**，所以
        `size=128, bins=32` 给出 16 样本/格子（见 `NMI_PAIRING_REPORT` 的样本饥饿结论）。
    EN: mutual information (nats) between two slices with equal-frequency bins. **Each pair provides only
        `size**2` samples**, so `size=128, bins=32` gives 16 samples per cell (see the sample-starvation
        finding in `current/NMI_PAIRING_REPORT.md`).
    """
    from .nmi_helpers import mutual_information_pair  # noqa: PLC0415

    return mutual_information_pair(_square(a, size=size).ravel(), _square(b, size=size).ravel(),
                                   bins=int(bins))


def z_gap(target: Float[np.ndarray, "n"], candidate: Float[np.ndarray, "n"]) -> Float[np.ndarray, "n"]:
    """
    CN: `|Δz_normalised|`；返回"越好越大"的 `1 - gap`（已裁剪到 [0,1]）。
    EN: `|Δz_normalised|` returned as the higher-is-better form `1 - gap`, clipped to [0, 1].
    """
    diff: Float[np.ndarray, "n"] = np.abs(np.asarray(target, dtype=np.float64) -
                                         np.asarray(candidate, dtype=np.float64))
    return np.clip(1.0 - diff, 0.0, 1.0)


def ssim(a: Float[np.ndarray, "pixels"], b: Float[np.ndarray, "pixels"]) -> float:
    """CN: 全局 SSIM（窗口统计取整幅）。EN: global SSIM (whole-image window statistics)."""
    x: Float[np.ndarray, "pixels"] = np.asarray(a, dtype=np.float64)
    y: Float[np.ndarray, "pixels"] = np.asarray(b, dtype=np.float64)
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    mx, my = float(x.mean()), float(y.mean())
    vx, vy = float(x.var()), float(y.var())
    cov = float(((x - mx) * (y - my)).mean())
    return float(((2 * mx * my + c1) * (2 * cov + c2)) / ((mx ** 2 + my ** 2 + c1) * (vx + vy + c2)))


def psnr(a: Float[np.ndarray, "pixels"], b: Float[np.ndarray, "pixels"]) -> float:
    """CN: PSNR（dB，MAX_I=1）；完全相同返回 inf。EN: PSNR in dB with MAX_I=1; identical returns inf."""
    x: Float[np.ndarray, "pixels"] = np.asarray(a, dtype=np.float64)
    y: Float[np.ndarray, "pixels"] = np.asarray(b, dtype=np.float64)
    mse: float = float(np.mean((x - y) ** 2))
    if mse <= 0.0:
        return float("inf")
    return float(10.0 * np.log10(1.0 / mse))


@dataclass(frozen=True)
class SlicePool:
    """
    CN: E1 用的一批切片。所有数组按切片对齐；`patient`/`centre` 用于构造三组对照。
    EN: the batch of slices used by E1. All arrays are slice-aligned; `patient`/`centre` build the three
        groups.
    """

    mr: Float[np.ndarray, "slices pixels"]
    ct: Float[np.ndarray, "slices pixels"]
    mask: Bool[np.ndarray, "slices pixels"]
    z_normalised: Float[np.ndarray, "slices"]
    patient: Sequence[str]
    centre: Sequence[str]

    def __len__(self) -> int:
        """CN: 切片数。EN: number of slices."""
        return int(np.asarray(self.mr).shape[0])

    def __post_init__(self) -> None:
        """CN: 长度必须一致。EN: all fields must be slice-aligned."""
        n = int(np.asarray(self.mr).shape[0])
        if len(self.patient) != n or len(self.centre) != n:
            raise E1Error(f"metadata not aligned: {n} slices, {len(self.patient)} patients, "
                          f"{len(self.centre)} centres")


def pairing_metrics(*, with_truth: bool = True) -> dict[str, Metric]:
    """
    CN: 返回给 `discrimination_check` 用的度量字典。

        * **代价可用**（不碰真值）：`mask_iou`、`z`、`nmi`；
        * **仅评估**（需要真实 CT）：`ssim`、`psnr` —— 调用方**不能**把它们用作配对代价。

    EN: the metric dict handed to `discrimination_check`. **Cost-usable** (never touches the truth):
        `mask_iou`, `z`, `nmi`. **Evaluation-only** (needs the true CT): `ssim`, `psnr` -- callers must not
        use them as a pairing cost.
    """
    out: dict[str, Metric] = {
        "mask_iou": lambda a, b: mask_iou(np.asarray(a["mask"]) > 0.5, np.asarray(b["mask"]) > 0.5),
        "z": lambda a, b: float(z_gap(np.asarray([a["z"]]), np.asarray([b["z"]]))[0]),
        "nmi": lambda a, b: nmi(np.asarray(a["mr"]), np.asarray(b["ct"])),
    }
    if with_truth:
        out["ssim"] = lambda a, b: ssim(np.asarray(b["ct"]), np.asarray(a["truth"]))
        # CN: PSNR 对完全相同的图像是 inf；取有限部分，避免后续均值出现 inf - inf。
        # EN: PSNR is inf for identical images; cap it so later means cannot produce inf - inf.
        out["psnr"] = lambda a, b: min(psnr(np.asarray(b["ct"]), np.asarray(a["truth"])), 99.0)
    return out


def as_metric_rows(pool: SlicePool) -> tuple[list[dict[str, np.ndarray]], list[dict[str, np.ndarray]]]:
    """
    CN: 把 `SlicePool` 变成 `discrimination_check` 需要的两个对齐列表
        （`a` 侧带 `truth`，即该切片的真实 CT；`b` 侧是被选中的候选）。
    EN: turn a `SlicePool` into the two aligned lists `discrimination_check` expects; the `a` side carries
        `truth`, the slice's own true CT.
    """
    a_rows: list[dict[str, np.ndarray]] = []
    b_rows: list[dict[str, np.ndarray]] = []
    for i in range(len(pool)):
        a_rows.append({"mr": pool.mr[i], "mask": pool.mask[i], "z": np.asarray(pool.z_normalised[i]),
                       "truth": pool.ct[i]})
        b_rows.append({"ct": pool.ct[i], "mask": pool.mask[i], "z": np.asarray(pool.z_normalised[i])})
    return a_rows, b_rows


def run_e1(pool: SlicePool, *, draws: int = 3, seed: int = 0,
           threshold: float = 0.5) -> CheckResult:
    """
    CN: 跑 E1 的判别力检查。`same` 组自动限定在**同一中心**（需要 `pool.centre`）。
    EN: run E1's discrimination check; the `same` group is automatically restricted to the **same centre**
        via `pool.centre`.
    """
    centres: list[str] = list(pool.centre)
    codes: dict[str, int] = {c: i for i, c in enumerate(sorted(set(centres)))}
    centre_idx = np.asarray([codes[c] for c in centres], dtype=np.int64)
    a_rows, b_rows = as_metric_rows(pool)
    return discrimination_check(pairing_metrics(), a_rows, b_rows, centres=centre_idx,
                                draws=int(draws), seed=int(seed), threshold=float(threshold))


__all__ = [
    "E1Error",
    "NMI_BINS",
    "NMI_SIZE",
    "SlicePool",
    "as_metric_rows",
    "mask_iou",
    "nmi",
    "pairing_metrics",
    "psnr",
    "run_e1",
    "ssim",
    "z_gap",
]
