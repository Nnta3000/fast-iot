"""
CN: **两半协议（跨受试者配对）** —— 问题：当**每个配对都是不同病人**时，各方法还能否恢复正确配对？

    CN: 为什么需要它：同受试者的 T1/T2 切片是**配准过的**，像素几乎相同，因此"隐藏配对"的基准
        对 NMI 这类像素方法已经**饱和**（1.000），不能再用来比较方法优劣。两半协议消除了这条捷径：

          * 受试者随机分成两个**不相交**的一半 A、B；
          * 候选池的 T1 只来自 A，T2 只来自 B —— 于是**每个候选对的两个人也不同**；
          * 每幅 T1 的**真实 T2**（同一受试者）**不进入池**，只用于评分。

    CN: 四个臂（同一池、同一打分）：
        | 臂 | 代价 |
        |---|---|
        | `random` | 均匀随机置换（**下界**） |
        | `nmi` | 全分辨率像素互信息 + Hungarian |
        | `cosine_ot` | 白化 DINOv2 余弦 + EMD |
        | `learned` | G2 内学到的 θ（**拟合与应用同一空间**）|

    CN: 打分：对每幅 T1，取该臂选中的 T2 与这幅 T1 的**真实 T2**比较，报 SSIM 与 PSNR。
        1.0 / ∞ 意味着"选中的就是真实伙伴"；随机的 SSIM/PSNR 给出下界。PSNR 用 `MAX_I = 1.0`。

EN: **Two-halves protocol (cross-subject pairing)** -- when **every match is a different patient**, can the
    methods still recover the right partner?

    Why it is needed: same-subject T1/T2 slices are **registered**, so their pixels are nearly identical and
    a hidden-partner benchmark is **saturated** for pixel methods such as NMI (1.000). It therefore cannot
    compare methods. The two-halves protocol removes that shortcut: subjects are split at random into two
    **disjoint** halves A and B; the pool has T1 from A only and T2 from B only, so **every candidate pair is
    two different people**; and each T1's **true T2** (the same subject) **never enters the pool** and is
    used for scoring only.

    Arms (same pool, same scoring): `random` (uniform permutation, the **floor**), `nmi` (full-resolution
    pixel MI + Hungarian), `cosine_ot` (whitened DINOv2 cosine + EMD), `learned` (theta fitted and applied
    in the same G2 space).

    Scoring: for each T1, compare the T2 that the arm chose with that T1's **true T2** using SSIM and PSNR,
    so 1.0 / infinity means "it picked the real partner" and the random arm gives the floor.
    PSNR uses `MAX_I = 1.0` (pixels are normalised to [0, 1]).

用法 / EN: usage
    PYTHONPATH=/home/agent/Documents/dpm-inverse-ot python -u -m iot_experiments.studies.real.t1t2.nmi.two_halves
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from jaxtyping import Float, Int

_REPO_ROOT = Path(__file__).resolve().parents[4]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from iot_experiments.harness.config import RunConfig, build_parser  # noqa: E402
from iot_experiments.harness.records import RunRecord, format_table, summarise, write_records  # noqa: E402
from iot_experiments.harness.registry import (  # noqa: E402
    GeometryResult,
    apply_geometry,
    features,
    fit_geometry,
    geometry,
)
from iot_experiments.harness.solver import diagonal_cost, fit_full_lp  # noqa: E402
from iot_experiments.studies.real.t1t2.nmi.run import (  # noqa: E402
    emd_perm,
    hungarian,
    load_pixels,
    nmi_pixel_cost,
    whiten,
)

PROVENANCE: dict[str, object] = {
    "protocol": "two disjoint halves: pool T1 from half A, pool T2 from half B (every match is a different patient)",
    "true_t2": "the same subject's T2, withheld from the pool, used for scoring only",
    "metrics": "SSIM and PSNR (MAX_I = 1.0) between the chosen T2 and the true T2",
    "nmi_estimator": "joint pixel histogram at full resolution, equal-frequency bins",
    "whitening_fit_set": "all T1+T2 images (label-free, no pairing information)",
    "note": "no arm sees the true pairing; the learned theta is fitted on m labelled same-subject pairs only",
}


class TwoHalvesError(RuntimeError):
    """CN: 两半协议的前置条件失败。EN: a precondition of the two-halves protocol failed."""


def ssim(a: Float[np.ndarray, "pixels"], b: Float[np.ndarray, "pixels"], *, data_range: float = 1.0
         ) -> float:
    """
    CN: 结构相似度（Wang et al. 2004），窗口统计用**整幅图**的均值/方差（常用的全局近似）。
    EN: structural similarity (Wang et al. 2004) with **global** window statistics (the usual whole-image
        approximation).
    """
    x: Float[np.ndarray, "pixels"] = np.asarray(a, dtype=np.float64)
    y: Float[np.ndarray, "pixels"] = np.asarray(b, dtype=np.float64)
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    mu_x, mu_y = float(x.mean()), float(y.mean())
    var_x, var_y = float(x.var()), float(y.var())
    cov = float(((x - mu_x) * (y - mu_y)).mean())
    return float(((2 * mu_x * mu_y + c1) * (2 * cov + c2)) /
                 ((mu_x ** 2 + mu_y ** 2 + c1) * (var_x + var_y + c2)))


def psnr(a: Float[np.ndarray, "pixels"], b: Float[np.ndarray, "pixels"], *, data_range: float = 1.0
         ) -> float:
    """
    CN: 峰值信噪比（dB），`MAX_I = data_range`；完全相同的图像返回 `inf`。
    EN: peak signal-to-noise ratio in dB with `MAX_I = data_range`; identical images return `inf`.
    """
    x: Float[np.ndarray, "pixels"] = np.asarray(a, dtype=np.float64)
    y: Float[np.ndarray, "pixels"] = np.asarray(b, dtype=np.float64)
    mse = float(np.mean((x - y) ** 2))
    if mse <= 0.0:
        return float("inf")
    return float(10.0 * np.log10(data_range ** 2 / mse))


def score_pixels(chosen: Float[np.ndarray, "n pixels"],
                 reference: Float[np.ndarray, "n pixels"]) -> dict[str, float]:
    """
    CN: 逐行把 `chosen[i]` 与 `reference[i]` 比较，返回 SSIM/PSNR 均值，以及像素完全相同的比例
        （`exact_hit`，诊断用；真正的读数看 SSIM/PSNR）。
    EN: compare `chosen[i]` with `reference[i]` row by row and return mean SSIM/PSNR plus the fraction of
        pixel-identical rows (`exact_hit`, a diagnostic; SSIM/PSNR are the real read-out).
    """
    a: Float[np.ndarray, "n pixels"] = np.asarray(chosen, dtype=np.float64)
    b: Float[np.ndarray, "n pixels"] = np.asarray(reference, dtype=np.float64)
    if a.shape != b.shape:
        raise TwoHalvesError(f"shape mismatch: chosen {a.shape} vs reference {b.shape}")
    n: int = int(a.shape[0])
    ssims: list[float] = [ssim(a[i], b[i]) for i in range(n)]
    psnrs: list[float] = [psnr(a[i], b[i]) for i in range(n)]
    finite: list[float] = [p for p in psnrs if np.isfinite(p)]
    identical: int = len(psnrs) - len(finite)
    return {"ssim": float(np.mean(ssims)),
            "psnr": float(np.mean(finite)) if finite else float("inf"),
            "exact_hit": float(identical) / float(n)}


@dataclass(frozen=True)
class TwoHalvesConfig(RunConfig):
    """CN: 两半协议自己的旋钮。EN: the two-halves protocol's own knobs."""

    section = "two_halves"
    study: str = "two_halves"
    ks: tuple[int, ...] = (256,)
    ms: tuple[int, ...] = (300,)
    bins: int = 64
    pixel_size: int = 192
    fitted_geometry: str = "G2"
    arms: tuple[str, ...] = ("random", "nmi", "cosine_ot", "learned")

    @classmethod
    def add_study_arguments(cls, parser: argparse.ArgumentParser) -> None:
        """CN: 只加本研究特有的参数。EN: only this study's flags."""
        parser.add_argument("--ks", type=int, nargs="+", default=[256])
        parser.add_argument("--ms", type=int, nargs="+", default=[300],
                            help="labelled same-subject pairs m used to fit the learned arm")
        parser.add_argument("--bins", type=int, default=64, help="equal-frequency bins for the NMI histogram")
        parser.add_argument("--pixel-size", type=int, default=192, help="image side length")
        parser.add_argument("--fitted-geometry", default="G2", choices=["G0", "G1", "G2", "G3"])
        parser.add_argument("--arms", nargs="+", default=["random", "nmi", "cosine_ot", "learned"],
                            choices=["random", "nmi", "cosine_ot", "learned"])


def main(argv: list[str] | None = None) -> int:
    """CN: 入口。EN: entry point."""
    parser = build_parser(TwoHalvesConfig)
    args = parser.parse_args(list(argv) if argv is not None else None)
    config = TwoHalvesConfig.from_namespace(args, parser)

    records: list[RunRecord] = []
    # CN: 评分需要真实 T2 的**全分辨率**像素，故统一加载一次（每幅约 37k 像素 × 1000 幅，~300 MB）。
    # EN: scoring needs the **full-resolution** true-T2 pixels, so they are loaded once (~300 MB total).
    px1, px2 = load_pixels(size=int(config.pixel_size))

    for k in config.ks:
        x, y = features("dinov2-pool4", dim=int(k))
        n_subjects: int = int(x.shape[0])
        half: int = n_subjects // 2
        xw, yw = whiten(x, y, dim=int(k))

        def unit(matrix: Float[np.ndarray, "subjects K"]) -> Float[np.ndarray, "subjects K"]:
            norms = np.linalg.norm(matrix, axis=1, keepdims=True)
            return matrix / np.where(norms > 0.0, norms, 1.0)

        for draw in config.draws:
            rng = np.random.default_rng(int(draw))
            order: Int[np.ndarray, "subjects"] = rng.permutation(n_subjects)
            half_a: Int[np.ndarray, "half"] = order[:half]
            half_b: Int[np.ndarray, "half"] = order[half:]

            # CN: 池：T1 取 A（打乱），T2 取 B（打乱）。每幅 T1 的真实 T2 是**同一受试者**的 T2，
            #     不在池里，只用于评分；A 与 B 不相交 ⇒ 池内每一对都是不同病人。
            # EN: pool T1 from A (shuffled), pool T2 from B (shuffled). A T1's true T2 is the **same
            #     subject's** T2, absent from the pool and used for scoring only; A and B are disjoint, so
            #     every pool pair is two different patients.
            pool_t1: Int[np.ndarray, "n"] = rng.permutation(half_a)
            pool_t2: Int[np.ndarray, "n"] = rng.permutation(half_b)
            truth: Float[np.ndarray, "n pixels"] = px2[pool_t1]

            if "random" in config.arms:
                # CN: 均匀随机置换 = 下界；独立种子，避免与其它臂共用同一置换。
                # EN: a uniform random permutation is the floor; a separate seed keeps it independent.
                perm_random: Int[np.ndarray, "n"] = np.random.default_rng(9000 + int(draw)).permutation(half)
                metrics = score_pixels(px2[pool_t2[perm_random]], truth)
                records.append(RunRecord(study=str(config.study), geometry=f"pixels{config.pixel_size}",
                                         arm="random", k=int(k), m=0, draw=int(draw), metrics=metrics,
                                         config=config.to_dict()))
                print(f"  K={k} d={draw} random     ssim={metrics['ssim']:.4f} psnr={metrics['psnr']:.2f} "
                      f"exact={metrics['exact_hit']:.4f}", flush=True)

            if "nmi" in config.arms:
                # CN: NMI 只看得到池内像素（T1 来自 A、T2 来自 B），看不到真实 T2。
                # EN: NMI sees only the pool pixels (T1 from A, T2 from B) and never the true T2.
                cost = nmi_pixel_cost(px1[pool_t1], px2[pool_t2], bins=int(config.bins))
                perm_nmi: Int[np.ndarray, "n"] = hungarian(cost)
                metrics = score_pixels(px2[pool_t2[perm_nmi]], truth)
                records.append(RunRecord(study=str(config.study), geometry=f"pixels{config.pixel_size}",
                                         arm="nmi", k=int(k), m=0, draw=int(draw), metrics=metrics,
                                         config=config.to_dict()))
                print(f"  K={k} d={draw} nmi        ssim={metrics['ssim']:.4f} psnr={metrics['psnr']:.2f} "
                      f"exact={metrics['exact_hit']:.4f}", flush=True)

            if "cosine_ot" in config.arms:
                # CN: 白化余弦距离 + EMD；统计量在全部图像上拟合（无配对信息），再按池顺序切片。
                # EN: whitened cosine distance + EMD; statistics fitted on all images (no pairing
                #     information), then sliced in pool order.
                xu_pool: Float[np.ndarray, "n K"] = unit(xw)[pool_t1]
                yu_pool: Float[np.ndarray, "n K"] = unit(yw)[pool_t2]
                dist = 1.0 - xu_pool @ yu_pool.T
                perm_cos: Int[np.ndarray, "n"] = emd_perm(dist * dist)
                metrics = score_pixels(px2[pool_t2[perm_cos]], truth)
                records.append(RunRecord(study=str(config.study), geometry="whiten", arm="cosine_ot",
                                         k=int(k), m=0, draw=int(draw), metrics=metrics,
                                         config=config.to_dict()))
                print(f"  K={k} d={draw} cosine_ot  ssim={metrics['ssim']:.4f} psnr={metrics['psnr']:.2f} "
                      f"exact={metrics['exact_hit']:.4f}", flush=True)

            if "learned" in config.arms:
                for m in config.ms:
                    # CN: 学到的臂在 m 个**同受试者**标注对上拟合（标签不进入池）。
                    # EN: the learned arm is fitted on m **same-subject** labelled pairs whose labels never
                    #     enter the pool.
                    lab: Int[np.ndarray, "m"] = rng.permutation(n_subjects)[: int(m)]
                    unl: Int[np.ndarray, "n"] = np.setdiff1d(np.arange(n_subjects), lab)
                    geo: GeometryResult = geometry(x, y, str(config.fitted_geometry),
                                                   labelled=lab, unlabelled=unl)
                    fit = fit_full_lp(geo.x_label, geo.y_label, solver=str(config.solver),
                                      solve_limit_s=float(config.solve_limit_s))
                    if fit.status != "ok":
                        records.append(RunRecord(study=str(config.study),
                                                 geometry=str(config.fitted_geometry), arm="learned",
                                                 k=int(k), m=int(m), draw=int(draw), metrics={},
                                                 config=config.to_dict(), excluded=True, status=fit.status))
                        print(f"  K={k} m={m} d={draw} learned EXCLUDED: {fit.status}", flush=True)
                        continue
                    # CN: θ 必须在它被拟合的几何里使用。`fit_geometry` 在未标注集上拟合统计量，
                    #     `apply_geometry` 用它变换**整个池**——同一空间，且标注集与池共享统计量。
                    # EN: theta must be applied in the geometry it was fitted in. `fit_geometry` fits the
                    #     statistics on the unlabelled set and `apply_geometry` transforms the **whole pool**
                    #     with them, so the labelled set and the pool share one space.
                    stats = fit_geometry(x[unl], y[unl], str(config.fitted_geometry))
                    x_pool, y_pool = apply_geometry(x[pool_t1], y[pool_t2], stats)
                    perm_lrn: Int[np.ndarray, "n"] = hungarian(
                        diagonal_cost(x_pool, y_pool, fit.theta))
                    metrics = score_pixels(px2[pool_t2[perm_lrn]], truth)
                    records.append(RunRecord(study=str(config.study),
                                             geometry=str(config.fitted_geometry), arm="learned",
                                             k=int(k), m=int(m), draw=int(draw), metrics=metrics,
                                             config=config.to_dict(),
                                             extra={"R_prime": fit.R_prime, "R_uniform": fit.R_uniform,
                                                    "theta_space": str(config.fitted_geometry)}))
                    print(f"  K={k} m={m} d={draw} learned    ssim={metrics['ssim']:.4f} "
                          f"psnr={metrics['psnr']:.2f} exact={metrics['exact_hit']:.4f}", flush=True)

    rows = summarise(records, group=("arm",), metrics=("ssim", "psnr", "exact_hit"))
    print("\n" + format_table(rows, ("arm", "n_draws", "ssim_mean", "ssim_std",
                                     "psnr_mean", "psnr_std", "exact_hit_mean")))
    write_records(config.run_dir / "records.json", records, provenance=PROVENANCE)
    print(f"\n[two_halves] wrote {config.run_dir / 'records.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
