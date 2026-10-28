"""
CN: **廉价项 A/B/C** —— (A) 原始像素距离基线；(B) 中和"身体轮廓捷径"的评估；(C) 配对的双 CT 窗表。

    CN: (A) 最朴素的基线：**直接用像素**比较 MRI 与 CT，不学任何东西（均方误差、相关系数）。仍用精确指派。
    CN: (B) **中和轮廓捷径**：把每个 MRI 行的候选限制在**轮廓面积接近**的 CT 片段，使"靠轮廓配对"失效；
        再用**同一个限制**跑朴素像素与学到的特征。学习若真有内容，优势应当在去掉捷径后仍然存在。
    CN: (C) **配对的双窗表**：同一批患者、同一个池、同一个 draw，只换 CT 窗，给出 soft 与 bone 的逐臂数值。

EN: **Cheap items A/B/C** -- (A) raw-pixel baseline; (B) outline-shortcut neutralised; (C) paired two-window
    table.

    (A) The plainest baseline: compare MRI and CT **pixels directly** (MSE, correlation), learning nothing,
        still with an exact one-to-one assignment.
    (B) **Neutralise the outline shortcut**: restrict each MRI row's candidates to CT slices with a similar
        outline **area**, then run both the naive pixel baseline and the learned features under the **same**
        restriction. If learning encodes content, its advantage must survive the restriction.
    (C) Same patients, same pool, same draw -- only the CT window changes, giving per-arm soft and bone values.
"""

from __future__ import annotations

import argparse
import json
import resource
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
from iot_experiments.harness.dinov2 import encode_pool4, pca_apply, pca_fit  # noqa: E402
from iot_experiments.harness.records import RunLog, RunRecord, format_table, summarise, write_records  # noqa: E402
from iot_experiments.harness.registry import apply_geometry, fit_geometry  # noqa: E402
from iot_experiments.harness.synthrad import CT_WINDOW, IMAGE_SIZE, discover_patients  # noqa: E402
from iot_experiments.studies.real.synthrad.recovery import score_permutation  # noqa: E402
from iot_experiments.studies.real.synthrad.v2_run import build_table  # noqa: E402

BONE_WINDOW: tuple[float, float] = (-200.0, 1500.0)

PROVENANCE: dict[str, object] = {
    "purpose": "cheap baselines and two corrections, after Phase C was blocked",
    "A": "raw-pixel MSE and correlation, no learning, exact assignment",
    "B": "candidates restricted by outline-area similarity, applied equally to pixels and features",
    "C": "paired soft vs bone CT-window table, same patients and pool",
    "metric": "exact_hit (patient AND slice) and patient_hit, scored against the ground-truth ids",
    "memory": "hard-capped at 8 GB by the launcher",
}


class CheapError(RuntimeError):
    """CN: 前置条件失败。EN: a precondition failed."""


def _rss_gb() -> float:
    """CN: 峰值 RSS（GB）。EN: peak RSS in GB."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6


def pixel_cost(mr: Float[np.ndarray, "m pixels"], ct: Float[np.ndarray, "n pixels"], *, mode: str,
               block: int = 256) -> Float[np.ndarray, "m n"]:
    """
    CN: 原始像素代价（越小越好）：`mse` 或 `corr`（= `1 - 相关系数`，行归一化后即余弦距离）。
        分块计算以控制内存。**不用任何学习、不用 DINOv2。**
    EN: raw-pixel cost (lower is better): `mse` or `corr` (`1 - correlation`, cosine distance on row-normalised
        rows). Computed in blocks to bound memory. **No learning, no DINOv2.**
    """
    a = np.asarray(mr, dtype=np.float64)
    b = np.asarray(ct, dtype=np.float64)
    out = np.empty((a.shape[0], b.shape[0]), dtype=np.float64)
    if mode == "mse":
        aa = (a * a).sum(axis=1)
        bb = (b * b).sum(axis=1)
        for r0 in range(0, a.shape[0], int(block)):
            r1 = min(r0 + int(block), a.shape[0])
            out[r0:r1] = aa[r0:r1, None] + bb[None, :] - 2.0 * (a[r0:r1] @ b.T)
        return out / float(a.shape[1])
    if mode == "corr":
        na = np.linalg.norm(a, axis=1, keepdims=True)
        nb = np.linalg.norm(b, axis=1, keepdims=True)
        ua = a / np.where(na > 0, na, 1.0)
        ub = b / np.where(nb > 0, nb, 1.0)
        for r0 in range(0, a.shape[0], int(block)):
            r1 = min(r0 + int(block), a.shape[0])
            out[r0:r1] = 1.0 - ua[r0:r1] @ ub.T
        return out
    raise CheapError(f"unknown pixel cost {mode!r}")


def hungarian(cost: Float[np.ndarray, "m n"]) -> Int[np.ndarray, "m"]:
    """CN: 精确指派（方阵）。EN: exact assignment (square)."""
    from scipy.optimize import linear_sum_assignment  # noqa: PLC0415

    mtx = np.asarray(cost, dtype=np.float64)
    if mtx.shape[0] != mtx.shape[1]:
        raise CheapError(f"need a square cost, got {mtx.shape}")
    rows, cols = linear_sum_assignment(mtx)
    perm = np.empty(mtx.shape[0], dtype=np.int64)
    perm[rows] = cols
    return perm


def restricted_assignment(cost: Float[np.ndarray, "m n"], *, allowed: Bool[np.ndarray, "m n"]
                          ) -> tuple[Int[np.ndarray, "m"], int]:
    """
    CN: 只允许走 `allowed` 的边；不允许的边给极大代价（保持方阵）。返回 `(perm, 无候选的行数)`。
        CN: 无候选的行退回该行全局最小并计数——报告里给出，不静默。
    EN: only `allowed` edges are usable; disallowed ones get a huge cost (the matrix stays square). Returns
        `(perm, rows_without_any_candidate)`. Rows with no candidate fall back to their global minimum and are
        counted, so the report can state it rather than hide it.
    """
    mtx = np.asarray(cost, dtype=np.float64)
    big = float(np.max(np.abs(mtx)) + 1.0) * 10.0
    empty = int((~np.asarray(allowed, dtype=bool).any(axis=1)).sum())
    masked = np.where(np.asarray(allowed, dtype=bool), mtx, big)
    return hungarian(masked), empty


def pool_features(mr: Float[np.ndarray, "n pixels"], ct: Float[np.ndarray, "n pixels"], *, k: int,
                  geometry: str) -> tuple[Float[np.ndarray, "n k"], Float[np.ndarray, "n k"]]:
    """
    CN: 池内两模态共享编码器 + label-free PCA（在 4,000 行样本上拟合，内存友好）+ 几何 + 逐行 L2。
    EN: shared encoder and label-free PCA for the pool (fitted on a 4,000-row sample for memory) + geometry +
        row L2.
    """
    side = IMAGE_SIZE
    a = np.asarray(mr).reshape(-1, side, side)
    b = np.asarray(ct).reshape(-1, side, side)
    fa = encode_pool4(a, device="cuda").pooled.astype(np.float64)
    fb = encode_pool4(b, device="cuda").pooled.astype(np.float64)
    stacked = np.concatenate([fa, fb], axis=0)
    if stacked.shape[0] > 4000:
        take = np.sort(np.random.default_rng(424_242).permutation(stacked.shape[0])[:4000])
        sample = stacked[take]
    else:
        sample = stacked
    basis = pca_fit(sample, k=int(k))
    pa, pb = pca_apply(fa, basis), pca_apply(fb, basis)
    stats = fit_geometry(pa, pb, str(geometry))
    out_a, out_b = apply_geometry(pa, pb, stats)
    return np.asarray(out_a, dtype=np.float64), np.asarray(out_b, dtype=np.float64)


@dataclass(frozen=True)
class CheapConfig(RunConfig):
    """CN: 本组的旋钮。EN: this group's knobs."""

    section = "synthrad_cheap"
    study: str = "synthrad_cheap"
    ks: tuple[int, ...] = (64, 128)
    geometry: str = "G3"
    draw: int = 0
    slices_per_patient: int = 14
    min_gap: int = 8
    outline_tolerance: float = 0.10
    features: bool = True

    @classmethod
    def add_study_arguments(cls, parser) -> None:
        """CN: 只加本组特有的参数。EN: only this group's flags."""
        parser.add_argument("--ks", type=int, nargs="+", default=[64, 128])
        parser.add_argument("--outline-tolerance", type=float, default=0.10,
                            help="relative body-outline area difference allowed between a pair")
        parser.add_argument("--features", action=argparse.BooleanOptionalAction, default=True,
                            help="also run the feature arms (needs CUDA); --no-features for pixels only")


def main(argv: list[str] | None = None) -> int:
    """CN: 入口。EN: entry point."""
    parser = build_parser(CheapConfig)
    args = parser.parse_args(list(argv) if argv is not None else None)
    config = CheapConfig.from_namespace(args, parser)
    log = RunLog(config.run_dir / "run.log")
    log.line(f"config: ks={config.ks} geometry={config.geometry} draw={config.draw} "
             f"outline_tol={config.outline_tolerance}")

    split = json.loads((config.run_dir.parent / "synthrad_v2" / "split_v2.json").read_text())
    patients = discover_patients(_REPO_ROOT / "data" / "Task1", anatomy="brain")
    dev_pats = [p for p in patients if p.patient in set(split["dev"])]

    records: list[RunRecord] = []
    for window_name, window in (("soft", CT_WINDOW), ("bone", BONE_WINDOW)):
        log.line(f"=== window {window_name} {window} ===")
        t0 = time.perf_counter()
        pool = build_table(dev_pats, window=window, max_per_patient=int(config.slices_per_patient),
                           min_gap=int(config.min_gap), seed=10_000 + int(config.draw))
        n = len(pool)
        n_pat = len(set(pool.patient))
        log.line(f"[{window_name}] dev pool {n} slices, {n_pat} patients; chance exact 1/{n}="
                 f"{1.0 / n:.5f}, patient 1/{n_pat}={1.0 / n_pat:.5f} "
                 f"({time.perf_counter() - t0:.1f}s, peak {_rss_gb():.1f} GB)")

        order = np.random.default_rng(50_000 + int(config.draw)).permutation(n)
        pat_cols = [pool.patient[i] for i in order]
        z_cols = pool.z_index[order]
        ct_cols = np.asarray(pool.ct)[order]
        mask_cols = np.asarray(pool.mask)[order]

        def score(perm: Int[np.ndarray, "n"], arm: str, cell: str, extra: dict[str, object]) -> None:
            """CN: 评分并记一条。EN: score and record one row."""
            met = score_permutation(perm, patient_rows=pool.patient, patient_cols=pat_cols,
                                    z_rows=pool.z_index, z_cols=z_cols)
            records.append(RunRecord(study=str(config.study), geometry=cell, arm=arm,
                                     k=int(extra.get("k", 0)), m=n, draw=int(config.draw),
                                     metrics=met, config=config.to_dict(), extra=extra))
            log.line(f"[{window_name}] {cell:<18} {arm:<11} exact={met['exact_hit']:.4f} "
                     f"patient={met['patient_hit']:.4f}")

        # ---- (A) raw pixels -------------------------------------------------
        for mode in ("mse", "corr"):
            t1 = time.perf_counter()
            cost = pixel_cost(pool.mr, ct_cols, mode=mode)
            score(hungarian(cost), f"pixel_{mode}", "pixel", {"window": window_name, "n": n,
                                                             "seconds": round(time.perf_counter() - t1, 2)})

        # ---- features: cosine, and the SAME restriction as the pixels --------
        if not bool(config.features):
            log.line(f"[{window_name}] features disabled (--no-features)")
            continue
        for k in config.ks:
            t1 = time.perf_counter()
            fm, fc = pool_features(pool.mr, ct_cols, k=int(k), geometry=str(config.geometry))
            cos_cost = 1.0 - fm @ fc.T
            score(hungarian(cos_cost), "cosine", "feat", {"window": window_name, "k": int(k),
                                                          "n": n,
                                                          "seconds": round(time.perf_counter() - t1, 2)})
            areas_r = np.asarray(pool.mask).mean(axis=1)
            areas_c = np.asarray(mask_cols).mean(axis=1)
            rel = np.abs(areas_r[:, None] - areas_c[None, :]) / np.maximum(areas_c[None, :], 1e-6)
            allowed = rel <= float(config.outline_tolerance)
            mean_cand = float(allowed.sum(axis=1).mean())
            perm_c, empty_c = restricted_assignment(cos_cost, allowed=allowed)
            score(perm_c, "cosine", "restricted", {"window": window_name, "k": int(k), "n": n,
                                                   "mean_candidates": round(mean_cand, 1),
                                                   "rows_without_candidate": empty_c})
            perm_p, _ = restricted_assignment(pixel_cost(pool.mr, ct_cols, mode="corr"), allowed=allowed)
            score(perm_p, "pixel_corr", "restricted", {"window": window_name, "k": int(k), "n": n,
                                                       "mean_candidates": round(mean_cand, 1)})
            del fm, fc

    log.line(f"peak RSS {_rss_gb():.1f} GB")
    rows = summarise(records, group=("geometry", "arm"), metrics=("exact_hit", "patient_hit"))
    log.line("\n" + format_table(rows, ("geometry", "arm", "n_draws", "exact_hit_mean", "exact_hit_std",
                                        "patient_hit_mean", "patient_hit_std")))
    write_records(config.run_dir / "records.json", records, provenance=PROVENANCE)
    log.line(f"wrote {config.run_dir / 'records.json'}")
    log.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
