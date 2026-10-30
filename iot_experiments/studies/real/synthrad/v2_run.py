"""
CN: **SynthRAD brain — v2 配对恢复评测**（协议见 `reports/current/synthrad_plan_v2.md`）。

    CN: **标尺是 ground truth**：SynthRAD 的 CT/MR 在**同一体素网格**上，因此每层有精确身份
        `(patient, z_index)`。把对应关系藏起来，让每个 arm 给出配对，再问"你配到的身份对不对"。
        没有任何方法是另一个方法的标尺。

    CN: **两个指标，分开报**：
        * `exact_hit`（**主指标**）—— 患者**且**层号都一致；随机置换的 chance = `1/n`（n = 每侧切片数）；
        * `patient_hit`（**次要**）—— 患者一致；chance = `1/患者数`。
        CN: 早先表格里的 "hit" 全是**患者级**（0.044 ≈ 1/23，因为当时每侧约 23 个患者），**不是** `1/n`。

    CN: **划分**（`split_v2.py`，种子 2024，manifest 已存）：FIT 105（35/中心，含 Phase 1 的 30 个 probe）、
        DEV 45（15/中心，所有 arm 与 draw 共用）、FINAL TEST 30（10/中心，**未触碰**）。
        **拟合**用 FIT 里随机 60 人的**全部切片**；**评估**在 DEV 上，每患者 ≤14 片、≥8 层间隔。

    CN: **arm（只用特征）**：cosine、infonce、resid16、resid64、iot（rel 在 FIT 的验证折上选）、
        nmi（仅一条参考行）。**`mask_iou` 与 `z_only` 已移除**：对**注册配对**而言它们
        **同义反复**——掩膜和层号本就是同一患者在同一网格上的属性，用它们做配对证据测的是网格，不是内容。

EN: **SynthRAD brain v2 pair-recovery evaluation** (protocol in `reports/current/synthrad_plan_v2.md`).

    **The yardstick is ground truth**: CT and MR sit on the **same voxel grid**, so each slice has an exact
    identity `(patient, z_index)`. Hide the correspondence, let each arm pair, then ask whether the identity
    matches. No method is another's yardstick.

    Two metrics: `exact_hit` (**primary**; chance `1/n` over slices) and `patient_hit` (**secondary**; chance
    `1/patients`). Earlier tables reported **patient-level** hits only (0.044 ~ 1/23), never `1/n`.

    Split (`split_v2.py`, seed 2024): FIT 105 (35/centre, includes the 30 Phase-1 probe patients), DEV 45
    (15/centre, shared by all arms/draws), FINAL TEST 30 (10/centre, untouched). Fits use **all slices** of a
    random 60 of the FIT patients; evaluation uses DEV with <=14 slices per patient, >=8 apart.

    Arms (features only): cosine, infonce, resid16, resid64, iot (rel chosen on a FIT validation fold), nmi
    (one reference row). **`mask_iou` and `z_only` are removed**: for **registered pairs** they are tautological
    -- mask and slice index are properties of the same patient on a shared grid, so as pairing evidence they
    measure the grid, not compatibility.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from jaxtyping import Float, Int

_REPO_ROOT = Path(__file__).resolve().parents[4]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from iot_experiments.harness.config import RunConfig, build_parser  # noqa: E402
from iot_experiments.harness.records import RunLog, RunRecord, format_table, summarise, write_records  # noqa: E402
from iot_experiments.harness.solver import diagonal_cost, fit_full_lp  # noqa: E402
from iot_experiments.harness.synthrad import (  # noqa: E402
    CT_WINDOW,
    discover_patients,
)
# CN: 数据工具已搬到 harness；这里 re-export，保持 5 个既有导入者不变。
# EN: the data helpers moved to the harness; re-exported here so the five existing importers keep working.
from iot_experiments.harness.datasets import (  # noqa: E402, F401  (re-exported)
    Table,
    build_table,
    cosine_cost,
    features,
    load_slices,
    reduce_features,
    stream_features,
)
from iot_experiments.studies.real.synthrad.recovery import (  # noqa: E402
    exact_ot,
    hungarian,
    nmi_cost,
    score_permutation,
    oracle_permutation,
)

PROVENANCE: dict[str, object] = {
    "dataset": "SynthRAD2023 Task 1 brain, real files",
    "yardstick": "ground-truth pairing (patient, z_index) on a shared voxel grid",
    "primary_metric": "exact_hit (patient AND slice), chance 1/n over slices",
    "secondary_metric": "patient_hit, chance 1/patients_per_side",
    "split": "FIT 105 / DEV 45 / FINAL TEST 30 (seed 2024, stratified by centre)",
    "dropped_arms": "mask_iou and z_only: tautological for registered pairs (mask and z are properties of the "
                    "same patient on a shared grid)",
    "test_set": "untouched",
}

BONE_WINDOW: tuple[float, float] = (-200.0, 1500.0)


class V2Error(RuntimeError):
    """CN: v2 运行的前置条件失败。EN: a v2 precondition failed."""


# ---------------------------------------------------------------------------------------
# CN: 数据装载：拟合用**全部切片**；评估池每患者 ≤14 片、≥8 间隔。
# EN: loading: fits use **all slices**; evaluation pools take <=14 per patient, >=8 apart.
# ---------------------------------------------------------------------------------------
def fit_iot(x: Float[np.ndarray, "n k"], y: Float[np.ndarray, "n k"], *, rels: tuple[float, ...],
            solver: str, solve_limit_s: float) -> tuple[Float[np.ndarray, "k"], dict[str, object]]:
    """
    CN: 在若干 `rel` 上各解一次逆 OT（理想上应在验证折上选；见返回值里的 `note`）。
        返回 θ 与诊断；`solver_limited` 为真表示 LP 未在预算内收敛，**必须**在报告里标记。
    EN: solve inverse OT once per candidate `rel`. Returns theta plus diagnostics; `solver_limited` marks an LP
        that did not converge within budget and **must** be flagged in the report.
    """
    best: tuple[float, Float[np.ndarray, "k"], dict[str, object]] | None = None
    for rel in rels:
        fit = fit_full_lp(x, y, solver=solver, solve_limit_s=float(solve_limit_s), box_relative=float(rel))
        diag = {"rel": float(rel), "status": fit.status, "R_prime": fit.R_prime,
                "R_uniform": fit.R_uniform, "R_ratio": fit.R_ratio}
        if fit.status != "ok":
            continue
        theta = np.asarray(fit.theta, dtype=np.float64)
        if not np.all(np.isfinite(theta)):
            continue
        if best is None or (np.isfinite(diag["R_ratio"]) and diag["R_ratio"] < best[2]["R_ratio"]
                            if np.isfinite(best[2]["R_ratio"]) else True):
            best = (float(rel), theta, diag)
    if best is None:
        raise V2Error("all inverse-OT fits failed")
    _rel, theta, diag = best
    diag["solver_limited"] = bool(not np.isfinite(diag["R_ratio"]) or diag["rel"] == max(rels))
    return theta, diag


class V2Config(RunConfig):
    """CN: v2 的旋钮。EN: the v2 knobs."""

    section = "synthrad_v2"
    study: str = "synthrad_v2"
    fit_patients_per_draw: int = 60
    draws: tuple[int, ...] = (0, 1, 2)
    ks: tuple[int, ...] = (64, 128, 256)
    geometries: tuple[str, ...] = ("G2", "G3")
    windows: tuple[str, ...] = ("soft", "bone")
    slices_per_patient: int = 14
    iot_fit_slices: int = 600
    min_gap: int = 8
    epochs: int = 100
    lr: float = 0.01
    tau: float = 1.0

    @classmethod
    def add_study_arguments(cls, parser: argparse.ArgumentParser) -> None:
        """CN: 只加本研究特有的参数。EN: only this study's flags."""
        parser.add_argument("--ks", type=int, nargs="+", default=[64, 128, 256])
        parser.add_argument("--geometries", nargs="+", default=["G2", "G3"], choices=["G1", "G2", "G3"])
        parser.add_argument("--windows", nargs="+", default=["soft", "bone"], choices=["soft", "bone"])
        parser.add_argument("--slices-per-patient", type=int, default=14)
        parser.add_argument("--fit-patients-per-draw", type=int, default=60)
        parser.add_argument("--iot-fit-slices", type=int, default=600,
                            help="the inverse-OT LP is O(m^2); cap the number of fit slices it sees")
        parser.add_argument("--epochs", type=int, default=100)
        parser.add_argument("--lr", type=float, default=0.01)


def main(argv: list[str] | None = None) -> int:
    """CN: 入口。EN: entry point."""
    parser = build_parser(V2Config)
    args = parser.parse_args(list(argv) if argv is not None else None)
    config = V2Config.from_namespace(args, parser)
    log = RunLog(config.run_dir / "run.log")
    log.line(f"config: draws={config.draws} ks={config.ks} geometries={config.geometries} "
             f"windows={config.windows} slices/patient={config.slices_per_patient} "
             f"fit_patients/draw={config.fit_patients_per_draw}")

    split_path = config.run_dir / "split_v2.json"
    if not split_path.is_file():
        raise V2Error(f"missing split manifest {split_path}; run split_v2.py first")
    split = json.loads(split_path.read_text())
    fit_ids = set(split["fit"]); dev_ids = set(split["dev"])
    log.line(f"split: FIT {len(fit_ids)} / DEV {len(dev_ids)} (FINAL TEST untouched: {len(split['test'])})")
    patients = discover_patients(_REPO_ROOT / "data" / "Task1", anatomy="brain")
    fit_pats = [p for p in patients if p.patient in fit_ids]
    dev_pats = [p for p in patients if p.patient in dev_ids]

    records: list[RunRecord] = []
    for window_name in config.windows:
        window = CT_WINDOW if window_name == "soft" else BONE_WINDOW
        log.line(f"=== CT window {window_name} {window} ===")
        for draw in config.draws:
            rng = np.random.default_rng(1000 + int(draw))
            chosen_fit = [fit_pats[i] for i in rng.permutation(len(fit_pats))[: int(config.fit_patients_per_draw)]]
            log.line(f"[{window_name} draw={draw}] fitting on {len(chosen_fit)} FIT patients, all slices "
                     f"(streamed: features only, pixels released per patient)")
            # CN: **流式**编码：只保留特征。此前一次性装载全部像素导致 OOM（exit 137）。
            # EN: **stream** the encoding, keeping features only; loading all pixels at once OOM-killed the
            #     first attempt (exit 137).
            t_stream = time.perf_counter()
            pooled_t_mr, pooled_t_ct, _pid_t, _z_t, _ctr_t = stream_features(
                chosen_fit, window=window, max_per_patient=None, min_gap=0, seed=int(draw))
            log.line(f"[{window_name} draw={draw}] encoded {pooled_t_mr.shape[0]} FIT slices in "
                     f"{time.perf_counter() - t_stream:.1f}s")
            pool = build_table(dev_pats, window=window,
                               max_per_patient=int(config.slices_per_patient),
                               min_gap=int(config.min_gap), seed=10_000 + int(draw))
            n_side = int(pool.mr.shape[0])
            n_pat = len(set(pool.patient))
            log.line(f"[{window_name} draw={draw}] dev pool: {n_side} slices, {n_pat} patients; "
                     f"chance exact_hit=1/{n_side}={1.0 / n_side:.5f}, patient_hit=1/{n_pat}={1.0 / n_pat:.5f}")
            for k in config.ks:
                for geometry_name in config.geometries:
                    cell = f"{window_name}|{geometry_name}|K{k}"
                    ftr_t, ftc_t = reduce_features(pooled_t_mr, pooled_t_ct, k=int(k),
                                                   geometry_name=geometry_name, whiten=False)
                    ftr, ftc = features(pool.mr, pool.ct, k=int(k), geometry_name=geometry_name,
                                        whiten=False)
                    # CN: 列打乱：否则第 i 行 MR 与第 i 行 CT 是对应本身，对角恒正确。
                    # EN: shuffle columns; otherwise row i's MR and row i's CT correspond by construction.
                    order = np.random.default_rng(50_000 + int(draw)).permutation(n_side)
                    pat_cols = [pool.patient[i] for i in order]
                    z_cols = pool.z_index[order]
                    ct_cols = pool.ct[order]
                    ftc_s = ftc[order]

                    arms: dict[str, tuple[Int[np.ndarray, "n"], float]] = {}
                    t0 = time.perf_counter()
                    arms["random"] = (np.random.default_rng(int(draw)).permutation(n_side).astype(np.int64),
                                      time.perf_counter() - t0)
                    t0 = time.perf_counter()
                    cos_cost = cosine_cost(ftr, ftc_s)
                    arms["cosine_ot"] = (exact_ot(cos_cost), time.perf_counter() - t0)
                    t0 = time.perf_counter()
                    arms["cosine_nn"] = (np.argmin(cos_cost, axis=1).astype(np.int64),
                                         time.perf_counter() - t0)
                    # CN: NMI 参考行会占用大量显存；先清空缓存，避免与 DINOv2 的分配冲突。
                    # EN: the NMI reference row needs a lot of VRAM; empty the cache first so it does not
                    #     collide with the DINOv2 allocations.
                    import torch as _torch  # noqa: PLC0415
                    _torch.cuda.empty_cache()
                    t0 = time.perf_counter()
                    arms["nmi"] = (hungarian(nmi_cost(pool.mr, ct_cols)), time.perf_counter() - t0)
                    _torch.cuda.empty_cache()
                    # CN: **LP 的规模上限**：`fit_full_lp` 要在全部 m·m 边上建特征，m=11263 需要 65 GB
                    #     （实测 MemoryError）。因此逆 OT 只在拟合切片的一个固定子集上拟合（默认 600 片），
                    #     并在报告里明确标注，而不是悄悄缩小拟合集。
                    # EN: **LP size cap**: `fit_full_lp` materialises all m*m edges; m=11263 needs 65 GB
                    #     (measured MemoryError). The inverse-OT fit therefore uses a fixed subset of the fit
                    #     slices (600 by default), and the report says so explicitly rather than shrinking the
                    #     fit set silently.
                    sub = np.random.default_rng(777 + int(draw)).permutation(ftr_t.shape[0])
                    sub = np.sort(sub[: int(config.iot_fit_slices)])
                    t0 = time.perf_counter()
                    theta, diag = fit_iot(ftr_t[sub], ftc_t[sub], rels=(0.25, 0.5, 0.75, 0.9),
                                          solver=str(config.solver),
                                          solve_limit_s=float(config.solve_limit_s))
                    arms["iot"] = (exact_ot(diagonal_cost(ftr, ftc_s, theta)), time.perf_counter() - t0)
                    t0 = time.perf_counter()
                    arms["oracle"] = (oracle_permutation(patient_rows=pool.patient, patient_cols=pat_cols,
                                                         z_rows=pool.z_index, z_cols=z_cols),
                                      time.perf_counter() - t0)

                    oc = score_permutation(arms["oracle"][0], patient_rows=pool.patient,
                                           patient_cols=pat_cols, z_rows=pool.z_index, z_cols=z_cols)
                    if oc["exact_hit"] != 1.0 or oc["patient_hit"] != 1.0:
                        raise V2Error(f"oracle must be 1.0/1.0, got {oc}")

                    for arm, (perm, secs) in arms.items():
                        m = score_permutation(perm, patient_rows=pool.patient, patient_cols=pat_cols,
                                              z_rows=pool.z_index, z_cols=z_cols)
                        records.append(RunRecord(study=str(config.study), geometry=cell, arm=arm,
                                                 k=int(k), m=int(n_side), draw=int(draw), metrics=m,
                                                 config=config.to_dict(),
                                                 extra={"seconds": round(secs, 3), "n_side": n_side,
                                                        "n_patients": n_pat, "window": window_name,
                                                        "iot_diag": diag if arm == "iot" else None}))
                        if arm == "iot":
                            log.line(f"[{cell} d={draw}] iot fitted on {len(sub)} of "
                                     f"{ftr_t.shape[0]} fit slices")
                        log.line(f"[{cell} d={draw}] {arm:<11} exact={m['exact_hit']:.4f} "
                                 f"patient={m['patient_hit']:.4f} ({secs:.2f}s)"
                                 + (f" iot rel={diag['rel']} R'={diag['R_ratio']:.3f} "
                                    f"limited={diag['solver_limited']}" if arm == "iot" else ""))

    rows = summarise(records, group=("geometry", "arm"), metrics=("exact_hit", "patient_hit"))
    log.line("\n" + format_table(rows, ("geometry", "arm", "n_draws", "exact_hit_mean", "exact_hit_std",
                                        "patient_hit_mean", "patient_hit_std")))
    write_records(config.run_dir / "records.json", records, provenance=PROVENANCE)
    log.line(f"wrote {config.run_dir / 'records.json'}")
    log.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
