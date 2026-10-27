"""
CN: **数据探针（Phase 1）** —— 在**真实** SynthRAD brain 切片上做描述性审计，不做任何方法比较。

    CN: **患者划分先定，后看数据**：`reserve_split` 用固定种子把每个中心的患者分成
        probe / test 两部分，并把结果写入 manifest。probe 患者用于 Phase 1–3 的所有读数；
        **test 患者在被要求做最终评估之前不得触碰**。划分是**患者级**的，因此不存在切片泄漏。

    CN: **没有 fixture**：本脚本直接读 `data/Task1`。合成 fixture 曾与真实两层布局不一致，
        导致全绿而真实调用崩溃；这里只信任真实文件。

    CN: 本阶段只报告：
        * 每患者/中心/模态的切片数；
        * 每个中心的 `z_normalised` 覆盖；
        * 掩膜面积分布；
        * MR 与 CT 的强度直方图**按中心**（中心 B 为钆增强，单独标出）；
        * 异常：轴翻转、z 范围差异、空切片、掩膜过小。

EN: **Data probe (Phase 1)** -- a descriptive audit on **real** SynthRAD brain slices. No method comparison.

    **The patient split is fixed before looking at data**: `reserve_split` divides each centre's patients
    into probe / test with a fixed seed and writes a manifest. Probe patients feed every read-out in
    Phases 1-3; **test patients are not touched until final evaluation is requested**. The split is at
    **patient level**, so no slice can leak across it.

    There is **no fixture**: this reads `data/Task1` directly. A synthetic fixture once disagreed with the
    real two-level layout, which let a green suite hide a crash on real data; here only real files count.

    Reported: slice counts per patient/centre/modality, `z_normalised` coverage per centre, mask-area
    distribution, MR and CT intensity histograms **per centre** (centre B is gadolinium-enhanced and is
    called out explicitly), and anomalies (axis flips, z-range differences, empty slices, small masks).
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from jaxtyping import Float

_REPO_ROOT = Path(__file__).resolve().parents[4]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from iot_experiments.harness.cache import read_slices, save_slices, slices_path  # noqa: E402
from iot_experiments.harness.config import RunConfig, build_parser  # noqa: E402
from iot_experiments.harness.records import RunLog, format_table, write_records  # noqa: E402
from iot_experiments.harness.synthrad import (  # noqa: E402
    CT_WINDOW,
    IMAGE_SIZE,
    MIN_MASK_AREA,
    MR_PERCENTILES,
    PatientPaths,
    SynthRadSlices,
    discover_patients,
    load_volume,
    preprocess_patient,
)

PROVENANCE: dict[str, object] = {
    "dataset": "SynthRAD2023 Task 1, brain (real files under data/Task1/brain)",
    "preprocessing": "fixed: axial, 192x192, drop mask area < 5%, CT window -200..300 HU, MR per-volume 1/99 pct",
    "split": "patient-level, fixed seed, probe/test disjoint; test patients untouched",
    "scope": "descriptive only -- no method is scored in this phase",
}


class ProbeError(RuntimeError):
    """CN: 探针的前置条件失败。EN: a precondition of the probe failed."""


def reserve_split(patients: list[PatientPaths], *, probe_per_centre: int, seed: int
                  ) -> tuple[list[PatientPaths], list[PatientPaths], dict[str, object]]:
    """
    CN: 每个中心随机取 `probe_per_centre` 个患者作探针，其余作**测试集**（先保留，不触碰）。
        种子固定，因此划分可复现；manifest 记录两边的患者 id 与种子。
    EN: take `probe_per_centre` patients per centre as the probe and keep the rest as the **test set**
        (reserved, untouched). The seed is fixed so the split reproduces; the manifest records both id lists
        and the seed.
    """
    by_centre: dict[str, list[PatientPaths]] = {}
    for p in patients:
        by_centre.setdefault(p.centre, []).append(p)
    rng = np.random.default_rng(int(seed))
    probe: list[PatientPaths] = []
    reserved: list[PatientPaths] = []
    for centre in sorted(by_centre):
        group = sorted(by_centre[centre], key=lambda p: p.patient)
        order = rng.permutation(len(group))
        if len(group) < 2 * int(probe_per_centre):
            raise ProbeError(f"centre {centre} has only {len(group)} patients; need at least "
                             f"2*{probe_per_centre} to keep the test set disjoint and non-empty")
        chosen = set(order[: int(probe_per_centre)].tolist())
        for idx, p in enumerate(group):
            (probe if idx in chosen else reserved).append(p)
    manifest: dict[str, object] = {
        "seed": int(seed),
        "probe_per_centre": int(probe_per_centre),
        "probe": sorted(p.patient for p in probe),
        "test_reserved": sorted(p.patient for p in reserved),
        "probe_centres": {c: sum(1 for p in probe if p.centre == c) for c in sorted(by_centre)},
        "test_centres": {c: sum(1 for p in reserved if p.centre == c) for c in sorted(by_centre)},
    }
    return probe, reserved, manifest


def load_or_preprocess(paths: PatientPaths, *, cache: bool) -> SynthRadSlices:
    """CN: 优先读缓存；没有则预处理并写缓存。EN: read the cache when present, else preprocess and write it."""
    target = slices_path(paths.patient, size=IMAGE_SIZE, ct_window=CT_WINDOW,
                         mr_percentiles=MR_PERCENTILES, min_mask_area=MIN_MASK_AREA,
                         anatomy=paths.anatomy.lower())
    if cache and target.is_file():
        data = read_slices(target)
        return SynthRadSlices(
            patient=str(data["meta"]["patient"]), centre=str(data["meta"]["centre"]),
            mr=np.asarray(data["mr"]), ct=np.asarray(data["ct"]), mask=np.asarray(data["mask"]),
            z_normalised=np.asarray(data["z_normalised"]), z_index=np.asarray(data["z_index"]),
            meta=dict(data["meta"]))
    slices = preprocess_patient(paths)
    if cache:
        save_slices(target, mr=slices.mr, ct=slices.ct, mask=slices.mask,
                    z_normalised=slices.z_normalised, z_index=slices.z_index,
                    patient=slices.patient, centre=slices.centre, meta=slices.meta)
    return slices


@dataclass(frozen=True)
class ProbeConfig(RunConfig):
    """CN: 探针的旋钮。EN: the probe's knobs."""

    section = "synthrad_probe"
    study: str = "synthrad_probe"
    anatomy: str = "brain"
    probe_per_centre: int = 10
    split_seed: int = 7
    cache: bool = True

    @classmethod
    def add_study_arguments(cls, parser: argparse.ArgumentParser) -> None:
        """CN: 只加本阶段特有的参数。EN: only this phase's flags."""
        parser.add_argument("--anatomy", default="brain", choices=["brain", "pelvis"])
        parser.add_argument("--probe-per-centre", type=int, default=10,
                            help="patients per centre used for the probe; the rest form the reserved test set")
        parser.add_argument("--split-seed", type=int, default=7, help="fixed seed for the patient split")
        parser.add_argument("--cache", action=argparse.BooleanOptionalAction, default=True,
                            help="read/write the slice cache (default: on)")


def _nonzero_quantile(volume: Float[np.ndarray, "x y z"], q: float) -> float:
    """
    CN: 在**非零体素**上取分位数。原始 MR/CT 中约 60% 是精确的背景 0，直接在全体素上取中位数恒为 0。
    EN: quantile over **nonzero voxels**. About 60% of raw MR/CT voxels are exact background zero, so a
        plain median over all voxels is always 0.
    """
    values = np.asarray(volume, dtype=np.float64).reshape(-1)
    positive = values[values > 0.0]
    return float(np.percentile(positive, float(q))) if positive.size else float("nan")


def _hist(values: Float[np.ndarray, "pixels"], *, bins: int, lo: float, hi: float) -> list[int]:
    """CN: 固定边界的直方图计数（便于跨中心比较）。EN: histogram counts on fixed edges (comparable across centres)."""
    counts, _edges = np.histogram(np.clip(values, lo, hi), bins=int(bins), range=(lo, hi))
    return [int(c) for c in counts]


def main(argv: list[str] | None = None) -> int:
    """CN: 入口。EN: entry point."""
    parser = build_parser(ProbeConfig)
    args = parser.parse_args(list(argv) if argv is not None else None)
    config = ProbeConfig.from_namespace(args, parser)

    log = RunLog(config.run_dir / "run.log")
    log.line(f"config: anatomy={config.anatomy} probe_per_centre={config.probe_per_centre} "
             f"split_seed={config.split_seed} cache={config.cache}")
    root = _REPO_ROOT / "data" / "Task1"
    patients = discover_patients(root, anatomy=str(config.anatomy))
    probe, reserved, manifest = reserve_split(patients, probe_per_centre=int(config.probe_per_centre),
                                              seed=int(config.split_seed))
    log.line(f"split: {len(probe)} probe / {len(reserved)} reserved test patients "
             f"(seed {config.split_seed}); probe per centre {manifest['probe_centres']}")
    (config.run_dir / "split_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n",
                                                        encoding="utf-8")
    log.line(f"wrote {config.run_dir / 'split_manifest.json'}")

    rows: list[dict[str, object]] = []
    anomalies: list[str] = []
    for paths in probe:
        slices = load_or_preprocess(paths, cache=bool(config.cache))
        mr_vol = load_volume(paths.mr)
        ct_vol = load_volume(paths.ct)
        if not (mr_vol.is_axial_first() and ct_vol.is_axial_first()):
            anomalies.append(f"{paths.patient}: axis 2 is not superior-inferior")
        if mr_vol.data.shape != ct_vol.data.shape:
            anomalies.append(f"{paths.patient}: mr/ct shape mismatch {mr_vol.data.shape} vs {ct_vol.data.shape}")
        mask_area: Float[np.ndarray, "slices"] = slices.mask.mean(axis=1)
        if len(slices) == 0:
            anomalies.append(f"{paths.patient}: no slices survived the mask rule")
            continue
        if float(mask_area.min()) < MIN_MASK_AREA:
            anomalies.append(f"{paths.patient}: kept a slice with mask area {float(mask_area.min()):.4f}")
        # CN: 同一个患者内 MR 与 CT 的切片必须对齐（同一 z）。
        # EN: within a patient, MR and CT slices must be aligned (same z).
        rows.append({
            "patient": paths.patient, "centre": paths.centre,
            "n_total": int(slices.meta["n_total_slices"]), "n_kept": int(len(slices)),
            "dropped": int(slices.meta["n_total_slices"]) - int(len(slices)),
            "z_min": float(slices.z_normalised.min()), "z_max": float(slices.z_normalised.max()),
            "mask_mean": float(mask_area.mean()), "mask_p05": float(np.percentile(mask_area, 5)),
            "mask_p95": float(np.percentile(mask_area, 95)),
            "mr_mean": float(slices.mr.mean()), "mr_p05": float(np.percentile(slices.mr, 5)),
            "mr_p95": float(np.percentile(slices.mr, 95)),
            "ct_mean": float(slices.ct.mean()), "ct_p05": float(np.percentile(slices.ct, 5)),
            "ct_p95": float(np.percentile(slices.ct, 95)),
            # CN: 钆增强在**原始强度**上更清楚。注意 60% 的体素是精确的背景 0，因此**中位数恒为 0**、
            #     没有信息量；这里只在**非零体素**上取分位数（`_nonzero`），否则中心差异会被背景淹没。
            # EN: gadolinium enhancement is clearer on raw intensities. About 60% of voxels are exact
            #     background zero, so the plain median is always 0 and carries no information; quantiles are
            #     taken over **nonzero voxels** only, or the background would swamp the centre differences.
            "raw_mr_p50_nonzero": _nonzero_quantile(mr_vol.data, 50),
            "raw_mr_p95_nonzero": _nonzero_quantile(mr_vol.data, 95),
            "raw_ct_p50_nonzero": _nonzero_quantile(ct_vol.data, 50),
            "raw_ct_hu_p95": float(np.percentile(ct_vol.data, 95)),
        })
        log.line(f"{paths.patient} centre={paths.centre} slices {rows[-1]['n_total']}->{rows[-1]['n_kept']} "
                 f"z=[{rows[-1]['z_min']:.2f},{rows[-1]['z_max']:.2f}] mask {rows[-1]['mask_mean']:.3f}")

    if not rows:
        raise ProbeError("no probe patient produced slices")
    log.line("\n" + format_table(rows, ("patient", "centre", "n_total", "n_kept", "dropped",
                                        "mask_mean", "mask_p05", "mr_mean", "ct_mean",
                                        "raw_ct_hu_p95")))

    # CN: 按中心汇总：切片数、z 覆盖、掩膜面积、强度分位数。
    # EN: per-centre summary: slice counts, z coverage, mask area, intensity quantiles.
    summary: list[dict[str, object]] = []
    for centre in sorted({str(r["centre"]) for r in rows}):
        group = [r for r in rows if r["centre"] == centre]
        summary.append({
            "centre": centre, "patients": len(group),
            "slices_kept": sum(int(r["n_kept"]) for r in group),
            "slices_per_patient": round(float(np.mean([int(r["n_kept"]) for r in group])), 1),
            "z_min": round(min(float(r["z_min"]) for r in group), 3),
            "z_max": round(max(float(r["z_max"]) for r in group), 3),
            "mask_mean": round(float(np.mean([float(r["mask_mean"]) for r in group])), 4),
            "mask_p05": round(float(np.mean([float(r["mask_p05"]) for r in group])), 4),
            "mr_p50_nz": round(float(np.nanmean([float(r["raw_mr_p50_nonzero"]) for r in group])), 1),
            "mr_p95_nz": round(float(np.nanmean([float(r["raw_mr_p95_nonzero"]) for r in group])), 1),
            "raw_ct_hu_p95": round(float(np.mean([float(r["raw_ct_hu_p95"]) for r in group])), 1),
        })
    log.line("\n" + format_table(summary, ("centre", "patients", "slices_kept", "slices_per_patient",
                                           "z_min", "z_max", "mask_mean", "mask_p05",
                                           "mr_p50_nz", "mr_p95_nz", "raw_ct_hu_p95")))

    # CN: 强度直方图按中心，固定 20 个箱；中心 B 是钆增强，报告里必须显式对照。
    # EN: intensity histograms per centre on 20 fixed bins; centre B is gadolinium-enhanced and must be shown
    #     explicitly against the others.
    histograms: dict[str, object] = {}
    for centre in sorted({str(r["centre"]) for r in rows}):
        group = [r for r in rows if r["centre"] == centre]
        mr_lo = min(float(r["mr_p05"]) for r in group)
        mr_hi = max(float(r["mr_p95"]) for r in group)
        histograms[centre] = {
            "mr_window": [mr_lo, mr_hi],
            "mr_p05_mean": round(float(np.mean([float(r["mr_p05"]) for r in group])), 4),
            "mr_p95_mean": round(float(np.mean([float(r["mr_p95"]) for r in group])), 4),
            "mr_mean_mean": round(float(np.mean([float(r["mr_mean"]) for r in group])), 4),
            "ct_mean_mean": round(float(np.mean([float(r["ct_mean"]) for r in group])), 4),
            "mr_p50_nz": round(float(np.nanmean([float(r["raw_mr_p50_nonzero"]) for r in group])), 1),
            "mr_p95_nz": round(float(np.nanmean([float(r["raw_mr_p95_nonzero"]) for r in group])), 1),
        }
    for centre, payload in histograms.items():
        log.line(f"intensity centre {centre}: {payload}")

    if anomalies:
        log.line(f"ANOMALIES ({len(anomalies)}):")
        for item in anomalies:
            log.line(f"  - {item}")
    else:
        log.line("anomalies: none (axes superior-inferior, shapes aligned, no kept slice under the mask floor)")

    write_records(config.run_dir / "records.json", [], provenance={**PROVENANCE,
                                                                  "summary": summary,
                                                                  "histograms": histograms,
                                                                  "anomalies": anomalies,
                                                                  "rows": rows})
    log.line(f"wrote {config.run_dir / 'records.json'}")
    log.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
