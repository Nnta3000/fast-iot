"""
EN: **FAUST data check -- pre-flight gate 6.0.**

    Everything in the correspondence plan rests on one claim: the 100 training registrations share a single
    mesh structure with 6890 vertices, so **vertex i on one shape corresponds to vertex i on every other**.
    If that is false, `pi_hat = I/n` is wrong and every number downstream is meaningless. This module tests
    it directly and refuses to pass silently.

    Checks:

      1. every registration has the same vertex and face counts;
      2. the **face arrays are identical** (topology, not just counts);
      3. per-vertex 1-ring **areas** correlate across registrations -- if the vertex order were scrambled this
         would be ~0, and it is the cheapest positive evidence that vertex i is the same anatomical point;
      4. edge lengths correlate, as a second, independent fingerprint;
      5. the id -> (subject, pose) structure is recovered from the data rather than assumed;
      6. the ground-truth-vertex files are per **scan**, not per registration (their length is the scan's
         vertex count), so they cannot be used to mask registration vertices directly.

    Run:
        .venv/bin/python -u -m iot_experiments.studies.real.faust.data_check

    CN: **FAUST 数据检查（预检门 6.0）**：注册网格是否共享同一拓扑、顶点顺序是否即对应关系、编号到
        (受试者, 姿态) 的结构、以及 gt 文件是**按扫描**而非按注册索引的。
"""

from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

import numpy as np
from jaxtyping import Float

_REPO_ROOT = Path(__file__).resolve().parents[4]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from iot_experiments.harness.config import RunConfig, build_parser  # noqa: E402
from iot_experiments.harness.faust import (  # noqa: E402
    default_registrations_root,
    edge_lengths,
    training_registrations,
    vertex_areas,
)
from iot_experiments.harness.preflight import check as preflight_check  # noqa: E402
from iot_experiments.harness.records import (  # noqa: E402
    RunLog,
    RunRecord,
    format_table,
    summarise,
    write_records,
)

DEFAULT_ROOT = str(default_registrations_root().parent)

PROVENANCE: dict[str, object] = {
    "question": "do the FAUST training registrations share one vertex ordering, so that pi_hat = I/n is the "
                "true correspondence?",
    "gate": "6.0 of notes/FAUST_CORRESPONDENCE_GUIDELINE.md -- stop if this fails",
    "data": "MPI-FAUST.zip -> training/registrations/tr_reg_{000..099}.ply (100 files, binary PLY)",
    "test_1": "vertex and face counts identical across all registrations",
    "test_2": "FACE ARRAYS identical (topology), not merely the same counts",
    "test_3": "per-vertex 1-ring area: pairwise Pearson correlation across registrations. A scrambled vertex "
              "order would give ~0; the observed value is the evidence that vertex i is the same point",
    "test_4": "per-edge length correlation, an independent fingerprint of the same claim",
    "test_5": "id -> (subject, pose) recovered from the area-correlation matrix, not assumed from a convention",
    "gt_files": "training/ground_truth_vertices/tr_gt_XXX.txt has one line per SCAN vertex (the scan has far "
                "more vertices than the 6890-vertex registration), so it indexes scans, not registrations",
    "no_geodesics": "no geodesic solver is installed; this check needs none (areas and edge lengths only)",
}


@dataclasses.dataclass(frozen=True)
class DataCheckConfig(RunConfig):
    """CN: 旋钮。EN: knobs."""

    study: str = "faust_data_check"
    section: str = "faust_data_check"
    root: str = DEFAULT_ROOT
    limit: int = 0          # 0 = all registrations
    pairs: int = 40         # pairs sampled for the correlation statistics

    @classmethod
    def add_study_arguments(cls, parser) -> None:
        """CN: 只加本研究特有的参数。EN: only this study's flags."""
        parser.add_argument("--root", default=DEFAULT_ROOT)
        parser.add_argument("--limit", type=int, default=0)
        parser.add_argument("--pairs", type=int, default=40)


def _structure(corr: Float[np.ndarray, "n n"], ids: list[int], log: RunLog) -> dict[str, object]:
    """
    CN: 从面积相关矩阵恢复编号结构：比较"subject-major"（id = 10*s + p）与"pose-major"（id = 10*p + s）
        两种分组，组内相关的均值更高者胜。**不假定**约定，用数据决定。
    EN: recover the id structure from the area-correlation matrix: compare "subject-major" (id = 10*s + p)
        against "pose-major" (id = 10*p + s) and keep whichever has the higher within-group mean
        correlation. The convention is **decided by the data**, not assumed.
    """
    n = len(ids)
    idx = {v: i for i, v in enumerate(ids)}
    out: dict[str, object] = {}
    for name, group in (("subject_major", lambda v: v // 10), ("pose_major", lambda v: v % 10)):
        within, between = [], []
        for a in range(n):
            for b in range(a + 1, n):
                (within if group(ids[a]) == group(ids[b]) else between).append(corr[a, b])
        out[name] = {"within": float(np.mean(within)) if within else float("nan"),
                     "between": float(np.mean(between)) if between else float("nan")}
        log.line(f"   {name}: within={out[name]['within']:.4f} between={out[name]['between']:.4f}")
    best = max(("subject_major", "pose_major"), key=lambda k: out[k]["within"])
    out["chosen"] = best
    _ = idx
    return out


def main(argv: list[str] | None = None) -> int:
    """CN: 入口。EN: entry point."""
    parser = build_parser(DataCheckConfig)
    args = parser.parse_args(list(argv) if argv is not None else None)
    config = DataCheckConfig.from_namespace(args, parser)
    run_dir = config.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    log = RunLog(run_dir / "run.log")
    pre = preflight_check(run_tiny=False, need_gpu=False)
    pre.write(run_dir / "preflight.json")
    log.line(f"preflight ok={pre.ok} mem={pre.mem_available_gb:.1f} GB")
    root = Path(config.root) / "registrations" if (Path(config.root) / "registrations").is_dir() else Path(config.root)
    log.line(f"registrations: {root}")

    meshes = training_registrations(root)
    ids = [int(m.path.stem.split("_")[-1]) for m in meshes]
    if int(config.limit) > 0:
        meshes, ids = meshes[: int(config.limit)], ids[: int(config.limit)]
    log.line(f"loaded {len(meshes)} registrations, ids {min(ids)}..{max(ids)}")

    rows: list[RunRecord] = []
    counts = {(m.n_vertices, m.n_faces) for m in meshes}
    rows.append(RunRecord(study=str(config.study), geometry="dataset", arm="counts", k=0,
                          m=len(meshes), draw=0,
                          metrics={"distinct_vertex_face_pairs": float(len(counts)),
                                   "n_vertices": float(min(c[0] for c in counts)),
                                   "n_faces": float(min(c[1] for c in counts))},
                          config=config.to_dict(),
                          extra={"pairs": sorted([[int(a), int(b)] for a, b in counts])}))
    log.line(f"1. vertex/face counts: {sorted(counts)}  -> {'PASS' if len(counts) == 1 else 'FAIL'}")

    faces_equal = all(np.array_equal(meshes[0].faces, m.faces) for m in meshes[1:])
    rows.append(RunRecord(study=str(config.study), geometry="dataset", arm="topology", k=0,
                          m=len(meshes), draw=0,
                          metrics={"faces_identical": float(bool(faces_equal))},
                          config=config.to_dict()))
    log.line(f"2. face arrays identical: {faces_equal} -> {'PASS' if faces_equal else 'FAIL'}")

    areas = np.stack([vertex_areas(m.vertices, m.faces) for m in meshes])
    edges = np.stack([edge_lengths(m.vertices, m.faces) for m in meshes])
    log.line(f"3. per-vertex area matrix {areas.shape}; per-edge length matrix {edges.shape}")

    rng = np.random.default_rng(int(config.seed))
    n = len(meshes)
    pairs = [(int(a), int(b)) for a, b in
             (rng.integers(0, n, size=(int(config.pairs), 2)) if n > 1 else np.zeros((0, 2), dtype=int))]
    pairs = [(a, b) for a, b in pairs if a != b]
    area_corr = [float(np.corrcoef(areas[a], areas[b])[0, 1]) for a, b in pairs]
    edge_corr = [float(np.corrcoef(edges[a], edges[b])[0, 1]) for a, b in pairs]
    rows.append(RunRecord(study=str(config.study), geometry="dataset", arm="area_corr", k=0, m=n, draw=0,
                          metrics={"mean": float(np.mean(area_corr)), "min": float(np.min(area_corr))},
                          config=config.to_dict(), extra={"pairs": [list(p) for p in pairs]}))
    rows.append(RunRecord(study=str(config.study), geometry="dataset", arm="edge_corr", k=0, m=n, draw=0,
                          metrics={"mean": float(np.mean(edge_corr)), "min": float(np.min(edge_corr))},
                          config=config.to_dict()))
    log.line(f"3/4. area corr mean={np.mean(area_corr):.4f} min={np.min(area_corr):.4f}; "
             f"edge corr mean={np.mean(edge_corr):.4f} min={np.min(edge_corr):.4f}")

    corr = np.corrcoef(areas)
    structure = _structure(corr, ids, log)
    rows.append(RunRecord(study=str(config.study), geometry="dataset", arm="structure", k=0, m=n, draw=0,
                          metrics={"within_" + str(structure["chosen"]): float(
                              structure[str(structure["chosen"])]["within"])},
                          config=config.to_dict(), extra={"structure": structure}))
    log.line(f"5. id structure: {structure['chosen']}")

    gt_dir = Path(config.root) / "ground_truth_vertices"
    gt_note: dict[str, object] = {"present": gt_dir.is_dir()}
    if gt_dir.is_dir():
        gt_files = sorted(gt_dir.glob("tr_gt_*.txt"))
        if gt_files:
            first = np.loadtxt(gt_files[0], dtype=np.int64)
            gt_note.update({"n_files": len(gt_files), "first_len": int(first.size),
                            "unique_values": sorted({int(v) for v in np.unique(first)})})
    rows.append(RunRecord(study=str(config.study), geometry="dataset", arm="ground_truth", k=0, m=n, draw=0,
                          metrics={}, config=config.to_dict(), extra=gt_note))
    log.line(f"6. ground-truth vertices: {gt_note}")

    passed = len(counts) == 1 and faces_equal and float(np.min(area_corr)) > 0.5 and float(np.min(edge_corr)) > 0.5
    rows.append(RunRecord(study=str(config.study), geometry="dataset", arm="gate_6.0", k=0, m=n, draw=0,
                          metrics={"passed": float(bool(passed))}, config=config.to_dict()))
    log.line(f"\nGATE 6.0: {'PASS' if passed else 'FAIL'} "
             f"(counts {len(counts) == 1}, topology {faces_equal}, "
             f"area>0.5 {float(np.min(area_corr)) > 0.5}, edge>0.5 {float(np.min(edge_corr)) > 0.5})")

    table = summarise(rows, group=("arm",), metrics=("mean", "min", "passed"))
    log.line("\n" + format_table(table, ("arm", "n_draws", "mean_mean", "min_mean", "passed_mean")))
    write_records(run_dir / "records.json", rows, provenance=PROVENANCE)
    log.line(f"wrote {run_dir / 'records.json'} ({len(rows)} rows)")
    log.close()
    return 0 if passed else 3


if __name__ == "__main__":
    raise SystemExit(main())
