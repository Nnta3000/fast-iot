"""
EN: **FAUST error heat maps and left/right flip counts.**

    Two diagnostics the correspondence numbers cannot show:

      1. **Heat maps.** Colour the target surface by the geodesic error of each vertex, per arm
         (`position_only`, `uniform_theta`, `learned`, `oracle`). A good cost gives an evenly cool surface;
         a bad one gives hot patches on the limbs. Rendered at FULL resolution (6890 vertices): the cost is
         assembled with the bilinear identity so only an `n x n` matrix is ever materialised, and EMD on
         6890 x 6890 takes ~10 s.

      2. **Left/right flips.** `HKS` and curvature cannot tell the two sides of a body apart, so a cost can be
         confidently wrong in the mirrored way. A vertex counts as flipped when the prediction is geodesically
         **closer to the mirrored vertex than to the true one**, counted only where the error exceeds 1 %
         (`sqrt(area)`-normalised) and the mirror is a different vertex.

    **The mirror is a proxy.** FAUST poses are not symmetric (an arm can be raised), so the mirror is
    reflection across the estimated sagittal plane followed by nearest-neighbour; the reflection residual is
    recorded. Treat the flip rate as a large-scale diagnostic, not an exact isometry count.

    CN: **FAUST 误差热图与左右翻转计数**。热图在全分辨率下渲染（利用双线性恒等式，只物化 n×n 代价矩阵）；
        翻转的判定是"预测更接近镜像点而不是真值点"，只在误差 > 1% 且镜像不同于自身处统计。镜像是**代理**：
        FAUST 姿态并不左右对称。

    Run:
        .venv/bin/python -u -m iot_experiments.studies.real.faust.heatmaps --pairs 0 66 1 36
"""

from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

import numpy as np
from jaxtyping import Float, Int

_REPO_ROOT = Path(__file__).resolve().parents[4]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.collections import PolyCollection  # noqa: E402

from iot.core.features import SquaredDifferenceFeatures  # noqa: E402
from iot_experiments.harness.config import RunConfig, build_parser  # noqa: E402
from iot_experiments.harness.faust import (  # noqa: E402
    default_registrations_root,
    FaustMesh,
    geodesic_matrix,
    mirror_indices,
    read_ply,
    surface_area,
)
from iot_experiments.harness.records import (  # noqa: E402
    RunLog,
    RunRecord,
    format_table,
    summarise,
    write_records,
)
from iot_experiments.harness.faust_match import (  # noqa: E402
    cost_matrix,
    geodesic_errors,
    match,
)
from iot_experiments.studies.real.faust.correspondence import (  # noqa: E402
    descriptors,
    fit_theta,
    standardise,
)

DEFAULT_ROOT = str(default_registrations_root())

PROVENANCE: dict[str, object] = {
    "question": "where on the body are the errors, and how often is the mistake a left/right flip?",
    "resolution": "FULL 6890 vertices: the cost is built with the bilinear identity so only an n x n matrix is "
                  "materialised; EMD on 6890x6890 is ~10 s",
    "arms": "position_only, uniform_theta, learned (simplex, fitted on the training pair), oracle (fitted on "
            "the evaluated pair)",
    "flip_definition": "prediction geodesically closer to the MIRRORED vertex than to the true one, counted "
                       "only where error > 1% of sqrt(area) and the mirror is a different vertex",
    "mirror": "reflection across the estimated sagittal plane (x = mean x), then nearest neighbour; FAUST poses "
              "are not symmetric so this is a PROXY, and the reflection residual is recorded per figure",
    "colours": "one fixed colour scale per pair across arms, so panels are comparable; grey = no data",
    "memory": "theta is FITTED through `SquaredDifferenceFeatures`, which computes features only for the "
              "edges the solver requests, so no `(n, n, K)` tensor is ever built (that allocation is 4.94 GB "
              "at 6890 vertices and OOM-killed this job once, exit 137). The heat map is evaluated at full "
              "6890 resolution through the bilinear identity, which materialises only `n x n`.",
}


@dataclasses.dataclass(frozen=True)
class HeatmapConfig(RunConfig):
    """CN: 旋钮。EN: knobs."""

    study: str = "faust_heatmaps"
    section: str = "faust_heatmaps"
    root: str = DEFAULT_ROOT
    train_pair: tuple[int, ...] = (0, 13)
    pairs: tuple[int, ...] = (0, 66, 1, 36)
    rel: float | None = None
    n_sub: int = 800
    max_iterations: int = 30
    views: tuple[str, ...] = ("front", "side")

    @classmethod
    def add_study_arguments(cls, parser) -> None:
        """CN: 只加本研究特有的参数。EN: only this study's flags."""
        parser.add_argument("--root", default=DEFAULT_ROOT)
        parser.add_argument("--train-pair", type=int, nargs=2, default=[0, 13])
        parser.add_argument("--pairs", type=int, nargs="+", default=[0, 66, 1, 36])
        parser.add_argument("--rel", type=float, default=-1.0,
                            help="box width; negative means simplex-only")
        parser.add_argument("--n-sub", type=int, default=800,
                            help="vertices used to FIT theta (the evaluation stays at full resolution)")
        parser.add_argument("--max-iterations", type=int, default=30)
        parser.add_argument("--views", nargs="+", default=["front", "side"])


def _errors(pred: Int[np.ndarray, "n"], geo: Float[np.ndarray, "n n"], mirror: Int[np.ndarray, "n"],
            norm: float) -> dict[str, float]:
    """
    EN: the heat-map subset of `geodesic_errors`: mean and miss@5 % for colouring, plus the flip statistics.
        A thin wrapper, not a second implementation -- the keys are subsetted so the committed records keep
        their exact schema.
    CN: `geodesic_errors` 的热图子集（均值、miss@5% 与翻转统计）。只是薄包装，保留旧记录的确切字段。
    """
    full = geodesic_errors(pred, geo, norm, mirror)
    return {"mean": full["mean"], "miss_5pct": full["miss_5pct"], "flip_rate": full["flip_rate"],
            "flip_share_of_errors": full["flip_share_of_errors"]}


def _render(mesh: FaustMesh, values: Float[np.ndarray, "n"], path: Path, *, title: str,
            vmax: float, view: str) -> None:
    """
    CN: 正交投影 + 画家算法渲染曲面，逐面颜色 = 顶点误差均值。`view='front'` 沿 +z 看，`'side'` 沿 +x 看。
    EN: orthographic projection with a painter's algorithm; each face is coloured by the mean vertex error.
        `view='front'` looks along +z, `view='side'` along +x.
    """
    v = np.asarray(mesh.vertices, dtype=np.float64)
    f = np.asarray(mesh.faces, dtype=np.int64)
    centre = 0.5 * (v.min(axis=0) + v.max(axis=0))
    p = v - centre
    if view == "front":
        xy, depth = p[:, [0, 1]], p[:, 2]
    elif view == "side":
        xy, depth = p[:, [2, 1]], p[:, 0]
    else:
        raise SystemExit(f"unknown view {view!r}")
    order = np.argsort(depth[f].mean(axis=1))
    polys = xy[f[order]]
    face_colour = np.asarray(values, dtype=np.float64)[f[order]].mean(axis=1)
    fig, ax = plt.subplots(figsize=(4.2, 6.0), dpi=140)
    coll = PolyCollection(polys, array=face_colour, cmap="inferno", edgecolors="none",
                          clim=(0.0, float(vmax)))
    ax.add_collection(coll)
    ax.set_xlim(xy[:, 0].min(), xy[:, 0].max())
    ax.set_ylim(xy[:, 1].min(), xy[:, 1].max())
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title, fontsize=7)
    fig.colorbar(coll, ax=ax, fraction=0.035, pad=0.02,
                 label="geodesic error / sqrt(area)")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    """CN: 入口。EN: entry point."""
    parser = build_parser(HeatmapConfig)
    args = parser.parse_args(list(argv) if argv is not None else None)
    config = HeatmapConfig.from_namespace(args, parser)
    run_dir = config.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    log = RunLog(run_dir / "run.log")
    root = Path(config.root)
    rel = None if float(config.rel) < 0 else float(config.rel)
    flat = [int(v) for v in config.pairs]
    pairs = [(flat[i], flat[i + 1]) for i in range(0, len(flat), 2)]
    train_pair = (int(config.train_pair[0]), int(config.train_pair[1]))
    log.line(f"train pair {train_pair}; heat-map pairs {pairs}; rel={rel}")

    # CN: **拟合在子样本上做**。全分辨率 (6890,6890,13) 张量是 4.94 GB，而广播减法还会再分配一份同样大小
    #     的临时数组——实测就是这样被 OOM 杀掉的（exit 137）。评估仍走全分辨率，但用双线性恒等式，
    #     只物化 n×n。
    # EN: **fit on a subsample.** The full-resolution (6890, 6890, 13) tensor is 4.94 GB, and the broadcasting
    #     subtraction allocates a second array of the same size -- that is exactly how this job was OOM-killed
    #     (exit 137). Evaluation stays at full resolution but uses the bilinear identity, so only `n x n` is
    #     materialised.
    a_tr = read_ply(root / f"tr_reg_{train_pair[0]:03d}.ply")
    b_tr = read_ply(root / f"tr_reg_{train_pair[1]:03d}.ply")
    sub = np.sort(np.random.default_rng(0).permutation(a_tr.n_vertices)[: int(config.n_sub)])
    da_sub, db_sub = standardise(descriptors(a_tr)[sub], descriptors(b_tr)[sub])
    # CN: 按需特征，不物化 (n, n, K)。EN: on-demand features; no `(n, n, K)` tensor.
    fit = fit_theta(SquaredDifferenceFeatures(da_sub, db_sub), rel=rel,
                    max_iterations=int(config.max_iterations))
    theta = fit["theta"]
    if theta is None:
        raise SystemExit(f"training fit failed: {fit}")
    log.line(f"learned theta={np.round(theta, 4).tolist()} regret={fit['regret']:.4g}")
    log.line(f"theta mass by block: xyz {theta[0:3].sum():.3f} area {theta[3]:.3f} "
             f"curv {theta[4]:.3f} hks {theta[5:].sum():.3f}")

    rows: list[RunRecord] = []
    for pair in pairs:
        a_m = read_ply(root / f"tr_reg_{pair[0]:03d}.ply")
        b_m = read_ply(root / f"tr_reg_{pair[1]:03d}.ply")
        da, db = standardise(descriptors(a_m), descriptors(b_m))
        n = int(da.shape[0])
        all_idx = np.arange(n, dtype=np.int64)
        geo = geodesic_matrix(b_m.vertices, b_m.faces, all_idx)
        norm = float(np.sqrt(surface_area(b_m.vertices, b_m.faces)))
        mirror, residual = mirror_indices(b_m.vertices, all_idx)

        # CN: 神谕同样在子样本上拟合（全分辨率张量会 OOM）。EN: the oracle fits on the subsample too.
        o_sub = np.sort(np.random.default_rng(0).permutation(n)[: int(config.n_sub)])
        oa, ob = da[o_sub], db[o_sub]
        inner_oracle = fit_theta(SquaredDifferenceFeatures(oa, ob), rel=rel,
                                 max_iterations=int(config.max_iterations))
        thetas = {
            "position_only": np.asarray([1.0, 1.0, 1.0] + [0.0] * (da.shape[1] - 3)),
            "uniform_theta": np.full(da.shape[1], 1.0 / da.shape[1]),
            "learned": theta,
        }
        if inner_oracle["theta"] is not None:
            thetas["oracle"] = inner_oracle["theta"]

        per_arm: dict[str, dict[str, float]] = {}
        values: dict[str, Float[np.ndarray, "n"]] = {}
        for arm, th in thetas.items():
            pred = match(cost_matrix(da, db, th), num_itermax=20_000_000)
            met = _errors(pred, geo, mirror, norm)
            per_arm[arm] = met
            rows_dist = np.asarray(geo)[np.arange(n), pred] / norm
            values[arm] = rows_dist
            rows.append(RunRecord(study=str(config.study), geometry=f"pair{pair[0]:03d}-{pair[1]:03d}",
                                  arm=arm, k=int(da.shape[1]), m=n, draw=0, metrics=met,
                                  config=config.to_dict(),
                                  extra={"mirror_residual_median": float(np.median(residual)),
                                         "self_mirror_fraction": float(np.mean(mirror == all_idx))}))
            log.line(f"[{pair[0]:03d}-{pair[1]:03d}] {arm:<14} mean={met['mean']:.4f} "
                     f"miss5%={met['miss_5pct']:.3f} flip={met['flip_rate']:.3f} "
                     f"flip|err={met['flip_share_of_errors']:.3f}")

        vmax = float(np.percentile(np.concatenate([values[k] for k in values]), 98))
        # CN: 逐顶点误差数组随记录一起落盘，绘图端才能在不重算的前提下重绘（figures/ 只读记录）。
        # EN: the per-vertex error arrays are written next to the record so the figure builder can re-render
        #     them without recomputing anything (the figures/ tree only reads records).
        np.savez(
            run_dir / f"errors_pair{pair[0]:03d}-{pair[1]:03d}.npz",
            **{arm: np.asarray(values[arm], dtype=np.float64) for arm in values},
            vmax=np.float64(vmax),
            pair=np.asarray(pair, dtype=np.int64),
        )
        for view in config.views:
            for arm in thetas:
                _render(b_m, values[arm], run_dir / f"pair{pair[0]:03d}-{pair[1]:03d}_{arm}_{view}.png",
                        title=f"{pair[0]:03d}-{pair[1]:03d}  {arm}  {view}", vmax=vmax, view=view)
        log.line(f"   wrote {len(thetas) * len(config.views)} panels for this pair "
                 f"(colour scale 0..{vmax:.4f})")

    table = summarise(rows, group=("arm",), metrics=("mean", "miss_5pct", "flip_rate",
                                                     "flip_share_of_errors"))
    log.line("\n" + format_table(table, ("arm", "n_draws", "mean_mean", "miss_5pct_mean",
                                          "flip_rate_mean", "flip_share_of_errors_mean")))
    write_records(run_dir / "records.json", rows, provenance=PROVENANCE)
    log.line(f"wrote {run_dir / 'records.json'} ({len(rows)} rows)")
    log.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
