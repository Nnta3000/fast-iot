"""
EN: **Study figures** -- F6 to F14, built only from records in `experiments/out/`.

    Every value is parsed from a `records.json` (or, for F6, from the per-vertex error arrays that sit next to
    the record) and compared against `EXPECTED`; a mismatch stops the run. The style contract is
    `figures/FIGURE_STYLE.md`; the shared helpers are `style.py` (look) and `metadata.py` (csv + caption).

    F6 is the only builder that needs a figure-specific input beyond a records file: the per-vertex error
    arrays written by `studies/iot/real/faust/heatmaps.py`. If they are absent the builder says so instead of
    recomputing the experiment here.

    CN: **研究图** F6–F13，只读 `experiments/out/` 下的记录。样式契约见 `figures/FIGURE_STYLE.md`；
        F6 需要 `heatmaps.py` 落盘的逐顶点误差数组，缺了就报错，不在这里重算实验。
"""

from __future__ import annotations

import sys
from pathlib import Path


import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PolyCollection

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from experiments.figures.metadata import (  # noqa: E402
    FigureError,
    check_values,
    load_records,
    print_caption,
    write_metadata,
)
from experiments.figures.style import (  # noqa: E402
    SEQUENTIAL_CMAP,
    apply_style,
    colour,
    marker,
    add_footer,
    add_title,
    save,
)

OUT: Path = _REPO_ROOT / "experiments" / "out"
FAUST_ROOT: Path = _REPO_ROOT / "data" / "faust_extract" / "MPI-FAUST" / "training" / "registrations"

# CN: 每个被画的数都必须与这里一致（值来自记录与报告，已逐项核对）。
# EN: every plotted aggregate must match these (record- and report-derived, verified item by item).
EXPECTED: dict[str, dict[str, float]] = {
    "F7": {"multi": 0.4666, "single": 0.1974, "uniform_theta": 0.4043},
    "F8": {
        "relaxed_lp/softmax_margin/auc": 0.6318,
        "relaxed_lp/dual_margin/auc": 0.5044,
        "relaxed_lp/cost_margin/auc": 0.4638,
        "relaxed_lp/softmax_margin/cov0.5": 0.1441,
        "relaxed_lp/softmax_margin/cov1": 0.2028,
    },
    "F9": {"learned/sigma0": 0.1613, "learned/sigma1": 0.7080, "uniform_theta/sigma1": 0.7747},
    "F10": {"selected": 0.1890, "reference": 0.2028, "median": 0.2232, "uniform": 0.4219},
    "F12": {"relaxed_lp": 0.1825, "row_softmax": 0.3425, "uniform": 0.4150},
    "F13": {"field": 0.3559, "global": 0.2006},
    "F14": {"position_only": 0.8550, "uniform_theta": 0.4567, "learned": 0.1550},
}

PAIR_ORDER: tuple[str, ...] = (
    "pair050-066", "pair053-085", "pair056-071", "pair059-099",
    "pair061-086", "pair064-072", "pair067-091", "pair072-085",
)


def _mean(values: list[float]) -> float:
    """CN: 均值（空列表即报错，不返回 nan）。EN: mean; an empty list raises instead of returning nan."""
    if not values:
        raise FigureError("no values to average")
    return float(np.mean(values))


# --------------------------------------------------------------------------------------------------
# F6 -- FAUST error heat maps
# --------------------------------------------------------------------------------------------------

def _mesh_panel(ax, mesh, values: np.ndarray, vmax: float, *, view: str = "front") -> PolyCollection:
    """
    CN: 正交投影 + 画家算法，逐面颜色 = 顶点误差均值。EN: orthographic projection with a painter's algorithm;
        each face is coloured by the mean of its vertex errors.
    """
    v = np.asarray(mesh.vertices, dtype=np.float64)
    f = np.asarray(mesh.faces, dtype=np.int64)
    centre = 0.5 * (v.min(axis=0) + v.max(axis=0))
    p = v - centre
    if view == "front":
        xy, depth = p[:, [0, 1]], p[:, 2]
    else:
        xy, depth = p[:, [2, 1]], p[:, 0]
    order = np.argsort(depth[f].mean(axis=1))
    polys = xy[f[order]]
    face_colour = np.asarray(values, dtype=np.float64)[f[order]].mean(axis=1)
    coll = PolyCollection(polys, array=face_colour, cmap=SEQUENTIAL_CMAP, edgecolors="none", clim=(0.0, float(vmax)))
    ax.add_collection(coll)
    ax.set_xlim(float(xy[:, 0].min()), float(xy[:, 0].max()))
    ax.set_ylim(float(xy[:, 1].min()), float(xy[:, 1].max()))
    ax.set_aspect("equal")
    ax.axis("off")
    return coll


def build_f6() -> Path:
    """CN: **F6** —— 两对留出网格 × 四个 arm 的误差热图。EN: **F6** -- error heat maps, two held-out pairs
        by four arms, one shared colour scale per pair."""
    from experiments.harness.faust import read_ply

    rows, _ = load_records(OUT / "faust_heatmaps" / "records.json")
    pairs = (("pair050-066", 66), ("pair056-071", 71))
    arms = ("position_only", "uniform_theta", "learned", "oracle")
    entries = {(row["geometry"], row["arm"]): row["metrics"] for row in rows}

    fig, axes = plt.subplots(len(pairs), len(arms), figsize=(10.0, 7.4))
    for r, (pair, target) in enumerate(pairs):
        npz_path = OUT / "faust_heatmaps" / f"errors_{pair}.npz"
        if not npz_path.is_file():
            raise FigureError(f"missing per-vertex errors: {npz_path} (rerun studies/iot/real/faust/heatmaps.py)")
        payload = np.load(npz_path)
        vmax = float(payload["vmax"])
        mesh = read_ply(FAUST_ROOT / f"tr_reg_{target:03d}.ply")
        coll = None
        for c, arm in enumerate(arms):
            ax = axes[r, c]
            coll = _mesh_panel(ax, mesh, payload[arm], vmax)
            met = entries[(pair, arm)]
            ax.set_title(
                f"{pair.replace('pair', '')}  {arm}\n"
                f"miss@5% {met['miss_5pct']:.3f}   flip {met['flip_rate']:.3f}",
                fontsize=7.5, loc="center", pad=3)
        if coll is not None:
            bar = fig.colorbar(coll, ax=list(axes[r, :]), fraction=0.012, pad=0.01)
            bar.set_label("geodesic error / √area", fontsize=7)
            bar.ax.tick_params(labelsize=6)

    experiment = (
        "Two held-out FAUST pairs at full 6890-vertex resolution: the target surface is coloured by the "
        "geodesic distance between the predicted and the true correspondence, in units of √area.")
    comparing = (
        "Four cost weightings of the same 13 descriptors, left to right: position only, equal weights, "
        "weights learned on a different pair of subjects, and an oracle fitted on the pair being shown.")
    config = (
        "MPI-FAUST training registrations; the learned θ comes from pair 0-13; one colour scale per row, "
        "the 98th percentile over the four arms; a flip is a vertex whose prediction is closer to the "
        "mirrored vertex than to the true one.")
    add_title(fig, "Where the correspondence is wrong", "Per-vertex geodesic error on two held-out FAUST pairs, four weightings",
              axes_left=0.02)
    add_footer(fig, experiment=experiment, comparing=comparing, config=config, axes_left=0.02, labelled=False)
    fig.subplots_adjust(left=0.02, right=0.94, top=0.88, bottom=0.11, wspace=0.05, hspace=0.18)
    png = save(fig, "F6_mesh_error_maps")

    csv_rows: list[list[object]] = []
    for pair, _ in pairs:
        for arm in arms:
            met = entries[(pair, arm)]
            for metric in ("mean", "miss_5pct", "flip_rate", "flip_share_of_errors"):
                csv_rows.append([pair, arm, metric, f"{float(met[metric]):.4f}"])
    write_metadata(
        "F6_mesh_error_maps",
        caption={
            "axes": "no axes: each panel is an orthographic front view of the target surface; colour is the "
                    "per-vertex geodesic error divided by √(surface area), on one scale per row.",
            "instance": "MPI-FAUST training registrations, pairs 050-066 and 056-071, 6890 vertices, "
                        "π̂ = I/n as ground truth; K = 13 descriptors",
            "methods": "position_only, uniform_theta, learned (simplex θ fitted on pair 0-13), oracle "
                       "(fitted on the evaluated pair); the oracle is a ceiling, not a competitor",
            "config": config,
            "sources": "out/faust_heatmaps/records.json plus the per-vertex arrays errors_pair*.npz written "
                       "by studies/iot/real/faust/heatmaps.py; both are git-ignored, the run regenerates them",
            "conventions": "a flip is counted only where the error exceeds 1 % of √area and the mirrored "
                           "vertex is a different vertex; the mirror is a reflection across the estimated "
                           "sagittal plane, so it is a proxy",
        },
        header=["pair", "arm", "metric", "value"],
        rows=csv_rows,
    )
    print_caption("F6_mesh_error_maps", experiment=experiment, comparing=comparing, config=config)
    return png


# --------------------------------------------------------------------------------------------------
# F7 -- cross-regret matrix and the held-out cost of sharing
# --------------------------------------------------------------------------------------------------

def build_f7() -> Path:
    """CN: **F7** —— 交叉 regret 热格 + 共享 θ 的留出表现。EN: **F7** -- cross-regret heat grid, plus how the
        shared θ scores on the held-out pairs."""
    rows, _ = load_records(OUT / "faust_multi_pair" / "records.json")
    matrix_row = next((row for row in rows if row["arm"] == "matrix"), None)
    if matrix_row is None:
        raise FigureError("faust_multi_pair: no cross_regret matrix row")
    matrix = np.asarray(matrix_row["extra"]["matrix"], dtype=np.float64)
    train_pairs = [tuple(int(v) for v in p) for p in matrix_row["extra"]["train_pairs"]]

    held = {arm: {r["geometry"]: float(r["metrics"]["miss_5pct"]) for r in rows
                  if r["geometry"] in PAIR_ORDER and r["arm"] == arm}
            for arm in ("multi", "single", "uniform_theta")}
    got = {arm: _mean(list(by_pair.values())) for arm, by_pair in held.items()}
    check_values("F7", got, EXPECTED["F7"])

    fig, (ax_m, ax_h) = plt.subplots(1, 2, figsize=(10.0, 3.9), gridspec_kw={"width_ratios": [1.15, 1.0]})
    apply_style()
    vmax = 0.35
    image = ax_m.imshow(np.clip(matrix, 0.0, vmax), cmap=SEQUENTIAL_CMAP, vmin=0.0, vmax=vmax, aspect="auto")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            ax_m.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=7.5,
                      color="white" if value < 0.18 else "black")
    for i in range(min(matrix.shape[0], matrix.shape[1] - 1)):
        ax_m.add_patch(plt.Rectangle((i - 0.5, i - 0.5), 1, 1, fill=False, edgecolor="#3ddc84", lw=1.2))
    pair_count = len(train_pairs)
    ax_m.set_xticks(range(matrix.shape[1]))
    ax_m.set_xticklabels([f"θ from\nPair {i + 1}" for i in range(pair_count)]
                         + ["θ shared\nby all four"], fontsize=7.5)
    ax_m.set_yticks(range(matrix.shape[0]))
    ax_m.set_yticklabels([f"Pair {i + 1} plan" for i in range(pair_count)], fontsize=7.5)
    ax_m.set_xlabel("the θ that was fitted")
    ax_m.set_ylabel("the plan it is scored on")
    ax_m.set_title("regret left on a plan, relative to equal weights", fontsize=9, loc="left", pad=4)
    bar = fig.colorbar(image, ax=ax_m, fraction=0.035, pad=0.02)
    bar.set_label("regret relative to uniform", fontsize=7)
    bar.ax.tick_params(labelsize=6.5)

    labels = {"multi": "θ shared by all four pairs", "single": "θ from Pair 1 alone",
              "uniform_theta": "equal weights"}
    for arm in ("multi", "single", "uniform_theta"):
        pairs_present = [i for i, p in enumerate(PAIR_ORDER) if p in held[arm]]
        ax_h.plot([held[arm][PAIR_ORDER[i]] for i in pairs_present], pairs_present, color=colour(arm),
                  marker=marker(arm), ls="none", label=labels[arm], zorder=3)
    ax_h.axvline(_mean(list(held["uniform_theta"].values())), color="#c9c9c9", lw=0.8, ls=(0, (3, 2)), zorder=1)
    ax_h.set_yticks(range(len(PAIR_ORDER)))
    ax_h.set_yticklabels([p.replace("pair", "") for p in PAIR_ORDER], fontsize=7.5)
    ax_h.set_xlabel("miss@5 % on the held-out pairs")
    ax_h.set_title("held-out error, one dot per pair", fontsize=9, loc="left", pad=4)
    ax_h.legend(loc="lower center", bbox_to_anchor=(0.5, -0.36), ncol=3, fontsize=7.5,
                columnspacing=1.0, handletextpad=0.3)
    ax_h.tick_params(length=3)
    for side in ("top", "right"):
        ax_h.spines[side].set_visible(False)
    ax_m.tick_params(length=0)

    mapping = "; ".join(
        f"Pair {i + 1} is subject {a // 10} in pose {a % 10} against subject {b // 10} in pose {b % 10}"
        for i, (a, b) in enumerate(train_pairs))
    experiment = (
        "Four FAUST pairs are fitted together: the same person in four different poses, each matched "
        "against a different second person.")
    comparing = (
        "Each column is one θ: the first four are fitted so that a single pair's plan becomes optimal, the "
        "last is one θ fitted so that all four plans are optimal at once. A row asks how much regret that θ "
        "leaves on one pair's plan; the right panel scores the shared θ, the Pair 1 θ and equal weights on "
        "the eight held-out pairs.")
    config = (
        "MPI-FAUST, n = 1200 vertices, K = 13 descriptors; regret is divided by that pair's uniform regret, "
        "so 1.0 means no better than equal weights; Pairs 3 and 4 use a second person who also appears "
        "among the evaluation subjects.")
    add_title(fig, "Fitting many plans with one θ", "Cross-regret between four FAUST pairs and the held-out cost of sharing",
              axes_left=0.075)
    add_footer(fig, experiment=experiment, comparing=comparing, config=config, axes_left=0.075, labelled=False)
    fig.subplots_adjust(left=0.075, right=0.99, top=0.80, bottom=0.22, wspace=0.32)
    png = save(fig, "F7_cross_regret")

    csv_rows = [[f"Pair {i + 1} plan", f"theta from Pair {j + 1}", f"{matrix[i, j]:.4f}"]
                for i in range(pair_count) for j in range(pair_count)]
    csv_rows += [[f"Pair {i + 1} plan", "theta shared by all four", f"{matrix[i, 4]:.4f}"]
                 for i in range(pair_count)]
    csv_rows += [[f"Pair {i + 1}", "registration_ids", f"{a:03d}-{b:03d}"]
                 for i, (a, b) in enumerate(train_pairs)]
    csv_rows += [[p, arm, f"{value:.4f}"]
                 for arm, by_pair in held.items() for p, value in by_pair.items()]
    write_metadata(
        "F7_cross_regret",
        caption={
            "axes": "left: rows are the plan being scored, columns the θ that was fitted, colour and number "
                    "the regret normalised by that plan's uniform regret; right: held-out miss@5 % on the "
                    "eight held-out pairs.",
            "instance": "MPI-FAUST, n = 1200 vertices, K = 13; " + mapping,
            "methods": "the first fourθs are fitted on one pair each, the last θ is shared by all four; the "
                       "green boxes mark the diagonal, where a θ is scored on the plan it was fitted to",
            "config": config,
            "sources": "out/faust_multi_pair/records.json; the matrix is the 'cross_regret' row, the held-out "
                       "values are the per-pair rows of the same record",
            "conventions": "the shared θ is the last column; colour saturates at 0.35 and the two values "
                           "above it are printed in full; green boxes mark the diagonal",
        },
        header=["evaluated_pair", "fitted_theta", "value"],
        rows=csv_rows,
    )
    print_caption("F7_cross_regret", experiment=experiment, comparing=comparing, config=config)
    return png


# --------------------------------------------------------------------------------------------------
# F8 -- confidence scores and abstention
# --------------------------------------------------------------------------------------------------

def build_f8() -> Path:
    """CN: **F8** —— 四类无标签置信度的 AUROC，以及放弃一半顶点后的误差。EN: **F8** -- AUROC of four
        label-free confidence scores, and the error after abstaining on half the vertices."""
    rows, _ = load_records(OUT / "faust_abstention" / "records.json")
    scores = ("cost_margin", "dual_margin", "rel_cost_margin", "softmax_margin")
    thetas = ("relaxed_lp", "row_softmax", "uniform")
    auc = {(t, s): [] for t in thetas for s in scores}
    cov = {(t, s, q): [] for t in thetas for s in scores for q in ("cov0.5", "cov1")}
    for row in rows:
        theta, score, kind = str(row["arm"]).split(":")
        if kind == "auc":
            auc[(theta, score)].append(float(row["metrics"]["auc"]))
        else:
            cov[(theta, score, kind)].append(float(row["metrics"]["miss_5pct"]))
    got = {
        "relaxed_lp/softmax_margin/auc": _mean(auc[("relaxed_lp", "softmax_margin")]),
        "relaxed_lp/dual_margin/auc": _mean(auc[("relaxed_lp", "dual_margin")]),
        "relaxed_lp/cost_margin/auc": _mean(auc[("relaxed_lp", "cost_margin")]),
        "relaxed_lp/softmax_margin/cov0.5": _mean(cov[("relaxed_lp", "softmax_margin", "cov0.5")]),
        "relaxed_lp/softmax_margin/cov1": _mean(cov[("relaxed_lp", "softmax_margin", "cov1")]),
    }
    check_values("F8", got, EXPECTED["F8"], tol=2e-3)

    fig, (ax_a, ax_c) = plt.subplots(1, 2, figsize=(9.6, 3.8), gridspec_kw={"width_ratios": [1.25, 1.0]})
    apply_style()
    xs = np.arange(len(scores))
    offsets = {"relaxed_lp": -0.18, "row_softmax": 0.0, "uniform": 0.18}
    names = {"relaxed_lp": "inverse-OT θ", "row_softmax": "supervised θ", "uniform": "equal weights"}
    for theta in thetas:
        values = [_mean(auc[(theta, s)]) for s in scores]
        ax_a.plot(xs + offsets[theta], values, ls="none", color=colour(theta), marker=marker(theta),
                  label=names[theta], zorder=3)
        if theta != "relaxed_lp":
            continue
        # CN: 只标一组，其余两组靠高度和颜色读，避免重复数值叠在一起。
        # EN: label one series only; the other two are read from height and colour, so numbers do not collide.
        for x, value in zip(xs + offsets[theta], values):
            ax_a.annotate(f"{value:.3f}", xy=(x, value), xytext=(0, 7), textcoords="offset points",
                          ha="center", fontsize=6.5, color=colour(theta))
    ax_a.axhline(0.5, color="#9a9a9a", lw=0.8, ls=(0, (3, 2)), zorder=1)
    ax_a.annotate("chance", xy=(xs[-1] + 0.30, 0.5), xytext=(2, 4), textcoords="offset points",
                  fontsize=7, color="#8c8c8c")
    ax_a.set_xticks(xs)
    ax_a.set_xticklabels([s.replace("_", " ") for s in scores], fontsize=8)
    ax_a.set_xlim(-0.5, len(scores) - 0.35)
    ax_a.set_ylim(0.44, 0.68)
    ax_a.set_ylabel("AUROC of “this vertex is correct”")
    ax_a.set_title("does the score rank erroneous vertices lower?", fontsize=9, loc="left", pad=4)
    ax_a.legend(loc="upper left", fontsize=7.5, ncol=1)

    ys = np.arange(len(thetas))
    for y, theta in zip(ys, thetas):
        start = _mean(cov[(theta, "softmax_margin", "cov1")])
        end = _mean(cov[(theta, "softmax_margin", "cov0.5")])
        ax_c.plot([start, end], [y, y], color=colour(theta), lw=1.0, zorder=2)
        ax_c.plot([start], [y], marker=marker(theta), color=colour(theta), ls="none", markersize=5, zorder=3)
        ax_c.plot([end], [y], marker=marker(theta), ls="none", markersize=5, zorder=3,
                  markerfacecolor="white", markeredgecolor=colour(theta), markeredgewidth=1.1)
        # CN: 每个点上方直接写它自己的值；不写百分比、不画箭头。
        # EN: each point carries its own value above it: no percentage, no arrow.
        ax_c.annotate(f"{start:.3f}", xy=(start, y), xytext=(0, 8), textcoords="offset points",
                      ha="center", fontsize=7, color=colour(theta))
        ax_c.annotate(f"{end:.3f}", xy=(end, y), xytext=(0, 8), textcoords="offset points",
                      ha="center", fontsize=7, color=colour(theta))
    ax_c.set_yticks(ys)
    ax_c.set_yticklabels([names[t] for t in thetas], fontsize=8)
    ax_c.set_xlim(0.12, 0.45)
    ax_c.set_ylim(-0.6, len(thetas) - 0.4)
    ax_c.set_xlabel("miss@5 % on the vertices that are kept")
    ax_c.set_title("error on the kept vertices (filled = all, hollow = half)", fontsize=9, loc="left", pad=4)
    ax_c.tick_params(length=3)
    for side in ("top", "right"):
        ax_c.spines[side].set_visible(False)
    ax_a.tick_params(length=3)
    for side in ("top", "right"):
        ax_a.spines[side].set_visible(False)

    experiment = (
        "Each vertex of a held-out FAUST pair is given a label-free confidence from the n×n cost matrix, "
        "then the confidence is ranked against the true per-vertex geodesic error on eight held-out pairs.")
    comparing = (
        "Left: four scores — raw cost margin, the OT dual certificate, the margin divided by the runner-up "
        "cost, and the softmax probability margin — under three weightings. Right: the error on the vertices "
        "that survive when the least confident half is dropped, for the softmax margin.")
    config = (
        "MPI-FAUST, n = 4000 vertices per pair, eight held-out pairs, K = 13. AUROC here is the probability "
        "that a correctly matched vertex is ranked more confident than a wrongly matched one, so 0.5 is "
        "chance; the softmax temperature is 3e-3; only two coverage levels (100 % and 50 %) were recorded.")
    add_title(fig, "Confidence from the cost matrix",
              "AUROC on the true vertex error, and the error on the confident half",
              axes_left=0.09)
    add_footer(fig, experiment=experiment, comparing=comparing, config=config, axes_left=0.09, labelled=False)
    fig.subplots_adjust(left=0.09, right=0.99, top=0.80, bottom=0.22, wspace=0.38)
    png = save(fig, "F8_confidence_scores")

    csv_rows = [[theta, score, "auc", f"{_mean(auc[(theta, score)]):.4f}"]
                for theta in thetas for score in scores]
    csv_rows += [[theta, score, kind, f"{_mean(cov[(theta, score, kind)]):.4f}"]
                 for theta in thetas for score in scores for kind in ("cov0.5", "cov1")]
    write_metadata(
        "F8_confidence_scores",
        caption={
            "axes": "left: x = confidence score, y = AUROC (0.5 is chance, higher means correct vertices "
                    "outrank wrong ones); right: x = miss@5 % among the kept vertices, filled dot = all "
                    "vertices kept, hollow dot = only the most confident half kept.",
            "instance": "MPI-FAUST, n = 4000 vertices, eight held-out pairs, K = 13, τ = 3e-3",
            "methods": "cost margin, OT dual margin, relative cost margin, softmax margin; three θ "
                       "(inverse-OT, row-softmax, equal weights). Keeping a subset and scoring it only on the "
                       "kept vertices is what the right panel measures.",
            "config": config,
            "sources": "out/faust_abstention/records.json (288 rows: 3 θ × 4 scores × 3 readouts × 8 pairs)",
            "conventions": "AUROC is per pair and averaged; the dual-margin AUC moves by 0.002–0.006 "
                           "between reruns because the spectral basis is not bit-reproducible",
        },
        header=["theta", "score", "readout", "value"],
        rows=csv_rows,
    )
    print_caption("F8_confidence_scores", experiment=experiment, comparing=comparing, config=config)
    return png


# --------------------------------------------------------------------------------------------------
# F9 -- deformation transfer curve
# --------------------------------------------------------------------------------------------------

def build_f9() -> Path:
    """CN: **F9** —— 固定 θ 在受控变形下的误差曲线。EN: **F9** -- error of a fixed θ under a controlled
        deformation sweep."""
    rows, _ = load_records(OUT / "faust_deformation" / "records.json")
    sigmas = ("sigma0", "sigma0.25", "sigma0.5", "sigma0.75", "sigma1")
    arms = ("learned", "uniform_theta", "uniform_late_hks", "position_only")
    values = {(arm, s): [float(r["metrics"]["miss_5pct"]) for r in rows
                         if r["arm"] == arm and r["geometry"] == s] for arm in arms for s in sigmas}
    got = {"learned/sigma0": _mean(values[("learned", "sigma0")]),
           "learned/sigma1": _mean(values[("learned", "sigma1")]),
           "uniform_theta/sigma1": _mean(values[("uniform_theta", "sigma1")])}
    check_values("F9", got, EXPECTED["F9"], tol=2e-3)

    fig, ax = plt.subplots(figsize=(7.4, 4.1))
    apply_style()
    xs = np.asarray([0.0, 0.25, 0.5, 0.75, 1.0])
    labels = {"learned": "learned θ", "uniform_theta": "equal weights",
              "uniform_late_hks": "late-HKS only", "position_only": "position only"}
    ends: list[float] = []
    for arm in arms:
        ys = np.asarray([_mean(values[(arm, s)]) for s in sigmas])
        ax.plot(xs, ys, color=colour(arm), marker=marker(arm), label=labels[arm], zorder=3)
        ends.append(float(ys[-1]))
    # CN: 三个终点都落在 0.69–0.78，标签必须错开；错开的标签用细引线指回真实终点。
    # EN: three endpoints land inside 0.69–0.78, so the labels are pushed apart and joined to the real
    #     endpoint by a thin leader line.
    placed: list[float] = []
    for arm, end in sorted(zip(arms, ends), key=lambda item: item[1]):
        y_text = end
        for previous in placed:
            if abs(y_text - previous) < 0.045:
                y_text = previous + 0.045
        placed.append(y_text)
        ax.plot([1.0, 1.02], [end, y_text], color=colour(arm), lw=0.6, zorder=2)
        ax.text(1.03, y_text, f"{end:.3f}", va="center", fontsize=7.5, color=colour(arm))
    ax.set_xlim(-0.05, 1.14)
    ax.set_ylim(0.10, 0.95)
    ax.set_xticks(xs)
    ax.set_xlabel("twist amplitude σ (fraction of π, applied to the target)")
    ax.set_ylabel("miss@5 %")
    ax.legend(loc="center right", fontsize=8)
    ax.tick_params(length=3)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

    experiment = (
        "One cost is fitted once on an undeformed FAUST pair, then applied unchanged while the target is "
        "twisted about the vertical axis by an increasing angle; vertex identity is preserved throughout.")
    comparing = (
        "The learned θ against three fixed weightings: equal weights, the late-HKS block and position only, "
        "each measured at the same five twist amplitudes.")
    config = (
        "MPI-FAUST, n = 1500 vertices, K = 13, a single pair and one seed; σ = 0, 0.25, 0.5, 0.75, 1 times "
        "π scaled by the normalised height; the θ is not refitted at any σ.")
    add_title(fig, "Accuracy under a twist", "The same θ across five deformation levels", axes_left=0.11)
    add_footer(fig, experiment=experiment, comparing=comparing, config=config, axes_left=0.11, labelled=False)
    fig.subplots_adjust(left=0.11, right=0.985, top=0.84, bottom=0.18)
    png = save(fig, "F9_accuracy_under_a_twist")

    csv_rows = [[arm, f"{xs[i]:.2f}", f"{_mean(values[(arm, s)]):.4f}"]
                for arm in arms for i, s in enumerate(sigmas)]
    write_metadata(
        "F9_accuracy_under_a_twist",
        caption={
            "axes": "x = twist amplitude σ as a fraction of π applied to the target; y = miss@5 %, the share "
                    "of vertices whose geodesic error exceeds 5 % of √area.",
            "instance": "MPI-FAUST, n = 1500, one pair, one seed, K = 13, σ ∈ {0, 0.25, 0.5, 0.75, 1}",
            "methods": "learned θ (fitted once at σ = 0), equal weights, late-HKS only, position only",
            "config": config,
            "sources": "out/faust_deformation/records.json (60 rows: 12 arms × 5 levels, the nn_ rows are "
                       "nearest-neighbour counterparts not plotted here)",
            "conventions": "a twist is not an isometry, so the curve bounds descriptor degradation under a "
                           "smooth distortion, not tolerance to real pose change",
        },
        header=["arm", "sigma", "miss_5pct"],
        rows=csv_rows,
    )
    print_caption("F9_accuracy_under_a_twist", experiment=experiment, comparing=comparing, config=config)
    return png


# --------------------------------------------------------------------------------------------------
# F10 -- selecting inside the near-optimal set
# --------------------------------------------------------------------------------------------------

def build_f10() -> Path:
    """CN: **F10** —— 在近最优集内用验证对挑选 θ。EN: **F10** -- picking a θ inside the near-optimal set
        on validation pairs."""
    rows, _ = load_records(OUT / "faust_select" / "records.json")
    arms = ("uniform", "median", "reference", "selected")
    held = {arm: {} for arm in arms}
    for row in rows:
        if row["geometry"] in PAIR_ORDER and row["arm"] in held:
            held[row["arm"]][row["geometry"]] = float(row["metrics"]["miss_5pct"])
    means = {arm: _mean(list(v.values())) for arm, v in held.items() if v}
    check_values("F10", means, EXPECTED["F10"], tol=2e-3)
    contrasts = {str(r["arm"]): r["metrics"] for r in rows if r["geometry"] == "contrast"}
    selection = next(r["metrics"] for r in rows if r["geometry"] == "selection")

    fig, (ax_p, ax_c) = plt.subplots(1, 2, figsize=(9.6, 3.9), gridspec_kw={"width_ratios": [1.35, 1.0]})
    apply_style()
    labels = {"uniform": "equal weights", "median": "median of the set",
              "reference": "max-margin (shipped)", "selected": "validation-selected"}
    ys = np.arange(len(PAIR_ORDER))
    for arm in arms:
        order = [held[arm][p] for p in PAIR_ORDER if p in held[arm]]
        ax_p.plot(order, ys[:len(order)], ls="none", color=colour(arm), marker=marker(arm),
                  label=labels[arm], zorder=3)
    ax_p.set_yticks(ys)
    ax_p.set_yticklabels([f"Pair {i + 1}" for i in range(len(PAIR_ORDER))], fontsize=7.5)
    ax_p.invert_yaxis()
    ax_p.set_xlabel("miss@5 % on the held-out pairs")
    ax_p.set_title("one dot per held-out pair", fontsize=9, loc="left", pad=4)
    ax_p.legend(loc="upper center", bbox_to_anchor=(0.5, -0.20), ncol=2, fontsize=7.5)
    ax_p.tick_params(length=3)
    for side in ("top", "right"):
        ax_p.spines[side].set_visible(False)

    # CN: 右图改成**逐对差值**：每一对一行，点在 0 左边就表示比出厂 θ 更好；底部两行给出均值与 CI。
    # EN: the right panel is now a per-pair difference plot: one row per pair, a dot left of zero means better
    #     than the shipped θ; the two bottom rows give the means with their intervals.
    deltas = {
        "selected": [held["selected"][p] - held["reference"][p] for p in PAIR_ORDER],
        "median": [held["median"][p] - held["reference"][p] for p in PAIR_ORDER],
    }
    for i in range(len(PAIR_ORDER)):
        ax_c.plot([deltas["median"][i]], [i], marker=marker("median"), color=colour("median"), ls="none",
                  markersize=6, zorder=3)
        ax_c.plot([deltas["selected"][i]], [i], marker=marker("selected"), color=colour("selected"), ls="none",
                  markersize=6, zorder=3)
    ax_c.axvline(0.0, color="#9a9a9a", lw=0.8, ls=(0, (3, 2)), zorder=1)
    ax_c.axhline(len(PAIR_ORDER) - 0.5, color="#dcdcdc", lw=0.8, zorder=1)

    rows = (("selected", "selected", 8.4), ("median", "median", 9.3))
    for key, arm, row in rows:
        met = contrasts[f"{key}_minus_reference"]
        mean, low, high = float(met["mean"]), float(met["ci_low"]), float(met["ci_high"])
        ax_c.plot([low, high], [row, row], color=colour(arm), lw=1.4, zorder=2)
        ax_c.plot([mean], [row], marker=marker(arm), color=colour(arm), ls="none", markersize=6, zorder=3)
        ax_c.annotate(f"{mean:+.3f}   {int(met['wins'])}/{int(met['pairs'])} pairs",
                      xy=(high, row), xytext=(6, 0), textcoords="offset points", va="center",
                      fontsize=7, color=colour(arm))

    ticks = list(range(len(PAIR_ORDER))) + [8.4, 9.3]
    tick_labels = [f"Pair {i + 1}" for i in range(len(PAIR_ORDER))] + ["mean selected", "mean median"]
    ax_c.set_yticks(ticks)
    ax_c.set_yticklabels(tick_labels, fontsize=7.5)
    ax_c.invert_yaxis()
    ax_c.set_ylim(9.9, -0.7)
    ax_c.set_xlabel("Δ miss@5 % relative to the shipped max-margin θ")
    ax_c.set_title("each pair against the shipped θ (left of 0 is better)", fontsize=9, loc="left", pad=4)
    ax_c.tick_params(length=3)
    for side in ("top", "right"):
        ax_c.spines[side].set_visible(False)
    # CN: 右图不再单独放图例：颜色由下方两行均值标签直接指明，左上角留给 Pair 1 的点。
    # EN: no separate legend on the right: the two coloured mean rows name the colours, and the upper left is
    #     where Pair 1's dots live.

    # CN: 与等权的比较放进页脚，右下角留给两行均值注释，避免叠字。
    # EN: the comparison with equal weights moves to the footer; the lower right is needed by the two mean
    #     rows, and three texts there collided.
    against_uniform = contrasts["selected_minus_uniform"]
    experiment = (
        "The shipped solver returns one point of a set of near-equally good weights; here 71 such candidates "
        "are ranked on six validation pairs from the training subjects and only the winner is scored.")
    comparing = (
        "Four rules: equal weights, the coordinate-wise median of the candidate set, the shipped max-margin "
        "point, and the validation-selected point. The right panel repeats this pair by pair as a difference "
        "from the shipped θ, with the two means and their bootstrap intervals underneath; against equal "
        f"weights the selected θ is {abs(float(against_uniform['mean'])):.3f} better on "
        f"{int(against_uniform['wins'])} of {int(against_uniform['pairs'])} pairs.")
    config = (
        "MPI-FAUST, n = 4000 vertices, eight held-out pairs, K = 13; candidates admitted within 1.25× the "
        "reference regret; selection uses six validation pairs only, never the held-out pairs.")
    add_title(fig, "Choosing inside the near-optimal set", "Validation selection and the shipped max-margin weights",
              axes_left=0.11)
    add_footer(fig, experiment=experiment, comparing=comparing, config=config, axes_left=0.11, labelled=False)
    fig.subplots_adjust(left=0.11, right=0.99, top=0.80, bottom=0.30, wspace=0.42)
    png = save(fig, "F10_selection")

    csv_rows = [[arm, p, f"{held[arm][p]:.4f}"] for arm in arms for p in PAIR_ORDER if p in held[arm]]
    csv_rows += [["selected", f"{p}_delta_vs_reference", f"{d:.4f}"]
                 for p, d in zip(PAIR_ORDER, deltas["selected"])]
    csv_rows += [["median", f"{p}_delta_vs_reference", f"{d:.4f}"]
                 for p, d in zip(PAIR_ORDER, deltas["median"])]
    for key in ("selected_minus_reference", "median_minus_reference", "selected_minus_uniform"):
        for stat in ("mean", "ci_low", "ci_high"):
            csv_rows.append([key, stat, f"{float(contrasts[key][stat]):.4f}"])
    csv_rows += [[
        "selection", "candidates", f"{int(selection['candidates'])}"
    ], [
        "selection", "val_reference_rank", f"{int(selection['val_reference_rank'])}"
    ], [
        "selection", "val_spread", f"{float(selection['val_spread']):.4f}"
    ]]
    write_metadata(
        "F10_selection",
        caption={
            "axes": "left: x = miss@5 % on the held-out pairs, y = pair; right: x = miss@5 % difference from "
                    "the shipped θ on that pair, zero marked, with the mean and its 95 % interval as the two "
                    "bottom rows.",
            "instance": "MPI-FAUST, n = 4000, eight held-out pairs, K = 13, 71 admitted candidates",
            "methods": "equal weights, median of the candidate set, the shipped max-margin θ, and the "
                       "validation-selected θ",
            "config": config,
            "sources": "out/faust_select/records.json (8 pairs × 4 rules plus four contrast rows)",
            "conventions": "the contrast rows come from the same record; wins counts pairs where the left "
                           "side is lower; selection uses validation pairs from training subjects only",
        },
        header=["arm", "pair_or_stat", "value"],
        rows=csv_rows,
    )
    print_caption("F10_selection", experiment=experiment, comparing=comparing, config=config)
    return png


# --------------------------------------------------------------------------------------------------
# F12 -- four objectives, one hypothesis class
# --------------------------------------------------------------------------------------------------

def build_f12() -> Path:
    """CN: **F12** —— 同一特征族、不同目标函数。EN: **F12** -- the same hypothesis class under five
        objectives."""
    rows, _ = load_records(OUT / "faust_arms" / "records.json")
    objectives = ("uniform", "entropic_iot", "row_softmax", "rowcol_softmax", "relaxed_lp")
    pairs = [p for p in PAIR_ORDER if any(r["geometry"] == p for r in rows)]
    values = {(obj, p): None for obj in objectives for p in pairs}
    for row in rows:
        if row["arm"] in objectives and row["geometry"] in pairs:
            values[(row["arm"], row["geometry"])] = row["metrics"]
    got = {obj: _mean([float(values[(obj, p)]["miss_5pct"]) for p in pairs]) for obj in objectives}
    check_values("F12", got, EXPECTED["F12"], tol=3e-3)

    fig, (ax5, ax1) = plt.subplots(1, 2, figsize=(9.6, 3.7), sharey=True)
    apply_style()
    xs = np.arange(len(objectives))
    labels = {"uniform": "equal\nweights", "entropic_iot": "entropic\niOT", "row_softmax": "one-sided\nsoftmax",
              "rowcol_softmax": "Sinkhorn\nsoftmax", "relaxed_lp": "regret LP\n(inverse OT)"}
    for ax, metric, name in ((ax5, "miss_5pct", "miss@5 %"), (ax1, "miss_1pct", "miss@1 %")):
        for i, obj in enumerate(objectives):
            values_here = [float(values[(obj, pair)][metric]) for pair in pairs]
            mean = _mean(values_here)
            # CN: 两个留出对各自一个空心点（左右错开 0.06），横线是它们的均值。
            # EN: one hollow dot per held-out pair, nudged apart by 0.06, and a horizontal bar for their mean.
            offsets = np.linspace(-0.06, 0.06, len(values_here))
            for offset, value in zip(offsets, values_here):
                ax.plot([i + offset], [value], marker="o", ls="none", markersize=5,
                        markerfacecolor="white", markeredgecolor=colour(obj), markeredgewidth=1.0, zorder=3)
            ax.plot([i - 0.17, i + 0.17], [mean, mean], color=colour(obj), lw=2.0, solid_capstyle="butt", zorder=4)
            # CN: 均值标在横线**右端之外**，否则会和两个空心点叠在一起。
            # EN: the mean is labelled just past the end of the bar, clear of the two hollow dots.
            ax.annotate(f"{mean:.3f}", xy=(i + 0.19, mean), xytext=(0, 0), textcoords="offset points",
                        ha="left", va="center", fontsize=7, color=colour(obj))
        ax.set_xticks(xs)
        ax.set_xticklabels([labels[o] for o in objectives], fontsize=8)
        ax.set_xlim(-0.55, len(objectives) - 0.45)
        ax.set_ylim(0.05, 0.95)
        ax.set_xlabel("objective")
        ax.set_title(f"{name}: share of vertices beyond the threshold", fontsize=9, loc="left", pad=4)
        ax.tick_params(length=3)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    ax5.set_ylabel("share of evaluated vertices")
    ax1.plot([], [], marker="o", ls="none", markersize=5, markerfacecolor="white", markeredgecolor="#6e6e6e",
             label="one held-out pair (two in the record)")
    ax1.plot([], [], color="#6e6e6e", lw=2.0, label="their mean")
    ax1.legend(loc="upper left", fontsize=7.5, handletextpad=0.4)

    experiment = (
        "One cost family — the same 13 standardised descriptors, one θ on the simplex, one training pair — "
        "is fitted under five different objectives and scored on the same two held-out pairs.")
    comparing = (
        "Equal weights (no learning), the entropic joint master, a one-sided softmax, a Sinkhorn two-sided "
        "softmax and the relaxed regret LP. Left panel: a vertex counts as wrong when its matched partner is "
        "more than 5 % of √area away; right panel uses a 1 % threshold.")
    config = (
        "MPI-FAUST, n = 1500 vertices, K = 13, pairs 050-066 and 056-071 (the two the record kept of eight); "
        "each hollow dot is one pair's value for that θ, the bar is the arithmetic mean over the two pairs; "
        "hyperparameters were chosen on validation pairs from the training subjects.")
    add_title(fig, "Five objectives, one cost family",
              "Held-out error by objective, at two error thresholds", axes_left=0.075)
    add_footer(fig, experiment=experiment, comparing=comparing, config=config, axes_left=0.075, labelled=False)
    fig.subplots_adjust(left=0.075, right=0.985, top=0.80, bottom=0.25, wspace=0.10)
    png = save(fig, "F12_objectives")

    csv_rows = [[obj, p, f"{float(values[(obj, p)]['miss_5pct']):.4f}",
                 f"{float(values[(obj, p)]['miss_1pct']):.4f}"] for obj in objectives for p in pairs]
    write_metadata(
        "F12_objectives",
        caption={
            "axes": "x = objective, y = share of evaluated vertices whose matched partner is farther than the "
                    "threshold from the true one (left 5 %, right 1 %). A hollow dot is one held-out pair, the "
                    "horizontal bar their mean.",
            "instance": "MPI-FAUST, n = 1500, K = 13; two held-out pairs, 050-066 and 056-071",
            "methods": "uniform, entropic iOT, one-sided softmax, Sinkhorn softmax, relaxed regret LP",
            "config": config,
            "sources": "out/faust_arms/records.json (21 rows: five objectives over two pairs plus the "
                       "hyperparameter-selection row)",
            "conventions": "the mean is taken over the two pairs' results for the same θ; it is not a θ fitted "
                           "on both pairs. The source report tabulates eight pairs, the record kept two, so these "
                           "values are not the reported 8-pair means",
        },
        header=["objective", "pair", "miss_5pct", "miss_1pct"],
        rows=csv_rows,
    )
    print_caption("F12_objectives", experiment=experiment, comparing=comparing, config=config)
    return png


# --------------------------------------------------------------------------------------------------
# F13 -- regional weights
# --------------------------------------------------------------------------------------------------

def build_f13() -> Path:
    """CN: **F13** —— 分区 θ：不同身体部位想要不同权重。EN: **F13** -- regional θ: different parts of the
        body want different weights."""
    rows, _ = load_records(OUT / "faust_regional" / "records.json")
    regions = sorted((r for r in rows if r["geometry"] == "positions"), key=lambda r: r["arm"])
    thetas = np.asarray([r["extra"]["theta"] for r in regions], dtype=np.float64)
    l1 = np.asarray([float(r["metrics"]["l1_to_global"]) for r in regions])
    sizes = np.asarray([int(float(r["metrics"]["size"])) for r in regions])
    test = {r["arm"]: float(r["metrics"]["miss_5pct"]) for r in rows if r["geometry"] == "test"}
    check_values("F13", test, EXPECTED["F13"], tol=3e-3)
    spread = next(r["metrics"] for r in rows if r["geometry"] == "spread")

    columns = ("x", "y", "z", "area", "curv") + tuple(f"hks{i}" for i in range(1, thetas.shape[1] - 4))
    fig, (ax_w, ax_b) = plt.subplots(1, 2, figsize=(9.8, 3.9), gridspec_kw={"width_ratios": [1.6, 1.0]})
    apply_style()
    image = ax_w.imshow(thetas, cmap=SEQUENTIAL_CMAP, aspect="auto", vmin=0.0, vmax=float(thetas.max()))
    for i in range(thetas.shape[0]):
        for j in range(thetas.shape[1]):
            if thetas[i, j] >= 0.01:
                ax_w.text(j, i, f"{thetas[i, j]:.2f}", ha="center", va="center", fontsize=6.5,
                          color="white" if thetas[i, j] < 0.6 * thetas.max() else "black")
    ax_w.set_xticks(range(len(columns)))
    ax_w.set_xticklabels(columns, fontsize=7, rotation=45, ha="right")
    ax_w.set_yticks(range(thetas.shape[0]))
    ax_w.set_yticklabels([f"region {i}  (n={sizes[i]})" for i in range(thetas.shape[0])], fontsize=7.5)
    ax_w.set_xlabel("descriptor")
    ax_w.set_title("weight per region (blank = below 0.01)", fontsize=9, loc="left", pad=4)
    ax_w.tick_params(length=0)
    bar = fig.colorbar(image, ax=ax_w, fraction=0.03, pad=0.02)
    bar.set_label("θ", fontsize=7)
    bar.ax.tick_params(labelsize=6.5)

    ys = np.arange(len(regions))
    # CN: 柱长已经表示数值，颜色不再编码大小；统一用 crimson，浅黄端在白底上读不出来。
    # EN: the bar length already carries the value, so colour does not encode magnitude here; a single
    #     crimson keeps the pale end of the ramp (unreadable on white) out of the figure.
    ax_b.barh(ys, l1, color=colour("iot"), height=0.62, zorder=3)
    for y, value in zip(ys, l1):
        ax_b.annotate(f"{value:.2f}", xy=(value, y), xytext=(3, 0), textcoords="offset points",
                      va="center", fontsize=7)
    ax_b.axvline(float(spread["mean"]), color="#9a9a9a", lw=0.8, ls=(0, (3, 2)), zorder=1)
    ax_b.annotate(f"pairwise mean {float(spread['mean']):.2f}", xy=(float(spread["mean"]), len(regions) - 0.35),
                  xytext=(2, 0), textcoords="offset points", fontsize=7, color="#8c8c8c")
    ax_b.set_yticks(ys)
    ax_b.set_yticklabels([f"region {i}" for i in ys], fontsize=7.5)
    ax_b.invert_yaxis()
    ax_b.set_xlim(0.0, float(l1.max()) * 1.25)
    ax_b.set_xlabel("L1 distance to the global θ")
    ax_b.set_title(f"field on held-out pairs: {test['field']:.3f} vs global {test['global']:.3f}",
                   fontsize=9, loc="left", pad=4)
    ax_b.tick_params(length=3)
    for side in ("top", "right"):
        ax_b.spines[side].set_visible(False)

    experiment = (
        "The body is split into six spatial regions by k-means on the standardised coordinates, and one cost "
        "is fitted per region on that region's block of the same training plan.")
    comparing = (
        "Left: the fitted weight of each descriptor, region by region, from one global fit for reference. "
        "Right: how far each regional θ sits from the global θ, and the held-out score of using each row's "
        "own region.")
    config = (
        "MPI-FAUST training pair 0-13, n = 2000 vertices, K = 13, simplex only, one seed; region sizes "
        "279–493 vertices; the naive regional field is compared with the global θ on eight held-out pairs.")
    add_title(fig, "Do body parts want different weights?", "Six regional fits plus one global fit", axes_left=0.075)
    add_footer(fig, experiment=experiment, comparing=comparing, config=config, axes_left=0.075, labelled=False)
    fig.subplots_adjust(left=0.075, right=0.99, top=0.80, bottom=0.23, wspace=0.34)
    png = save(fig, "F13_regional_theta")

    csv_rows = [[f"region{i}", col, f"{thetas[i, j]:.4f}"]
                for i in range(thetas.shape[0]) for j, col in enumerate(columns)]
    csv_rows += [[f"region{i}", "l1_to_global", f"{l1[i]:.4f}"] for i in range(len(regions))]
    csv_rows += [["field", "miss_5pct", f"{test['field']:.4f}"], ["global", "miss_5pct", f"{test['global']:.4f}"]]
    write_metadata(
        "F13_regional_theta",
        caption={
            "axes": "left: rows are spatial regions, columns descriptors, colour and number the fitted weight "
                    "θ; right: L1 distance from each regional θ to the global θ, plus the held-out result.",
            "instance": "MPI-FAUST pair 0-13, n = 2000 vertices, K = 13, six k-means regions, one seed",
            "methods": "one simplex fit per region on that region's sub-block of the identity plan; the "
                       "dashed line is the mean pairwise regional distance",
            "config": config,
            "sources": "out/faust_regional/records.json (six region rows with their θ, one spread row, and "
                       "the global/field rows)",
            "conventions": "weights below 0.01 are left unlabelled to keep the grid readable; the regional "
                           "field is preliminary and has no regional supervised control",
        },
        header=["region", "descriptor_or_stat", "value"],
        rows=csv_rows,
    )
    print_caption("F13_regional_theta", experiment=experiment, comparing=comparing, config=config)
    return png



# --------------------------------------------------------------------------------------------------
# F14 -- where the correspondence is wrong, on the surface
# --------------------------------------------------------------------------------------------------

def build_f14() -> Path:
    """CN: **F14** —— 留出对上的错误顶点，画在人体表面上。EN: **F14** -- the wrongly matched vertices of a
        held-out pair, drawn on the human surface."""
    from experiments.harness.faust import read_ply

    rows, _ = load_records(OUT / "faust_error_points" / "records.json")
    npz_path = OUT / "faust_error_points" / "errors_pair050-066.npz"
    if not npz_path.is_file():
        raise FigureError(f"missing per-vertex errors: {npz_path}")
    payload = np.load(npz_path)
    threshold = float(payload["threshold"])
    index = np.asarray(payload["index"], dtype=np.int64)
    arms = ("position_only", "uniform_theta", "learned")
    errors = {arm: np.asarray(payload[arm], dtype=np.float64) for arm in arms}
    metrics = {row["arm"]: row["metrics"] for row in rows}
    got = {arm: float(metrics[arm]["miss_5pct"]) for arm in arms}
    check_values("F14", got, EXPECTED["F14"], tol=3e-3)

    mesh = read_ply(FAUST_ROOT / "tr_reg_066.ply")
    v = np.asarray(mesh.vertices, dtype=np.float64)
    f = np.asarray(mesh.faces, dtype=np.int64)
    centre = 0.5 * (v.min(axis=0) + v.max(axis=0))
    p = v - centre
    xy, depth = p[:, [0, 1]], p[:, 2]
    order = np.argsort(depth[f].mean(axis=1))
    polys = xy[f[order]]
    visible = depth[index] >= np.percentile(depth[index], 35.0)
    # CN: 只画**错**的顶点：这是图要表达的东西，画对点只会让点云糊成一片。
    # EN: draw only the **wrong** vertices; drawing the correct ones as well just fills the body with dots.
    wrong_mask = {arm: visible & (errors[arm] > threshold) for arm in arms}
    vmax = float(np.percentile(np.concatenate([errors[a][wrong_mask[a]] for a in arms]), 95))

    fig, axes = plt.subplots(1, 3, figsize=(9.6, 4.4))
    apply_style()
    labels = {"position_only": "position only", "uniform_theta": "equal weights", "learned": "learned θ"}
    scatter = None
    for ax, arm in zip(axes, arms):
        ax.add_collection(PolyCollection(polys, facecolors="#e2e2e2", edgecolors="#cfcfcf", linewidths=0.05))
        points = index[wrong_mask[arm]]
        scatter = ax.scatter(xy[points, 0], xy[points, 1], c=errors[arm][wrong_mask[arm]], cmap=SEQUENTIAL_CMAP,
                             vmin=0.0, vmax=vmax, s=8, linewidths=0, zorder=3)
        ax.set_xlim(float(xy[:, 0].min()), float(xy[:, 0].max()))
        ax.set_ylim(float(xy[:, 1].min()), float(xy[:, 1].max()))
        ax.set_aspect("equal")
        ax.axis("off")
        wrong = int(metrics[arm]["wrong_vertices"])
        ax.set_title(f"{labels[arm]}\n{wrong} of {index.size} vertices wrong "
                     f"({100.0 * metrics[arm]['miss_5pct']:.1f} %)", fontsize=8.5, pad=4)
    if scatter is not None:
        bar_ax = fig.add_axes([0.935, 0.28, 0.010, 0.44])
        bar = fig.colorbar(scatter, cax=bar_ax)
        bar.set_label("geodesic error / √area of the target", fontsize=7)
        bar.ax.tick_params(labelsize=6.5)

    experiment = (
        "A held-out FAUST pair is matched once with each of three fixed weightings; every dot is an evaluated "
        "vertex of the second person, placed where that vertex sits on the surface.")
    comparing = (
        "Position only, equal weights on the 13 descriptors, and the weights learned on a different pair of "
        "people. A dot is a vertex whose matched partner landed more than 5 % of √area from the true one, and "
        "its colour is how far it landed.")
    config = (
        "MPI-FAUST pair 050-066, n = 600 evaluated vertices of 6890, K = 13 descriptors; the learned weights "
        "come from pair 000-013 fitted with an exact CPU master; dots are drawn for the front-facing part of "
        "the body only, so counts include vertices that are not visible in this view.")
    add_title(fig, "Wrong vertices on a FAUST pair", "Held-out pair, three weightings, each dot one evaluated vertex",
              axes_left=0.02)
    add_footer(fig, experiment=experiment, comparing=comparing, config=config, axes_left=0.02, axes_right=0.93,
               labelled=False)
    fig.subplots_adjust(left=0.02, right=0.92, top=0.80, bottom=0.10, wspace=0.02)
    png = save(fig, "F14_error_vertices")

    csv_rows = [[arm, "miss_5pct", f"{float(metrics[arm]['miss_5pct']):.4f}"] for arm in arms]
    csv_rows += [[arm, "wrong_vertices", f"{int(metrics[arm]['wrong_vertices'])}"] for arm in arms]
    csv_rows += [[arm, "mean", f"{float(metrics[arm]['mean']):.4f}"] for arm in arms]
    write_metadata(
        "F14_error_vertices",
        caption={
            "axes": "no axes: an orthographic front view of the target surface; each dot is one evaluated "
                    "vertex, coloured by the geodesic distance from its matched partner to the true one, in "
                    "units of √area.",
            "instance": "MPI-FAUST pair 050-066, n = 600 of 6890 vertices, K = 13 descriptors, one draw",
            "methods": "position_only, uniform_theta, learned (simplex fit on pair 000-013, exact CPU master)",
            "config": config,
            "sources": "out/faust_error_points/records.json and errors_pair050-066.npz, written by "
                       "studies/iot/real/faust/error_points.py; both are git-ignored and regenerated in ~1 min",
            "conventions": "dots are drawn only for the front-facing 65 % of the evaluated vertices, so the "
                           "counts in the titles exceed what is visible; grey is the rest of the surface, not "
                           "a data value",
        },
        header=["arm", "metric", "value"],
        rows=csv_rows,
    )
    print_caption("F14_error_vertices", experiment=experiment, comparing=comparing, config=config)
    return png


HEATMAP_PAIRS: tuple[str, ...] = ("pair050-066", "pair056-071")


def _f6_inputs_present() -> bool:
    """
    CN: F6 需要 `heatmaps.py` 落盘的逐顶点误差数组。没有就**跳过**，不重算实验、不伪造数据。
    EN: F6 needs the per-vertex error arrays written by `heatmaps.py`. Without them the figure is **skipped**:
        the experiment is not recomputed here and no values are invented.
    """
    return all((OUT / "faust_heatmaps" / f"errors_{pair}.npz").is_file() for pair in HEATMAP_PAIRS)


def main() -> int:
    """CN: 入口。EN: entry point."""
    apply_style()
    for build in (build_f6, build_f7, build_f8, build_f9, build_f10, build_f12, build_f13, build_f14):
        if build is build_f6 and not _f6_inputs_present():
            print("skipped F6_mesh_error_maps: no per-vertex error arrays on disk "
                  f"(expected {', '.join(f'errors_{p}.npz' for p in HEATMAP_PAIRS)})")
            continue
        print(f"written: {build()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
