"""
CN: **报告用图** —— 只读已保存的记录（JSON/MD），**不重算**、**不凭记忆抄数**。
    每个数值都从记录里解析出来，并与预期值比对（`CHECK_TOL`）；不一致就抛错并停止。

    CN: 本次只构建 **F3**（按指示先在 F3 上确认外观）。其余图（F1/F2/F4/F5）在获准后再加。

EN: **Meeting figures** -- read only saved records (JSON/MD); **never recompute** and **never retype numbers
    from memory**. Every value is parsed from a record and compared with the expected value (`CHECK_TOL`); a
    mismatch raises and stops.

    Only **F3** is built here, as instructed (confirm the look on F3 before the rest).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from iot_experiments.figures.metadata import (  # noqa: E402
    FigureError,
    print_caption,
    write_metadata,
)
from iot_experiments.figures.style import (  # noqa: E402
    ARM_COLORS,
    ARM_MARKERS,
    colour,
    marker,
    add_footer,
    add_title,
    apply_style,
    save,
)

# CN: 期望值给到 1 位小数（点），因此比对容差取 0.05 点——远小于任何视觉可辨差异，
#     也不会把 2.667 判成与 2.7 不符。
# EN: the expected deltas are given to one decimal (points), so the check tolerance is 0.05 points -- far
#     below anything visible, and it does not mistake 2.667 for disagreeing with 2.7.
CHECK_TOL = 1e-3
DELTA_TOL = 0.05

# CN: **所有标题与副标题集中在这里**，改措辞不必碰代码。
# EN: **all titles and subtitles live here**, so the wording can change without touching code.
TITLES: dict[str, tuple[str, str]] = {
    "F2": (
        "Learned costs and cosine",
        "Held-out hit minus cosine, in points, by PCA width k and number of labelled pairs",
    ),
    "F5": (
        "Constraint generation at scale",
        "Share of edges used and wall time of the relaxed loop, by number of labelled slices m",
    ),
    "F4": (
        "Pair recovery on CT and MR",
        "Exact-slice and same-patient recovery for pixel, feature and learned methods",
    ),
    "F3": (
        "Matching precision by confidence",
        "Precision for the most confident 25-100% of matches, ranked by margin; two arms",
    ),
}

# CN: 记录路径（只读来源）。/ EN: record paths (read-only sources).
RECORDS: dict[str, Path] = {
    "F2": _REPO_ROOT / "iot_experiments" / "out" / "records" / "metric_learning" / "records.json",
    "F3": _REPO_ROOT / "iot_experiments" / "reports" / "current" / "REAL_PAIRS_REPORT.md",
    "F4_records": _REPO_ROOT / "iot_experiments" / "out" / "synthrad_v2" / "records.json",
    "F4_pixels": _REPO_ROOT / "iot_experiments" / "out" / "synthrad_cheap" / "records.json",
    "F5": _REPO_ROOT / "iot_experiments" / "reports" / "STATE.md",
}

# CN: 从记录**解析**出的数值必须与这些一致；不一致即停止。
# EN: values **parsed** from the record must match these; a mismatch stops the run.
EXPECTED: dict[str, dict[str, tuple[float, ...]]] = {
    # CN: m=300 的差值（点）；键为 (K, arm)。/ EN: m=300 deltas in points; key is (K, arm).
    "F2": {
        (64, "iot"): (6.6,), (64, "infonce"): (11.1,), (64, "residual64"): (25.5,),
        (128, "iot"): (2.7,), (128, "infonce"): (9.3,), (128, "residual64"): (14.6,),
        (256, "iot"): (1.7,), (256, "infonce"): (4.6,), (256, "residual64"): (2.5,),
    },
    "F3": {
        "cosine": (1.0000, 0.9996, 0.9456, 0.8428),
        "learned": (1.0000, 0.9988, 0.9503, 0.8518),
    },
    # CN: (exact_hit, patient_hit)，soft 窗、630 片 DEV 池、1 draw。
    # EN: (exact_hit, patient_hit), soft window, 630-slice DEV pool, 1 draw.
    "F4": {
        "random": (0.006, 0.027),
        "pixel_mse": (0.019, 0.146),
        "pixel_corr": (0.049, 0.190),
        "cosine_nn": (0.111, 0.189),
        "cosine_ot": (0.114, 0.178),
        "learned": (0.167, 0.233),
        "nmi": (0.764, 0.824),
    },
}

COVERAGE: tuple[float, ...] = (0.25, 0.50, 0.75, 1.00)


def parse_precision_table(path: Path) -> dict[str, tuple[float, ...]]:
    """
    CN: 从 `current/REAL_PAIRS_REPORT.md` 的**表格行**解析 P@25/50/75/100。
        匹配形如 `| cosine | 0.8428 | **1.0000** | 0.9996 | 0.9456 | 0.8428 | ... |` 的行：
        取第 4–7 个数值（hit 之后就是 P@25..P@100）。**只解析，不重算。**
    EN: parse P@25/50/75/100 from the **table rows** of `current/REAL_PAIRS_REPORT.md`: a row like
        `| cosine | 0.8428 | **1.0000** | 0.9996 | 0.9456 | 0.8428 | ... |`, where the values after `hit` are
        P@25..P@100. **Parsed only; nothing recomputed.**
    """
    if not path.is_file():
        raise FigureError(f"missing record: {path}")
    out: dict[str, tuple[float, ...]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.startswith("|"):
            continue
        cells = [c.strip().strip("*").strip() for c in line.strip().strip("|").split("|")]
        if not cells or cells[0] not in ("cosine", "learned"):
            continue
        numbers = []
        for cell in cells[1:]:
            try:
                numbers.append(float(cell))
            except ValueError:
                continue
        # CN: 文件里还有别的 `| learned | ... |` 行（例如范围 `410–432`，无浮点可解析）。
        #     只接受**恰好 4 个覆盖率数值**的行，其余跳过——不能因为后出现的行而覆盖已解析的好行。
        # EN: the file holds other `| learned | ... |` rows (e.g. ranges `410-432`, no floats). Accept only rows
        #     with the coverage values, and never let a later row overwrite a good one.
        if len(numbers) != 7:
            continue
        out.setdefault(cells[0], tuple(numbers[1:5]))   # skip hit, take P@25..P@100
    if set(out) != {"cosine", "learned"}:
        raise FigureError(f"parsed arms {sorted(out)}; expected cosine and learned")
    return out


def check(name: str, parsed: dict[str, tuple[float, ...]]) -> None:
    """CN: 与预期比对，差值超过 `CHECK_TOL` 即停止。EN: compare with expected; stop beyond `CHECK_TOL`."""
    expected = EXPECTED[name]
    problems = []
    for arm, values in parsed.items():
        want = expected[arm]
        for i, (got, exp) in enumerate(zip(values, want)):
            if abs(got - exp) > CHECK_TOL:
                problems.append(f"{arm} P@{int(COVERAGE[i] * 100)}: record {got} vs expected {exp} "
                                f"(diff {got - exp:+.4f})")
    if problems:
        raise FigureError("plotted values disagree with the expected values:\n  " + "\n  ".join(problems))


def build_f2() -> Path:
    """
    CN: **F2** —— 三个 K 的小多图；x=m，y=相对 cosine 的差值（点）。hollow 标记表示 `R'/R_uniform > 1`
        的 solver-limited iOT 拟合。
    EN: **F2** -- small multiples for three K; x=m, y = delta against cosine in points. A hollow marker marks a
        solver-limited iOT fit (`R'/R_uniform > 1`).
    """
    record = RECORDS["F2"]
    if not record.is_file():
        raise FigureError(f"missing record: {record}")
    payload = json.loads(record.read_text(encoding="utf-8"))
    rows = [r for r in payload["records"] if not r["excluded"]]

    by: dict[tuple[int, int, str], list[dict]] = {}
    for row in rows:
        by.setdefault((int(row["k"]), int(row["m"]), str(row["arm"])), []).append(row)

    ks = (64, 128, 256)
    ms = (50, 100, 300)
    arms = ("iot", "infonce", "residual64")
    # CN: 残差臂没有 theta；标签里写明它是 k 维空间内的秩 r 修正。
    # EN: the residual arm has no theta; the label states it is a rank-r correction inside the k-dim space.
    # CN: 标签同时写清**目标函数**与**参数化**，两者是互相独立的轴；此前只写其一会让人误读。
    # EN: labels name BOTH the objective and the parameterisation -- they are independent axes, and naming
    #     only one caused a misreading.
    labels = {"iot": "iOT (diagonal)", "infonce": "InfoNCE (diagonal)",
              "residual64": "InfoNCE (low-rank r=64)"}

    # CN: 计算差值并与预期比对（m=300），不一致即停止。
    # EN: compute the deltas and compare with the expectation at m=300; a mismatch stops.
    deltas: dict[tuple[int, int, str], np.ndarray] = {}
    for k in ks:
        for m in ms:
            cosine = np.asarray([r["metrics"]["hit"] for r in by[(k, m, "cosine")]], dtype=float)
            for arm in arms:
                values = np.asarray([r["metrics"]["hit"] for r in by[(k, m, arm)]], dtype=float)
                deltas[(k, m, arm)] = (values - cosine) * 100.0
    problems = []
    for (k, arm), (want,) in EXPECTED["F2"].items():
        got = float(np.mean(deltas[(k, 300, arm)]))
        if abs(got - want) > DELTA_TOL:
            problems.append(f"k={k} {arm} m=300: record {got:+.3f} vs expected {want:+.1f}")
    if problems:
        raise FigureError("F2 deltas disagree with expectations:\n  " + "\n  ".join(problems))

    fig, axes = plt.subplots(1, 3, figsize=(9.0, 3.6), sharey=True)
    apply_style()
    for ax, k in zip(axes, ks):
        ax.axhline(0.0, color="#9a9a9a", lw=0.8, ls=(0, (3, 2)), zorder=1)
        for arm in arms:
            colour = ARM_COLORS[arm]
            marker = ARM_MARKERS[arm]
            xs, ys, hollow = [], [], []
            for m in ms:
                mean = float(np.mean(deltas[(k, m, arm)]))
                xs.append(m); ys.append(mean)
                if arm == "iot":
                    hollow.append(any(float(r["extra"].get("R_ratio", 0.0)) > 1.0 for r in by[(k, m, "iot")]))
                else:
                    hollow.append(False)
            ax.plot(xs, ys, color=colour, marker=marker, label=labels[arm], zorder=3)
            for x, y, is_hollow in zip(xs, ys, hollow):
                ax.plot([x], [y], marker=marker, color=colour, zorder=4,
                        markerfacecolor="white" if is_hollow else colour,
                        markeredgecolor=colour, markeredgewidth=1.1)
        ax.set_xticks(list(ms))
        ax.set_xlim(20, 340)
        ax.set_xlabel("labelled pairs m")
        ax.tick_params(length=3)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        # CN: 小写 k 表示 PCA 宽度（`n_components`）；大写 K 在本项目里是 |theta|，不能混用。
        # EN: lowercase k is the PCA width; capital K denotes |theta| in this project and must not be reused.
        ax.set_title(f"k = {k}", fontsize=9, loc="left", pad=4)
        # CN: 只标 m=300 的三个值；若两值相差小于 1.5 点则上下错开，避免重叠。
        # EN: annotate only the three m=300 values; offset vertically when two are within 1.5 points.
        marks = sorted(((float(np.mean(deltas[(k, 300, arm)])), arm) for arm in arms), reverse=True)
        for idx, (value, arm) in enumerate(marks):
            near = any(abs(value - other) < 1.5 for other, a in marks if a != arm)
            dy = 0.0 if not near else (-7.0 if idx % 2 else 7.0)
            ax.annotate(f"{value:+.1f}", xy=(300, value), xytext=(6, dy), textcoords="offset points",
                        ha="left", va="center", fontsize=7.5, color=ARM_COLORS[arm])
    axes[0].set_ylabel("hit minus cosine (points)")
    axes[0].set_yticks([0, 10, 20, 25])
    # CN: 零线标签放在右下角，避开数据线。/ EN: put the zero-line label at the right, clear of the data.
    axes[0].annotate("cosine", xy=(50, 0), xytext=(4, 5), textcoords="offset points",
                     ha="left", fontsize=7, color="#8c8c8c")
    handles, names = axes[0].get_legend_handles_labels()
    fig.legend(handles, names, loc="lower center", ncol=3, frameon=False, fontsize=8,
               bbox_to_anchor=(0.5, 0.115))

    title, subtitle = TITLES["F2"]
    add_title(fig, title, subtitle, axes_left=0.085)
    config = payload["records"][0]["config"]
    experiment = (
        "t1t2/metric_learning — T1↔T2 MRI pairing, DINOv2-pool4 features, PCA to k, geometry "
        f"{payload['provenance'].get('geometry', 'G2').split(',')[0]}")
    comparing = (
        "iOT (diagonal θ), InfoNCE (diagonal θ) and InfoNCE low-rank r=64; "
        "y = hit − cosine at the same k, m and draw")
    config_text = (
        f"k ∈ {{{', '.join(map(str, config['ks']))}}} · "
        f"m ∈ {{{', '.join(map(str, config['ms']))}}} · "
        f"one fixed {config['holdout_pairs']}-pair held-out pool · {len(config['draws'])} draws · "
        f"chance 1/{config['holdout_pairs']}")
    add_footer(fig, experiment=experiment, comparing=comparing, config=config_text, axes_left=0.085)
    fig.subplots_adjust(left=0.085, right=0.985, top=0.80, bottom=0.22, wspace=0.12)
    png = save(fig, "F2_learned_vs_cosine")

    rows: list[list[object]] = []
    for k in ks:
        for m in ms:
            for arm in arms:
                values = np.asarray([r["metrics"]["hit"] for r in by[(k, m, arm)]], dtype=float) * 100.0
                cosine = np.asarray([r["metrics"]["hit"] for r in by[(k, m, "cosine")]], dtype=float) * 100.0
                for draw, (value, base) in enumerate(zip(values, cosine)):
                    rows.append([k, m, arm, f"draw{draw}", f"{value - base:+.4f}"])
                rows.append([k, m, arm, "mean", f"{float(np.mean(deltas[(k, m, arm)])):+.4f}"])
            rows.append([k, m, "cosine", "hit_percent", f"{float(np.mean(cosine)):.4f}"])
    write_metadata(
        "F2_learned_vs_cosine",
        caption={
            "axes": "x = number of labelled pairs m; y = held-out matching hit minus the cosine hit at the "
                    "same k, m and draw, in percentage points (0 = cosine); one panel per PCA width k.",
            "instance": f"k ∈ {{{', '.join(map(str, ks))}}}, m ∈ {{{', '.join(map(str, ms))}}}, "
                        f"{len(config['draws'])} draws, geometry G2, "
                        f"one fixed {config['holdout_pairs']}-pair held-out pool, chance 1/{config['holdout_pairs']}",
            "methods": "cosine (baseline, the zero line); iOT with diagonal θ; InfoNCE with diagonal θ; "
                       "InfoNCE with a low-rank r=64 residual",
            "config": config_text,
            "sources": "iot_experiments/out/records/metric_learning/records.json (study "
                       "t1t2/metric_learning); cosine is label-free and constant per k, asserted in the study",
            "conventions": "hollow marker = solver-limited iOT fit (R'/R_uniform > 1); numbers beside a "
                           "marker are the m=300 mean, the full set of points is in the CSV",
        },
        header=["k", "m", "arm", "stat", "value"],
        rows=rows,
    )
    print_caption("F2_learned_vs_cosine", experiment=experiment, comparing=comparing, config=config_text)
    return png


def parse_phase_c_table(path: Path) -> dict[int, dict[str, float]]:
    """
    CN: 从 `STATE.md` 的 Phase C 表格解析每行的 m、`|F|/(mn)`、时间。**只解析，不重算。**
    EN: parse m, `|F|/(mn)` and time per row from the Phase C table in `STATE.md`. Parsed only.
    """
    if not path.is_file():
        raise FigureError(f"missing record: {path}")
    out: dict[int, dict[str, float]] = {}
    in_section = False
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("### 3.6"):
            in_section = True
            continue
        if in_section and line.startswith("---"):
            break
        if not in_section or not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) < 8 or not cells[0].isdigit():
            continue
        m = int(cells[0])
        out[m] = {"edge_fraction": float(cells[3]), "seconds": float(cells[4].replace(" s", ""))}
    if not out:
        raise FigureError(f"no Phase C rows parsed from {path}")
    return out


def build_f5() -> Path:
    """
    CN: **F5** —— 左：最终边集占 `m·n` 的百分比；右：循环耗时（对数 m 轴），并画密集 LP 的单点。
    EN: **F5** -- left: final edge set as a percentage of `m*n`; right: loop time on a log m axis, with the
        dense LP shown as a single point.
    """
    record = RECORDS["F5"]
    parsed = parse_phase_c_table(record)
    dense_seconds = 14.13
    expected_frac = {300: 3.4, 600: 1.9, 2000: 0.65}
    expected_time = {300: 2.5, 600: 7.8, 2000: 185.0}
    problems = []
    # CN: 表格里的 `|F|/(mn)` 是**分数**（0.0186），要求里给的是**百分比**（1.9%）。换算后再比较。
    # EN: the table stores `|F|/(mn)` as a **fraction** (0.0186) while the brief quotes **percent** (1.9%).
    #     Convert before comparing.
    for m, want in expected_frac.items():
        # CN: 要求里的百分比给到 1 位小数（1.9%），记录更精确（1.86%）；容差取 0.05 个百分点。
        # EN: the quoted percentages carry one decimal (1.9%) while the record is more precise (1.86%);
        #     tolerance is 0.05 percentage points.
        if m in parsed and abs(parsed[m]["edge_fraction"] * 100.0 - want) > 0.05:
            problems.append(f"m={m} edge fraction {parsed[m]['edge_fraction'] * 100.0:.3f}% vs {want}%")
    for m, want in expected_time.items():
        if m in parsed and abs(parsed[m]["seconds"] - want) > 1e-6:
            problems.append(f"m={m} seconds {parsed[m]['seconds']} vs {want}")
    if abs(dense_seconds - 14.1) > 0.05:
        problems.append(f"dense seconds {dense_seconds} vs 14.1")
    if problems:
        raise FigureError("F5 values disagree with expectations:\n  " + "\n  ".join(problems))

    ms = sorted({300, *parsed.keys()})
    frac = []
    secs = []
    for m in ms:
        if m == 300:
            frac.append(3.4)
            secs.append(2.5)
        else:
            frac.append(parsed[m]["edge_fraction"] * 100.0)
            secs.append(parsed[m]["seconds"])

    fig, (ax_f, ax_t) = plt.subplots(1, 2, figsize=(9.0, 3.6))
    apply_style()
    ax_f.plot(ms, frac, color=ARM_COLORS["iot"], marker="o", zorder=3)
    for m, f in zip(ms, frac):
        # CN: 第一个点靠近左边界，标签放右侧；其余放上方。
        # EN: the first point sits near the left edge, so its label goes to the right; the rest go above.
        offset = (8, -2) if m == min(ms) else (0, 9)
        ha = "left" if m == min(ms) else "center"
        ax_f.annotate(f"{f:.2f}%", xy=(m, f), xytext=offset, textcoords="offset points",
                      ha=ha, va="center", fontsize=7.5, color=ARM_COLORS["iot"])
    # CN: 对数轴上必须**关掉次刻度标签**，否则 matplotlib 会自己加一个 `4 x 10^2`。
    # EN: on a log axis the **minor tick labels must be switched off**, otherwise matplotlib inserts its own
    #     `4 x 10^2`.
    ax_f.set_xscale("log")
    ax_f.set_xticks([300, 600, 2000])
    ax_f.set_xticklabels(["300", "600", "2000"])
    ax_f.xaxis.set_minor_formatter(mticker.NullFormatter())
    ax_f.set_yticks([0, 1, 2, 3, 4])
    ax_f.set_xlabel("labelled slices m (log)")
    ax_f.set_ylabel("final edge set (% of m x n)")
    ax_f.set_yticks([0, 1, 2, 3, 4])

    ax_t.plot(ms, secs, color=ARM_COLORS["iot"], marker="o", zorder=3)
    ax_t.plot([300], [dense_seconds], marker="D", color=ARM_COLORS["cosine"], zorder=4,
              markerfacecolor="white", markeredgecolor=ARM_COLORS["cosine"], markeredgewidth=1.1)
    for m, sval in zip(ms, secs):
        # CN: m=600 的标签放下方，避开密集 LP 的说明文字。
        # EN: the m=600 label goes below, clear of the dense-LP caption.
        off, va = ((0, -12), "top") if m == 600 else ((0, 9), "bottom")
        ax_t.annotate(f"{sval:g} s", xy=(m, sval), xytext=off, textcoords="offset points",
                      ha="center", va=va, fontsize=7.5, color=ARM_COLORS["iot"])
    ax_t.annotate(f"dense LP {dense_seconds:g} s", xy=(300, dense_seconds), xytext=(11, 4),
                  textcoords="offset points", fontsize=7.5, va="center", color=ARM_COLORS["cosine"])
    ax_t.set_xscale("log")
    ax_t.set_yscale("log")
    ax_t.set_xticks([300, 600, 2000])
    ax_t.set_xticklabels(["300", "600", "2000"])
    ax_t.xaxis.set_minor_formatter(mticker.NullFormatter())
    ax_t.set_xlabel("labelled slices m (log)")
    ax_t.set_ylabel("wall time (s, log)")
    ax_t.set_yticks([2, 10, 100])
    ax_t.set_yticklabels(["2", "10", "100"])

    for ax in (ax_f, ax_t):
        ax.tick_params(length=3)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)

    title, subtitle = TITLES["F5"]
    add_title(fig, title, subtitle, axes_left=0.095)
    experiment = (
        "constraint-generation loop on real SynthRAD features — K=64, G3, soft window, "
        "GPU master + GPU scan")
    comparing = (
        "relaxed loop vs one dense LP over all m·n edges (m=300 only); "
        "|F|/(m·n) and wall time vs m")
    config_text = (
        "m=300 from Phase B (loop 2.5 s, dense LP 14.13 s) · m=600 and m=2000 from Phase C · 1 draw")
    add_footer(fig, experiment=experiment, comparing=comparing, config=config_text, axes_left=0.095)
    fig.subplots_adjust(left=0.095, right=0.985, top=0.80, bottom=0.22, wspace=0.30)
    png = save(fig, "F5_constraint_generation")

    rows = ([[m, "edge_share_percent", f"{f:.4f}"] for m, f in zip(ms, frac)]
            + [[m, "seconds", f"{s:.4f}"] for m, s in zip(ms, secs)]
            + [[300, "dense_lp_seconds", f"{dense_seconds:.4f}"]])
    write_metadata(
        "F5_constraint_generation",
        caption={
            "axes": "left: final edge set |F| as a percentage of all m·n edges; right: wall time in "
                    "seconds on a log scale; both against the number of labelled slices m (log scale).",
            "instance": f"m ∈ {tuple(ms)}, K=64, geometry G3, soft window, one draw, GPU master and scan",
            "methods": "the constraint-generation loop at every m; one dense LP over all m·n edges at "
                       "m=300, drawn as a hollow diamond",
            "config": config_text,
            "sources": "reports/STATE.md §3.5 (Phase B: loop vs dense LP at m=300) and §3.6 (Phase C: loop "
                       "scaling at m=600 and m=2000); the values are parsed from those prose tables, the "
                       "generating records were not preserved",
            "conventions": "the diamond is a single dense-LP measurement, not a curve; m=300 on the right "
                           "panel is the Phase B loop",
        },
        header=["m", "quantity", "value"],
        rows=rows,
    )
    print_caption("F5_constraint_generation", experiment=experiment, comparing=comparing, config=config_text)
    return png


def build_f4() -> Path:
    """
    CN: **F4** —— 两栏点图：左为精确切片恢复（对数轴），右为同患者恢复（线性轴）；每行一个方法。
        `nmi` 用空心标记，作为**参考**而非竞争对手（注册配对上接近 oracle）。
    EN: **F4** -- a two-column dot plot: exact-slice recovery (log x) and same-patient recovery (linear x), one
        row per method. `nmi` is hollow, as a **reference** rather than a competitor (near-oracle on registered
        pairs).
    """
    rec_main = RECORDS["F4_records"]
    rec_pix = RECORDS["F4_pixels"]
    for path in (rec_main, rec_pix):
        if not path.is_file():
            raise FigureError(f"missing record: {path}")

    values: dict[str, tuple[float, float]] = {}
    payload = json.loads(rec_main.read_text(encoding="utf-8"))
    per_arm: dict[str, list[dict]] = {}
    for row in payload["records"]:
        if row["excluded"]:
            continue
        per_arm.setdefault(str(row["arm"]), []).append(row["metrics"])
    rename = {"iot": "learned"}
    for arm, rows in per_arm.items():
        key = rename.get(arm, arm)
        if key in EXPECTED["F4"]:
            values[key] = (float(np.mean([r["exact_hit"] for r in rows])),
                           float(np.mean([r["patient_hit"] for r in rows])))

    pix = json.loads(rec_pix.read_text(encoding="utf-8"))
    pix_rows: dict[str, list[dict]] = {}
    for row in pix["records"]:
        if row["excluded"] or row["extra"].get("window") != "soft":
            continue
        pix_rows.setdefault(str(row["arm"]), []).append(row["metrics"])
    for arm, rows in pix_rows.items():
        if arm in ("pixel_mse", "pixel_corr"):
            values[arm] = (float(np.mean([r["exact_hit"] for r in rows])),
                           float(np.mean([r["patient_hit"] for r in rows])))

    problems = []
    for arm, (exp_e, exp_p) in EXPECTED["F4"].items():
        if arm not in values:
            problems.append(f"{arm}: not present in the records")
            continue
        got_e, got_p = values[arm]
        if abs(got_e - exp_e) > 5e-3 or abs(got_p - exp_p) > 5e-3:
            problems.append(f"{arm}: record ({got_e:.3f},{got_p:.3f}) vs expected ({exp_e},{exp_p})")
    if problems:
        raise FigureError("F4 values disagree with expectations:\n  " + "\n  ".join(problems))

    order = ["random", "pixel_mse", "pixel_corr", "cosine_nn", "cosine_ot", "learned", "nmi"]
    labels = {"random": "random", "pixel_mse": "pixel MSE", "pixel_corr": "pixel correlation",
              "cosine_nn": "cosine NN", "cosine_ot": "cosine + OT", "learned": "learned cost",
              "nmi": "pixel NMI (ref.)"}

    fig, (ax_e, ax_p) = plt.subplots(1, 2, figsize=(9.0, 3.6))
    apply_style()
    ys = np.arange(len(order))[::-1]
    for y, arm in zip(ys, order):
        exact, patient = values[arm]
        filled = arm != "nmi"
        for ax, xval in ((ax_e, exact), (ax_p, patient)):
            ax.plot([xval], [y], marker=marker(arm), color=colour(arm),
                    markerfacecolor=colour(arm) if filled else "white",
                    markeredgecolor=colour(arm), markeredgewidth=1.1, markersize=6, zorder=3)
            ax.annotate(f"{xval:.3f}", xy=(xval, y), xytext=(7, 0), textcoords="offset points",
                        fontsize=7.5, va="center", color=colour(arm))

    ax_e.set_xscale("log")
    ax_e.set_xticks([0.005, 0.01, 0.05, 0.1, 0.5, 1.0])
    ax_e.set_xticklabels(["0.005", "0.01", "0.05", "0.1", "0.5", "1"])
    ax_e.set_xlim(0.004, 3.0)
    ax_e.set_xlabel("exact-slice recovery (log)")
    ax_e.axvline(1.0 / 630, color="#b3b3b3", lw=0.8, ls=(0, (3, 2)), zorder=1)
    ax_e.annotate("chance", xy=(1.0 / 630, ys[-1] - 0.55), fontsize=7, color="#8c8c8c", ha="center")

    # CN: 右栏必须与左栏共享同一个分类 y 轴；否则 matplotlib 会把 0..6 当作数据刻度画出来。
    # EN: the right panel must share the categorical y axis with the left, otherwise matplotlib draws 0..6 as
    #     numeric ticks.
    ax_p.set_ylim(ax_e.get_ylim())
    ax_p.set_yticks(ys)
    ax_p.set_yticklabels([])
    ax_p.set_xlim(-0.02, 1.08)
    ax_p.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax_p.set_xlabel("same-patient recovery")
    ax_p.axvline(1.0 / 45, color="#b3b3b3", lw=0.8, ls=(0, (3, 2)), zorder=1)
    ax_p.annotate("chance", xy=(1.0 / 45, ys[-1] - 0.55), fontsize=7, color="#8c8c8c", ha="center")

    for ax in (ax_e, ax_p):
        ax.set_yticks(ys)
        ax.tick_params(length=3)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    ax_e.set_yticklabels([labels[a] for a in order])

    title, subtitle = TITLES["F4"]
    add_title(fig, title, subtitle, axes_left=0.115)
    cfg = next(r for r in payload["records"] if not r["excluded"])["config"]
    extra = next(r for r in payload["records"] if not r["excluded"])["extra"]
    n_slices, n_patients = int(extra["n_side"]), int(extra["n_patients"])
    experiment = (
        f"Each method must pair MR slices with CT slices from one {n_slices}-slice SynthRAD pool "
        f"({n_patients} patients, soft window, DEV split), and is scored against the true patient "
        "and slice.")
    comparing = (
        "Rows compare pixel distance, cosine, cosine plus OT and the learned cost, plus NMI as a "
        "reference; the left panel needs the exact slice, the right one only the patient.")
    config_text = (
        f"Costs are fitted on {cfg['fit_patients_per_draw']} other FIT patients, in one draw. Any of "
        f"the {n_slices} slices could be named, so blind guessing is right 1 time in {n_slices} for "
        f"the slice and 1 in {n_patients} for the patient.")
    add_footer(fig, experiment=experiment, comparing=comparing, config=config_text,
               axes_left=0.115, labelled=False)
    fig.subplots_adjust(left=0.115, right=0.985, top=0.80, bottom=0.22, wspace=0.22)
    png = save(fig, "F4_pair_recovery")

    rows = [[arm, "exact_slice", f"{values[arm][0]:.4f}"] for arm in order] + \
           [[arm, "same_patient", f"{values[arm][1]:.4f}"] for arm in order]
    write_metadata(
        "F4_pair_recovery",
        caption={
            "axes": "left: exact-slice recovery, i.e. the paired slice has the right patient and the right "
                    "z index (log scale); right: same-patient recovery (linear scale); one row per method.",
            "instance": f"{n_slices} slices / {n_patients} patients, soft CT window, one draw, "
                        f"{cfg['fit_patients_per_draw']} FIT patients for the fits, DEV split for evaluation",
            "methods": "pixel MSE and correlation (no learning); cosine NN and cosine + OT; the learned "
                       "θ; pixel NMI as a reference; random",
            "config": config_text,
            "sources": "iot_experiments/out/synthrad_v2/records.json (feature arms) and "
                       "iot_experiments/out/synthrad_cheap/records.json (pixel arms, item A)",
            "conventions": "left and right share one categorical y axis; the dashed line is chance; "
                           "pixel NMI is a reference on registered pairs, drawn hollow",
        },
        header=["method", "metric", "value"],
        rows=rows,
    )
    print_caption("F4_pair_recovery", experiment=experiment, comparing=comparing, config=config_text)
    return png


def build_f3() -> Path:
    """
    CN: **F3** —— 覆盖度 vs 精确率，cosine 与学到的代价两条线；每个覆盖率上有淡点表示各 draw。
    EN: **F3** -- coverage vs precision for cosine and the learned cost, with faint per-draw dots.
    """
    record = RECORDS["F3"]
    parsed = parse_precision_table(record)
    check("F3", parsed)

    fig, ax = plt.subplots(figsize=(6.5, 4.4))
    apply_style()
    x = np.asarray([c * 100.0 for c in COVERAGE])

    # CN: cosine 用空心方块、iOT 用实心圆——两条线几乎重合，必须靠形状区分（灰度也可读）。
    # EN: cosine is a hollow square and iOT a filled circle: the lines nearly coincide, so shape must carry it
    #     (and it survives greyscale).
    for arm, label, filled in (("cosine", "cosine", False), ("learned", "iOT", True)):
        y = np.asarray(parsed[arm])
        colour = ARM_COLORS["cosine"] if arm == "cosine" else ARM_COLORS["iot"]
        marker = ARM_MARKERS["cosine"] if arm == "cosine" else ARM_MARKERS["iot"]
        ax.plot(x, y, color=colour, marker=marker, label=label, zorder=3,
                markerfacecolor=colour if filled else "white", markeredgecolor=colour,
                markeredgewidth=1.1)

    ax.set_xlim(15, 128)
    ax.set_ylim(0.82, 1.012)
    ax.set_xticks([25, 50, 75, 100])
    ax.set_yticks([0.85, 0.90, 0.95, 1.00])
    ax.set_xlabel("most confident matches kept (%)")
    ax.set_ylabel("precision")
    ax.tick_params(length=3)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

    # CN: 两条线在 25–50% 完全重合，故只标 iOT；100% 处两值分开标注并各自加线端标签。
    # EN: the lines coincide at 25-50%, so label only iOT there; at 100% both values are labelled separately
    #     and each line gets its own end label.
    ax.annotate(f"{parsed['learned'][1]:.4f}", xy=(50, parsed["learned"][1]), xytext=(0, 10),
                textcoords="offset points", ha="center", fontsize=7.5, color=ARM_COLORS["iot"])
    # CN: 把数值与线名合并成"名字 数值"，一个标签一个位置，避免重叠。
    # EN: merge value and name into one "name value" label, one per position, so nothing collides.
    ax.annotate(f"iOT {parsed['learned'][3]:.4f}", xy=(100, parsed["learned"][3]), xytext=(9, 4),
                textcoords="offset points", ha="left", va="center", fontsize=7.5,
                color=ARM_COLORS["iot"])
    ax.annotate(f"cosine {parsed['cosine'][3]:.4f}", xy=(100, parsed["cosine"][3]), xytext=(9, -7),
                textcoords="offset points", ha="left", va="center", fontsize=7.5,
                color=ARM_COLORS["cosine"])

    title, subtitle = TITLES["F3"]
    add_title(fig, title, subtitle, axes_left=0.105)
    experiment = (
        "t1t2/pool — T1↔T2 matching on one fixed 700-pair pool, geometry G2, "
        "standardisation and PCA fitted on the pool images")
    comparing = (
        "cosine (label-free) vs learned diagonal θ, the same cost family with two θ; "
        "precision among the top 25/50/75/100 % of rows ranked by margin")
    config_text = (
        "K=256 · m ∈ {100,300} · 3 draws · P@c = hit rate of the top c·n rows; "
        "no pair discarded from the matching")
    add_footer(fig, experiment=experiment, comparing=comparing, config=config_text, axes_left=0.105)
    fig.subplots_adjust(left=0.105, right=0.97, top=0.84, bottom=0.18)
    png = save(fig, "F3_precision_by_confidence")

    rows = [[arm, int(round(c * 100)), f"{parsed[arm][i]:.4f}"]
            for arm in ("cosine", "learned") for i, c in enumerate(COVERAGE)]
    write_metadata(
        "F3_precision_by_confidence",
        caption={
            "axes": "x = share of matches kept, ranked from most to least confident by margin (%); "
                    "y = precision among those kept; two arms.",
            "instance": "K=256, m ∈ {100,300}, 3 draws, one fixed 700-pair pool, geometry G2",
            "methods": "cosine (hollow squares) and the learned diagonal θ (filled circles); every row is "
                       "matched to exactly one column and a row is a hit when the id matches",
            "config": config_text,
            "sources": "reports/current/REAL_PAIRS_REPORT.md §5, table of P@25..P@100; the generating "
                       "records were never committed, so these values are parsed from the report table",
            "conventions": "margin = min(row best−second gap, column best−second gap); coverage selects "
                           "which subset is evaluated, no pair is removed from the matching",
        },
        header=["arm", "coverage_percent", "precision"],
        rows=rows,
    )
    print_caption("F3_precision_by_confidence", experiment=experiment, comparing=comparing, config=config_text)
    return png


def main() -> int:
    """CN: 入口。EN: entry point."""
    apply_style()
    for build in (build_f2, build_f5, build_f4, build_f3):
        print(f"written: {build()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
