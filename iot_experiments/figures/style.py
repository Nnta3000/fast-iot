"""
CN: **图形样式** —— 一次定义，五张图共用。所有颜色、字号、坐标轴细节都在这里，脚本里不再出现。

    EN: **Figure style** -- defined once, shared by all figures. Colours, type sizes and axis details live
        here; the figure scripts contain none of it.

    CN: 设计约束（来自要求）：
        * 调色板取 matplotlib 的 "magma"，只在 [0.15, 0.80] 之间取色——避开过浅的黄色端与纯黑端；
        * **每个方法一个固定颜色**，五张图复用；基线/参考（cosine、random、像素方法、NMI）用中性灰；
        * 同时变化**标记形状**，因此在灰度打印下仍可区分；
        * 极简：无网格、无上/右边框、左/下边框 0.8 pt、显式刻度、刻度长 3、无次刻度；
        * **标题是名词短语，不是结论**；`add_title` 会**拒绝**含结论性词汇的标题或副标题。
    EN: Design constraints (from the brief):
        * "magma" palette sampled only in [0.15, 0.80], avoiding the pale-yellow and black ends;
        * **one fixed colour per method**, reused across all figures; baselines/references (cosine, random,
          pixel methods, NMI) in neutral greys;
        * **marker shape** varies too, so figures stay readable in greyscale;
        * minimal: no grid, no top/right spines, left/bottom at 0.8 pt, explicit ticks, tick length 3, no
          minor ticks;
        * **titles are noun phrases, not conclusions**; `add_title` **rejects** concluding words.
"""

from __future__ import annotations

import re
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import to_hex
from matplotlib.figure import Figure

# CN: 结论性词汇黑名单。标题/副标题命中任何一个就抛错。
# EN: blacklist of concluding words; any title/subtitle matching one raises.
FORBIDDEN = re.compile(
    r"beats|better|worse|improv|helps|matters|wins|fails|outperform|than|knows|finds|touches|best|only|"
    r"clearly|strong|weak|gain|lose|saturat",
    re.IGNORECASE,
)


# CN: 页脚里出现文件路径（`.md`/`.json`/…）就抛错：读者要的是"哪个实验、比了什么"，不是路径。
# EN: a file path in the footer (`.md`, `.json`, ...) raises: the reader needs "which experiment, what is
#     compared", not a path.
NO_PATH = re.compile(r"\.(?:md|json|csv|py|pdf|png|log)\b|/out/", re.IGNORECASE)


class StyleError(RuntimeError):
    """CN: 样式或标题不合规。EN: a style or title violation."""


def _magma(position: float) -> str:
    """
    CN: 从 "magma" 上取一个固定位置的颜色。位置必须在 [0.15, 0.80]：更低是黑，更高是浅黄，两端在
        打印和投影上都不可读。
    EN: sample "magma" at a fixed position. The position must lie in [0.15, 0.80]: below is black and above is
        pale yellow, and neither end is readable in print or on a projector.
    """
    if not 0.15 <= float(position) <= 0.80:
        raise StyleError(f"palette position {position!r} outside [0.15, 0.80]")
    return to_hex(matplotlib.colormaps["magma"](float(position)))


# CN: **每个方法一个颜色**。位置来自要求：iOT 0.25、InfoNCE 0.52、low-rank residual 0.78。
# EN: **one colour per method**. Positions from the brief: iOT 0.25, InfoNCE 0.52, low-rank residual 0.78.
ARM_COLORS: dict[str, str] = {
    # 学到的代价 / learned costs
    "iot": _magma(0.25),
    "infonce": _magma(0.52),
    "residual64": _magma(0.78),
    "residual16": _magma(0.68),
    "sinkhorn": _magma(0.40),
    # FAUST 学习臂与方法 / FAUST learned arms and objectives
    "relaxed_lp": _magma(0.25),
    "multi": _magma(0.78),
    "single": _magma(0.25),
    "entropic_iot": _magma(0.40),
    "row_softmax": _magma(0.52),
    "rowcol_softmax": _magma(0.68),
    "selected": _magma(0.25),
    "median": _magma(0.68),
    "reference": "#2e2e2e",
    # 基线 / 参考 —— 中性灰
    "cosine": "#4d4d4d",
    "cosine_nn": "#4d4d4d",
    "cosine_ot": "#2e2e2e",
    "random": "#b3b3b3",
    "pixel_mse": "#8c8c8c",
    "pixel_corr": "#6e6e6e",
    "nmi": "#999999",
    "cca": "#7f7f7f",
    "uniform": "#8c8c8c",
    "uniform_theta": "#8c8c8c",
    "uniform_late_hks": "#a8a8a8",
    "uniform_hks": "#bcbcbc",
    "grouped_uniform": "#6e6e6e",
    "position_only": "#c9c9c9",
    # 几何 G0-G3（F1）
    "G0": "#b3b3b3",
    "G1": "#8c8c8c",
    "G2": "#4d4d4d",
    "G3": "#2e2e2e",
}

# CN: 标记形状同时变化，保证灰度可读。EN: marker shapes vary too, for greyscale readability.
ARM_MARKERS: dict[str, str] = {
    "iot": "o",
    "infonce": "s",
    "residual64": "^",
    "residual16": "v",
    "sinkhorn": "D",
    "relaxed_lp": "o",
    "multi": "^",
    "single": "o",
    "entropic_iot": "v",
    "row_softmax": "s",
    "rowcol_softmax": "^",
    "selected": "o",
    "median": "s",
    "reference": "D",
    "cosine": "o",
    "cosine_nn": "o",
    "cosine_ot": "D",
    "random": "x",
    "pixel_mse": "v",
    "pixel_corr": "^",
    "nmi": "P",
    "cca": "s",
    "uniform": "s",
    "uniform_theta": "s",
    "uniform_late_hks": "v",
    "uniform_hks": "^",
    "grouped_uniform": "D",
    "position_only": "x",
    "G0": "o",
    "G1": "s",
    "G2": "^",
    "G3": "D",
}

# CN: 同一方法在不同记录里可能用不同名字（`iot` 与 `learned`），这里给一个别名表，
#     保证"一个方法一个颜色"在所有图中成立。
# EN: the same method may be named differently across records (`iot` vs `learned`); an alias table keeps
#     "one method, one colour" true across all figures.
ALIASES: dict[str, str] = {
    "learned": "iot",
    "cosine": "cosine",
    "nmi": "nmi",
    "pixel_corr": "pixel_corr",
    "pixel_mse": "pixel_mse",
}


def colour(arm: str) -> str:
    """CN: 取某方法的固定颜色（支持别名）。EN: fixed colour for a method, honouring aliases."""
    return ARM_COLORS[ALIASES.get(str(arm), str(arm))]


def marker(arm: str) -> str:
    """CN: 取某方法的固定标记（支持别名）。EN: fixed marker for a method, honouring aliases."""
    return ARM_MARKERS[ALIASES.get(str(arm), str(arm))]


FIGURE_DIR: Path = Path(__file__).resolve().parent
OUTPUT_DIR: Path = FIGURE_DIR / "out"
# CN: 图的元数据（caption.md + csv）放在 `out/` 之外，因为 `.../figures/out/` 被 .gitignore 忽略——
#     放进 out/ 就会在 clone 后消失，而"这张图画的是什么"必须能被复现。
# EN: figure metadata (caption.md + csv) lives outside `out/`, because `.../figures/out/` is git-ignored:
#     anything inside it disappears on a clone, and "what is plotted here" has to be reproducible.
METADATA_DIR: Path = FIGURE_DIR / "metadata"


def apply_style() -> None:
    """
    CN: 应用全局 rcParams：极简、白底、无网格。EN: apply global rcParams: minimal, white, no grid.
    """
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
        "axes.edgecolor": "#4d4d4d",
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "xtick.minor.size": 0,
        "ytick.minor.size": 0,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "lines.linewidth": 1.2,
        "lines.markersize": 4.5,
        "legend.frameon": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def add_title(fig: Figure, title: str, subtitle: str, *, axes_left: float = 0.13) -> None:
    """
    CN: 加标题（粗体 13 pt，左对齐）与副标题（10 pt 灰，一行），都与坐标轴左边缘对齐。
        **会拒绝**含结论性词汇的标题或副标题。
    EN: add the title (bold 13 pt, left-aligned) and a one-line subtitle (10 pt grey), both aligned with the
        left edge of the axes. **Rejects** titles or subtitles containing concluding words.
    """
    for text, label in ((title, "title"), (subtitle, "subtitle")):
        match = FORBIDDEN.search(text)
        if match:
            raise StyleError(
                f"{label} contains the concluding word {match.group(0)!r}: {text!r}. "
                "Titles must be descriptive noun phrases, not conclusions.")
    if len(title.split()) > 6:
        raise StyleError(f"title has {len(title.split())} words; the limit is 6: {title!r}")
    if "\n" in subtitle:
        raise StyleError("subtitle must be a single line")
    fig.text(axes_left, 0.97, title, ha="left", va="top", fontsize=13, fontweight="bold", color="#1a1a1a")
    fig.text(axes_left, 0.905, subtitle, ha="left", va="top", fontsize=10, color="#666666")


def direct_label(ax, x: float, y: float, text: str, *, colour: str, dx: float = 0.0, dy: float = 0.0,
                 fontsize: float = 8.0) -> None:
    """
    CN: 在线端直接标注，代替图例。`dx`/`dy` 以**点数**为单位的偏移，便于避开标记。
    EN: label at the end of a line instead of a legend; `dx`/`dy` offset in **points** so it clears markers.
    """
    ax.annotate(text, xy=(x, y), xytext=(dx, dy), textcoords="offset points",
                ha="left", va="center", fontsize=fontsize, color=colour)


def _wrap_to_width(fig: Figure, text: str, width_in: float, *, fontsize: float) -> list[str]:
    """
    CN: 按**英寸**宽度把一段文字折成若干行（用渲染器量宽，不靠字符数估算）。
    EN: wrap one paragraph to a width given in **inches**. Widths are measured with the renderer rather
        than estimated from character counts, so a wide glyph does not push a line off the canvas.
    """
    renderer = fig.canvas.get_renderer()
    probe = fig.text(0.0, 0.0, "", fontsize=fontsize)
    try:
        def width_of(candidate: str) -> float:
            probe.set_text(candidate)
            return float(probe.get_window_extent(renderer).width) / float(fig.dpi)

        words = str(text).split()
        lines: list[str] = []
        current = ""
        for word in words:
            trial = f"{current} {word}".strip()
            if not current or width_of(trial) <= width_in:
                current = trial
            else:
                lines.append(current)
                current = word
        if current:
            lines.append(current)
        return lines
    finally:
        probe.remove()


def add_footer(fig: Figure, *, experiment: str, comparing: str, config: str,
               axes_left: float = 0.13, axes_right: float = 0.985, labelled: bool = True,
               top: float | None = None, fontsize: float = 7.0) -> None:
    """
    CN: 页脚是**一个块**，不是固定三行：每段文字按图宽（`axes_left`..`axes_right`）自动折行，行数按内容
        增减；折出来的行**向下**排，必要时越过画布底边——`bbox_inches="tight"` 会把画布向下扩展，所以
        页脚永远不会挤压坐标区，也永远不会超出左右边界。
    EN: the footer is **one block**, not a fixed three lines: each field wraps at the plot width
        (`axes_left`..`axes_right`), so the number of lines follows the content. The lines run **downwards**
        and may pass the bottom edge of the canvas — `bbox_inches="tight"` extends the saved image down, so the
        footer never steals space from the axes and never runs past the left or right edge.

        Fields, in order: `experiment` (which run), `comparing` (what is side by side), `config` (instance and
        protocol). `labelled=False` prints them as plain sentences, which is the shipped style; `labelled=True`
        prefixes `experiment —` / `comparing —` / `config —`. Either way the footer **must not cite a record
        path** (`NO_PATH`).
    """
    for label, text in (("experiment", experiment), ("comparing", comparing), ("config", config)):
        if not str(text).strip():
            raise StyleError(f"footer {label} is empty; all three footer fields are required")
        if "\n" in text:
            raise StyleError(f"footer {label} must be a single paragraph: {text!r}")
        if (match := NO_PATH.search(text)) is not None:
            raise StyleError(
                f"footer {label} points at a file ({match.group(0)!r}): {text!r}. "
                "State the experiment, the comparison and the config; do not cite a path.")
    # CN: 留 1% 的右边距，避免最后一行正好顶到画布边缘；字段之间空半行，否则折行后三段糊成一段。
    # EN: keep 1 % of the width free at the right so the longest line does not touch the edge, and leave
    #     half a line between fields, otherwise the wrapped paragraphs merge into one block.
    width_in = max(1.0, 0.99 * (float(axes_right) - float(axes_left)) * float(fig.get_figwidth()))
    gap = 9.5 / 72.0 / float(fig.get_figheight())
    field_gap = 4.5 / 72.0 / float(fig.get_figheight())
    # CN: 页脚顶边贴在坐标区下方约 30 pt（标签加刻度的高度），而不是固定在画布底部——否则折行后
    #     坐标标签与页脚之间会留出一大块空白。
    # EN: the footer's top edge sits ~30 pt under the axes (label plus tick height) rather than at a fixed
    #     fraction of the canvas; a fixed fraction leaves a large blank gap once the text wraps.
    if top is None:
        axes_bottom = min((ax.get_position().y0 for ax in fig.axes), default=0.30)
        top = max(0.005, float(axes_bottom) - 30.0 / 72.0 / float(fig.get_figheight()))
    y = float(top)
    for index, (label, text) in enumerate(
            (("experiment", experiment), ("comparing", comparing), ("config", config))):
        wrapped = _wrap_to_width(fig, text, width_in, fontsize=fontsize)
        if labelled:
            wrapped = [f"{label} — {wrapped[0]}"] + wrapped[1:]
        for line in wrapped:
            fig.text(axes_left, y, line, ha="left", va="top", fontsize=fontsize, color="#8c8c8c")
            y -= gap
        if index < 2:
            y -= field_gap


def save(fig: Figure, name: str) -> Path:
    """
    CN: 以 300 dpi 只保存 PNG，tight bbox，白底。返回路径。**不再生成 PDF**：`out/` 里只有 PNG。
    EN: save PNG only, at 300 dpi, tight bbox, white background. Returns the path. **No PDF**: `out/`
        holds PNG files and nothing else.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    png = OUTPUT_DIR / f"{name}.png"
    fig.savefig(png, dpi=300, bbox_inches="tight", facecolor="white")
    return png


__all__ = [
    "ALIASES",
    "ARM_COLORS",
    "ARM_MARKERS",
    "METADATA_DIR",
    "NO_PATH",
    "OUTPUT_DIR",
    "FORBIDDEN",
    "StyleError",
    "add_footer",
    "add_title",
    "apply_style",
    "colour",
    "direct_label",
    "marker",
    "save",
]
