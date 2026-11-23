"""
CN: **图形样式** —— 一次定义，五张图共用。所有颜色、字号、坐标轴细节都在这里，脚本里不再出现。

    EN: **Figure style** -- defined once, shared by all figures. Colours, type sizes and axis details live
        here; the figure scripts contain none of it.

    CN: 设计约束（来自要求）：
        * 调色板是项目自己的 14 色条（`orange-palette-q3.png`），见 `PALETTE`；
        * **每个方法一个固定颜色**，五张图复用；基线/参考（cosine、random、像素方法、NMI）用中性灰；
        * 同时变化**标记形状**，因此在灰度打印下仍可区分；
        * 极简：无网格、无上/右边框、左/下边框 0.8 pt、显式刻度、刻度长 3、无次刻度；
        * **标题是名词短语，不是结论**；`add_title` 会**拒绝**含结论性词汇的标题或副标题。
    EN: Design constraints (from the brief):
        * the project's own 14-colour strip (`orange-palette-q3.png`), exposed as `PALETTE`;
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

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
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


# CN: **项目调色板** —— 14 色，取自 `orange-palette-q3.png`（14×1 px 的色条，逐像素读出来的，不是目测）。
#     按色条的从左到右顺序排列；浅色端（白、浅黄、粉）只用于填充/热图，不用于线条。
# EN: **the project palette** -- 14 colours taken from `orange-palette-q3.png` (a 14x1 px strip, read
#     pixel by pixel, not eyeballed). Ordered left to right along the strip. The pale end (white, pale
#     yellow, pink) is for fills and heat maps only, never for a line on white.
PALETTE: tuple[str, ...] = (
    "#c0c0e0",  # 0 lavender
    "#ffffff",  # 1 white
    "#c0c0e0",  # 2 lavender (repeated in the strip)
    "#606080",  # 3 slate
    "#202020",  # 4 ink
    "#000000",  # 5 black
    "#c04060",  # 6 crimson
    "#ff8020",  # 7 orange
    "#ffe020",  # 8 yellow
    "#ffffc0",  # 9 pale yellow
    "#60c0e0",  # 10 sky
    "#4060c0",  # 11 royal
    "#ffe0a0",  # 12 cream
    "#ffe0e0",  # 13 pink
)

# CN: 中性带里插值出来的三档浅灰，给基线与参考文献用（调色板本身只有 slate / ink / white 三档）。
# EN: three light neutrals interpolated inside the palette's neutral band, for baselines and references
#     (the strip itself only has slate / ink / white).
GREY_SOFT: str = "#9090b0"
GREY_FAINT: str = "#b0b0d0"

# CN: 热图的连续色表 —— 取色条中**亮度单调上升**的一段：ink → crimson → orange → yellow → pale yellow。
# EN: the continuous map for heat maps -- the segment of the strip whose lightness rises monotonically:
#     ink -> crimson -> orange -> yellow -> pale yellow.
SEQUENTIAL_COLORS: tuple[str, ...] = ("#202020", "#c04060", "#ff8020", "#ffe020", "#ffffc0")
SEQUENTIAL_CMAP = LinearSegmentedColormap.from_list("iot_palette", list(SEQUENTIAL_COLORS))


def _palette(index: int) -> str:
    """
    CN: 取调色板上的第 `index` 色（0–13）。越界即报错，避免悄悄回退到默认色。
    EN: the `index`-th colour of the palette (0-13). Out of range raises rather than silently falling back.
    """
    if not 0 <= int(index) < len(PALETTE):
        raise StyleError(f"palette index {index!r} outside 0..{len(PALETTE) - 1}")
    return PALETTE[int(index)]


# CN: **每个方法一个颜色**。学到的臂用暖色（crimson/orange）与冷色（royal/sky）区分族；基线与参考用中性色。
# EN: **one colour per method**. Learned arms split into a warm family (crimson/orange) and a cool one
#     (royal/sky); baselines and references stay neutral.
ARM_COLORS: dict[str, str] = {
    # 学到的代价 / learned costs
    "iot": _palette(6),
    "infonce": _palette(7),
    "residual64": _palette(11),
    "residual16": _palette(10),
    "sinkhorn": _palette(10),
    # FAUST 学习臂与方法 / FAUST learned arms and objectives
    "relaxed_lp": _palette(6),
    "multi": _palette(11),
    "single": _palette(6),
    "entropic_iot": _palette(10),
    "row_softmax": _palette(7),
    "rowcol_softmax": _palette(11),
    "selected": _palette(6),
    "median": _palette(7),
    "reference": _palette(4),
    # 基线 / 参考 —— 中性
    "cosine": _palette(3),
    "cosine_nn": _palette(3),
    "cosine_ot": _palette(4),
    "random": _palette(0),
    "pixel_mse": GREY_SOFT,
    "pixel_corr": _palette(3),
    "nmi": _palette(0),
    "cca": GREY_SOFT,
    "uniform": _palette(3),
    "uniform_theta": _palette(3),
    "uniform_late_hks": GREY_SOFT,
    "uniform_hks": _palette(0),
    "grouped_uniform": _palette(4),
    "position_only": _palette(0),
    # 几何 G0-G3（F1）
    "G0": _palette(0),
    "G1": GREY_SOFT,
    "G2": _palette(3),
    "G3": _palette(4),
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
