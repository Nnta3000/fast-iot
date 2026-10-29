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
    # 基线 / 参考 —— 中性灰
    "cosine": "#4d4d4d",
    "cosine_nn": "#4d4d4d",
    "cosine_ot": "#2e2e2e",
    "random": "#b3b3b3",
    "pixel_mse": "#8c8c8c",
    "pixel_corr": "#6e6e6e",
    "nmi": "#999999",
    "cca": "#7f7f7f",
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
    "cosine": "o",
    "cosine_nn": "o",
    "cosine_ot": "D",
    "random": "x",
    "pixel_mse": "v",
    "pixel_corr": "^",
    "nmi": "P",
    "cca": "s",
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


def add_footer(fig: Figure, source: str, caveat: str, *, axes_left: float = 0.13) -> None:
    """
    CN: 两行页脚（7 pt 灰）：第一行来源（记录路径 | commit | pool/draw），第二行注意事项与状态。
    EN: two-line footer (7 pt grey): line 1 the source (record path | commit | pool/draw), line 2 the
        caveats and status.
    """
    if "\n" in source or "\n" in caveat:
        raise StyleError("each footer line must be a single line")
    fig.text(axes_left, 0.035, source, ha="left", va="bottom", fontsize=7, color="#8c8c8c")
    fig.text(axes_left, 0.005, caveat, ha="left", va="bottom", fontsize=7, color="#8c8c8c")


def save(fig: Figure, name: str) -> tuple[Path, Path]:
    """
    CN: 以 300 dpi 保存 PNG 与 PDF，tight bbox，白底。返回两个路径。
    EN: save PNG and PDF at 300 dpi with a tight bbox on white. Returns both paths.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    png = OUTPUT_DIR / f"{name}.png"
    pdf = OUTPUT_DIR / f"{name}.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf, dpi=300, bbox_inches="tight", facecolor="white")
    return png, pdf


__all__ = [
    "ALIASES",
    "ARM_COLORS",
    "ARM_MARKERS",
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
