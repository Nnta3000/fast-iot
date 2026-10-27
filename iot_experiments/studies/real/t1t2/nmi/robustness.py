"""
CN: **NMI 鲁棒性扫描** —— 问题：之前 32×32 / 32 箱的 NMI 在加噪后崩到 0.003，是"NMI 本身不行"，
    还是**直方图样本不足**（1,024 个像素填 1,024 个格子）？

    CN: 设计：分辨率 192×192（每对 36,864 个样本），箱数 {32, 64}，噪声 σ ∈ {0, 0.1, 0.3}
        （加在 T2 像素上，`sigma=0` 是原图）。其余与 `studies/nmi` 完全相同：池 = 全部 1000 个受试者，
        两侧独立打乱，真实配对隐藏，Hungarian 解码，chance = 1/1000。

    CN: 判定标准（预先写下）：若高分辨率下 σ=0.3 的 `pair_acc` **明显高于** 32×32 的 0.003，
        则先前结论应改为"样本不足"，而不是"NMI 崩溃"。

EN: **NMI robustness sweep** -- is the earlier collapse at 32x32 / 32 bins (pair_acc 0.003 under noise)
    a property of NMI, or of **histogram sample starvation** (1,024 pixels filling 1,024 cells)?

    Design: 192x192 resolution (36,864 samples per pair), bins in {32, 64}, noise sigma in {0, 0.1, 0.3}
    added to the T2 pixels (`sigma=0` is the original image). Everything else matches `studies/nmi`: the
    pool is all 1000 subjects, both sides are shuffled independently, the true pairing is hidden, decoding
    is Hungarian, and chance is 1/1000.

    Pre-registered reading: if at high resolution the ``sigma=0.3`` `pair_acc` is **substantially above**
    the 32x32 value of 0.003, the earlier conclusion must be restated as "sample starvation", not
    "NMI collapses".

用法 / EN: usage
    PYTHONPATH=/home/agent/Documents/dpm-inverse-ot python -u -m iot_experiments.studies.real.t1t2.nmi.robustness
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from jaxtyping import Float, Int

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from iot_experiments.harness.config import RunConfig, build_parser  # noqa: E402
from iot_experiments.harness.records import RunRecord, format_table, write_records  # noqa: E402
from iot_experiments.studies.real.t1t2.nmi.run import (  # noqa: E402
    hungarian,
    load_pixels,
    nmi_pixel_cost,
    pair_accuracy,
)

PROVENANCE: dict[str, object] = {
    "estimator": "joint pixel histogram of the candidate pair, equal-frequency bins per image",
    "pool": "all 1000 subjects, T1 and T2 sides shuffled independently (true partners hidden)",
    "chance_pair_acc": 1.0 / 1000.0,
    "purpose": "separate 'NMI cannot pair' from 'the histogram was sample-starved'",
}


@dataclass(frozen=True)
class RobustConfig(RunConfig):
    """CN: 鲁棒性扫描的旋钮。EN: the robustness sweep's knobs."""

    section = "nmi_robustness"
    study: str = "nmi_robustness"
    sizes: tuple[int, ...] = (192,)
    bins: tuple[int, ...] = (32, 64)
    sigmas: tuple[float, ...] = (0.0, 0.1, 0.3)

    @classmethod
    def add_study_arguments(cls, parser: argparse.ArgumentParser) -> None:
        """CN: 只加本研究特有的参数。EN: only this study's flags."""
        parser.add_argument("--sizes", type=int, nargs="+", default=[192],
                            help="image side length (pixels = size^2)")
        parser.add_argument("--bins", type=int, nargs="+", default=[32, 64],
                            help="equal-frequency histogram bins")
        parser.add_argument("--sigmas", type=float, nargs="+", default=[0.0, 0.1, 0.3],
                            help="gaussian noise sigma added to the T2 pixels (0 = none)")


def main(argv: list[str] | None = None) -> int:
    """CN: 入口。EN: entry point."""
    parser = build_parser(RobustConfig)
    args = parser.parse_args(list(argv) if argv is not None else None)
    config = RobustConfig.from_namespace(args, parser)

    records: list[RunRecord] = []
    for size in config.sizes:
        px1, px2 = load_pixels(size=int(size))
        n_subjects: int = int(px1.shape[0])
        pixels: int = int(px1.shape[1])
        # CN: 与 studies/nmi 同一打乱，故结果可逐格对照。
        # EN: the same shuffles as studies/nmi, so the cells are directly comparable.
        order_t1: Int[np.ndarray, "n"] = np.random.default_rng(0).permutation(n_subjects)
        order_t2: Int[np.ndarray, "n"] = np.random.default_rng(1).permutation(n_subjects)
        inv_t2: Int[np.ndarray, "n"] = np.empty(n_subjects, dtype=np.int64)
        inv_t2[order_t2] = np.arange(n_subjects, dtype=np.int64)
        pos_of_true: Int[np.ndarray, "n"] = inv_t2[order_t1]
        a: Float[np.ndarray, "n pixels"] = px1[order_t1]

        for sigma in config.sigmas:
            rng = np.random.default_rng(1000 + int(round(float(sigma) * 100)))
            b: Float[np.ndarray, "n pixels"] = px2[order_t2]
            if float(sigma) > 0.0:
                b = np.clip(b + rng.normal(scale=float(sigma), size=b.shape), 0.0, 1.0)
            for bins in config.bins:
                cost = nmi_pixel_cost(a, b, bins=int(bins))
                accuracy = pair_accuracy(hungarian(cost), pos_of_true)
                samples_per_cell: float = float(pixels) / float(int(bins) ** 2)
                records.append(RunRecord(
                    study=str(config.study), geometry=f"{size}x{size}", arm=f"bins{int(bins)}",
                    k=int(bins), m=0, draw=-1,
                    metrics={"pair_acc": accuracy, "samples_per_cell": samples_per_cell},
                    config=config.to_dict(),
                    extra={"sigma": float(sigma), "pixels": pixels, "bins": int(bins)}))
                print(f"  {size}x{size} bins={bins:>3} sigma={sigma:<4} pair_acc={accuracy:.4f}  "
                      f"(samples/cell={samples_per_cell:.1f})", flush=True)

    rows = [{"size": r.geometry, "bins": int(r.extra["bins"]), "sigma": float(r.extra["sigma"]),
             "samples_per_cell": r.metrics["samples_per_cell"], "pair_acc": r.metrics["pair_acc"]}
            for r in records]
    print("\n" + format_table(rows, ("size", "bins", "sigma", "samples_per_cell", "pair_acc")))
    print(f"\n# chance pair_acc = 1/{n_subjects} = {1.0 / n_subjects:.4f}")
    print("# reference: 32x32 / 32 bins at sigma=0.3 gave 0.0030 (1024 pixels into 1024 cells)")
    write_records(config.run_dir / "records.json", records, provenance=PROVENANCE)
    print(f"\n[nmi_robustness] wrote {config.run_dir / 'records.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
