"""
CN: **几何研究（重构后的参考实现）** —— 一个问题：**PCA 之后的变换**（G0/G1/G2/G3）如何影响检索，
    以及在同一几何内"学到的 θ"是否胜过 cosine。

    CN: 配置只通过 `GeometryConfig` 添加自己的旋钮（K、m、几何、臂），共享旋钮全部来自 `RunConfig`。
        运行：
            PYTHONPATH=/home/agent/Documents/dpm-inverse-ot \
              python -u -m iot_experiments.studies.real.t1t2.geometry.run --ks 64 128 --ms 100 300 \
                --geometries G2 G3 --arms cosine learned --draws 0 1 2

EN: **Geometry study (the reference implementation after the refactor)** -- one question: how does the
    **post-PCA** transform (G0/G1/G2/G3) affect retrieval, and does a learned theta beat cosine within the
    same geometry?

    Only its own knobs are added through `GeometryConfig`; every shared knob comes from `RunConfig`.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[4]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy.typing as npt  # noqa: E402
from jaxtyping import Float  # noqa: E402

from iot_experiments.harness.config import RunConfig, build_parser  # noqa: E402
from iot_experiments.harness.driver import run_cells  # noqa: E402
from iot_experiments.harness.records import format_table, summarise, verdict  # noqa: E402
from iot_experiments.harness.registry import features  # noqa: E402

PROVENANCE: dict[str, object] = {
    "feature": "DINOv2 vit_small_patch14_dinov2, patch tokens pooled to 4x4, PCA (pooled mean)",
    "pca_fit_set": "all T1+T2 images (label-free)",
    "standardisation_fit_set": "pool images only (transductive, label-free)",
    "note": "standardisation and PCA use the pool images (transductive, no labels)",
}


@dataclass(frozen=True)
class GeometryConfig(RunConfig):
    """
    CN: 几何研究自己的旋钮。基类已提供 `--out/--solver/--draws/...`，这里只加 4 个。
    EN: The geometry study's own knobs; the base supplies `--out/--solver/--draws/...`.
    """

    section = "geometry"
    # CN: 默认 study 名即 section，输出落在 `gpu_bench/out/geometry/`。
    # EN: the default study name is the section, so output lands in `gpu_bench/out/geometry/`.
    study: str = "geometry"
    ks: tuple[int, ...] = (64, 128)
    ms: tuple[int, ...] = (100, 300)
    geometries: tuple[str, ...] = ("G2", "G3")
    arms: tuple[str, ...] = ("cosine", "learned")
    feature: str = "dinov2-pool4"

    @classmethod
    def add_study_arguments(cls, parser: argparse.ArgumentParser) -> None:
        """CN: 只加本研究特有的参数。EN: only this study's own flags."""
        parser.add_argument("--ks", type=int, nargs="+", default=[64, 128],
                            help="PCA components K")
        parser.add_argument("--ms", type=int, nargs="+", default=[100, 300],
                            help="labelled pairs m")
        parser.add_argument("--geometries", nargs="+", default=["G2", "G3"],
                            choices=["G0", "G1", "G2", "G3"], help="post-PCA transforms")
        parser.add_argument("--arms", nargs="+", default=["cosine", "learned"],
                            choices=["cosine", "learned"], help="arms to evaluate")
        parser.add_argument("--feature", default="dinov2-pool4",
                            choices=["dinov2-pool4", "dinov2-cls", "block"], help="feature provider")


def main(argv: list[str] | None = None) -> int:
    """CN: 入口。EN: entry point."""
    parser = build_parser(GeometryConfig)
    args = parser.parse_args(list(argv) if argv is not None else None)
    config = GeometryConfig.from_namespace(args, parser)

    def features_by_k(k: int) -> tuple[Float[npt.NDArray, "subjects d"],
                                       Float[npt.NDArray, "subjects d"]]:
        """CN: 取某 K 的原始特征（未后处理）。EN: raw features for one K (no post-processing)."""
        return features(str(config.feature), dim=int(k))

    records = run_cells(features_by_k, config=config, ks=config.ks, ms=config.ms,
                        geometries=config.geometries, arms=config.arms, provenance=PROVENANCE)

    rows = summarise(records, group=("geometry", "arm"), metrics=("hit", "R@1"))
    print("\n" + format_table(rows, ("geometry", "arm", "n_draws", "hit_mean", "hit_std",
                                     "R@1_mean", "R@1_std")))

    print("\n# pre-registered verdict (hit higher by >= 3 points on ALL draws)")
    for geom in config.geometries:
        learned = [r for r in records if r.geometry == geom and r.arm == "learned"
                   and not r.excluded]
        cosine = [r for r in records if r.geometry == geom and r.arm == "cosine"]
        if not learned or len(learned) != len(cosine):
            print(f"  {geom}: not comparable (learned={len(learned)}, cosine={len(cosine)})")
            continue
        for k in sorted({r.k for r in learned}):
            for m in sorted({r.m for r in learned}):
                l = [r.metrics["hit"] for r in learned if r.k == k and r.m == m]
                c = [r.metrics["hit"] for r in cosine if r.k == k and r.m == m]
                if len(l) == len(c) and l:
                    print(f"  {geom} K={k} m={m}: {verdict(l, c)}")
    print(f"\n[geometry] wrote {config.run_dir / 'records.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
