"""
EN: **FAUST error in the leaderboard's convention -- what number would we submit?**

    The project reports geodesic error normalised by `sqrt(surface area)`. Published FAUST correspondences
    are usually normalised by a length scale of the shape and reported as a *percentage* (error x 100) or via
    an accuracy curve. Those conventions differ by a constant factor per mesh, so a raw comparison of our
    0.083 against a leaderboard entry would be meaningless without converting.

    This driver recomputes the error for every arm at FULL resolution on one pair, and reports the SAME
    per-vertex distances under both normalisations:

        `sqrt(area)`  -- what `heatmaps.py` reports (our convention)
        `diameter`    -- the geodesic diameter of the target mesh (`max` of the n x n geodesic matrix)

    plus the accuracy curve (fraction of vertices below each threshold), which is what FAUST leaderboards
    tabulate. The two normalisations are related exactly by `error_diam = error_area * sqrt(area) / diameter`,
    because only the divisor changes; this driver still recomputes both from the raw distances so the
    relation is verified rather than assumed.

    **Protocol caveat, stated up front.** We evaluate on the 100 TRAINING registrations, which share one
    vertex ordering, so `pi_hat = I/n` is ground truth. The official challenge evaluates on the TEST scans,
    whose registrations are withheld. So this gives a number in the leaderboard's *units*, on a
    training-pair protocol -- not an official submission score.

CN: **按排行榜口径计算 FAUST 误差。** 本项目用 `sqrt(area)` 归一化；公开结果多用形状长度尺度的**百分比**。
    本驱动器在全分辨率下重算每个臂的误差，同时给出两种归一化与准确率曲线，并验证两者恰好相差常数因子。
    **协议注意**：我们评估的是 100 个训练配准（共享顶点序，故 `pi_hat=I/n` 是真值）；官方挑战用测试扫描
    （配准未公开），因此这是"同一单位"，不是官方提交成绩。
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
import time
from pathlib import Path

import numpy as np
from jaxtyping import Float

_REPO_ROOT = Path(__file__).resolve().parents[5]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from iot.core.features import SquaredDifferenceFeatures  # noqa: E402
from iot_experiments.harness.config import RunConfig, build_parser  # noqa: E402
from iot_experiments.harness.faust import default_registrations_root, read_ply, surface_area  # noqa: E402
from iot_experiments.harness.faust import geodesic_matrix  # noqa: E402
from iot_experiments.harness.faust_match import cost_matrix, match  # noqa: E402
from iot_experiments.harness.preflight import check as preflight_check  # noqa: E402
from iot_experiments.harness.records import RunLog, RunRecord, write_records  # noqa: E402
from iot_experiments.studies.iot.real.faust.correspondence import (  # noqa: E402
    descriptors,
    fit_theta,
    standardise,
)

DEFAULT_ROOT = str(default_registrations_root())
THRESHOLDS: tuple[float, ...] = (0.0, 0.01, 0.02, 0.03, 0.05, 0.10)

PROVENANCE: dict[str, object] = {
    "question": "what is our FAUST error in the leaderboard's convention, not just sqrt(area)?",
    "normalisations": ["sqrt(surface area) (ours)", "geodesic diameter (max of the n x n geodesic matrix)"],
    "relation": "error_diameter = error_area * sqrt(area) / diameter; both recomputed from raw distances",
    "accuracy_curve": "fraction of vertices with normalised error <= each threshold",
    "protocol": "TRAINING registrations (pi_hat = I/n is exact ground truth); the official challenge uses "
                "TEST scans whose registrations are withheld, so this is the same UNIT, not an official score",
    "resolution": "full 6890 vertices",
}


@dataclasses.dataclass(frozen=True)
class LeaderboardConfig(RunConfig):
    """EN: knobs; real dataclass fields so `from_namespace` cannot drop them. CN: 旋钮。"""

    section = "faust_leaderboard"
    study = "faust_leaderboard"
    root: str = DEFAULT_ROOT
    train_pair: tuple[int, ...] = (0, 13)
    pairs: tuple[int, ...] = (0, 66)
    n_sub: int = 300
    max_iterations: int = 5
    rel: float = -1.0

    def __post_init__(self) -> None:
        """EN: normalise. CN: 归一化。"""
        object.__setattr__(self, "train_pair", tuple(int(v) for v in self.train_pair))
        object.__setattr__(self, "pairs", tuple(int(v) for v in self.pairs))
        if len(self.train_pair) != 2:
            raise ValueError("--train-pair needs exactly two ids")
        if len(self.pairs) % 2 != 0 or not self.pairs:
            raise ValueError("--pairs needs an even number of ids (pairs of pairs)")

    @classmethod
    def add_study_arguments(cls, parser: argparse.ArgumentParser) -> None:
        """EN: only this study's flags. CN: 只加本研究特有的参数。"""
        parser.add_argument("--root", default=DEFAULT_ROOT)
        parser.add_argument("--train-pair", type=int, nargs=2, default=[0, 13])
        parser.add_argument("--pairs", type=int, nargs="+", default=[0, 66])
        parser.add_argument("--n-sub", type=int, default=300)
        parser.add_argument("--max-iterations", type=int, default=5)
        parser.add_argument("--rel", type=float, default=-1.0)


def accuracy_curve(distances: Float[np.ndarray, "n"], thresholds: tuple[float, ...]
                   ) -> dict[str, float]:
    """
    EN: `fraction(dist <= t)` for each threshold -- the convention leaderboards tabulate.
    CN: 每个阈值下的 `fraction(dist <= t)`，即排行榜使用的准确率曲线。
    """
    return {f"acc@{t:g}": float(np.mean(np.asarray(distances) <= float(t))) for t in thresholds}


def main(argv: list[str] | None = None) -> int:
    """EN: entry point. CN: 入口。"""
    parser = build_parser(LeaderboardConfig)
    args = parser.parse_args(list(argv) if argv is not None else None)
    config = LeaderboardConfig.from_namespace(args, parser)
    for flag in ("train_pair", "pairs", "n_sub", "max_iterations"):
        parsed = getattr(args, flag, None)
        carried = getattr(config, flag, None)
        if parsed is not None and carried is not None:
            left = tuple(int(v) for v in parsed) if isinstance(parsed, (list, tuple)) else int(parsed)
            right = tuple(int(v) for v in carried) if isinstance(carried, (list, tuple)) else int(carried)
            if left != right:
                raise SystemExit(f"flag --{flag}={parsed!r} did not reach the config ({carried!r})")

    run_dir = config.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    log = RunLog(run_dir / "run.log")
    log.line(f"FAUST error under both normalisations: train={config.train_pair} pairs={config.pairs}")
    preflight = preflight_check(run_tiny=False, need_gpu=False)
    preflight.write(run_dir / "preflight.json")
    log.line(f"preflight ok={preflight.ok} mem={preflight.mem_available_gb:.1f} GB")

    # EN: `rel < 0` means "no box" -- pure simplex, matching `heatmaps.py`. The box is a different
    #     estimator, so the convention must match the run the numbers are compared against.
    # CN: `rel < 0` 表示不加盒约束（纯单纯形），与 `heatmaps.py` 一致。
    rel = None if float(config.rel) < 0 else float(config.rel)
    log.line(f"rel={rel} (None = pure simplex)")

    root = Path(config.root)
    started = time.perf_counter()
    a_tr = read_ply(root / f"tr_reg_{config.train_pair[0]:03d}.ply")
    b_tr = read_ply(root / f"tr_reg_{config.train_pair[1]:03d}.ply")
    sub = np.sort(np.random.default_rng(0).permutation(a_tr.n_vertices)[: int(config.n_sub)])
    da_sub, db_sub = standardise(descriptors(a_tr)[sub], descriptors(b_tr)[sub])
    fit = fit_theta(SquaredDifferenceFeatures(da_sub, db_sub), rel=rel,
                    max_iterations=int(config.max_iterations))
    theta = fit["theta"]
    if theta is None:
        raise SystemExit(f"training fit failed: {fit}")
    log.line(f"learned theta fitted on {config.train_pair} in {time.perf_counter() - started:.1f}s: "
             f"xyz={theta[0:3].sum():.3f} area={theta[3]:.3f} curv={theta[4]:.3f} "
             f"hks={theta[5:].sum():.3f}")

    records: list[RunRecord] = []
    for start in range(0, len(config.pairs), 2):
        pair = (int(config.pairs[start]), int(config.pairs[start + 1]))
        geometry = f"pair{pair[0]:03d}-{pair[1]:03d}"
        log.line(f"[{geometry}] start (full 6890, geodesic matrix)")
        a_m = read_ply(root / f"tr_reg_{pair[0]:03d}.ply")
        b_m = read_ply(root / f"tr_reg_{pair[1]:03d}.ply")
        da, db = standardise(descriptors(a_m), descriptors(b_m))
        n = int(da.shape[0])
        all_idx = np.arange(n, dtype=np.int64)
        geo = geodesic_matrix(b_m.vertices, b_m.faces, all_idx)
        norm_area = float(np.sqrt(surface_area(b_m.vertices, b_m.faces)))
        # EN: the geodesic diameter is the max of the full n x n matrix -- the length scale the leaderboard
        #     convention divides by. It costs nothing extra because the matrix already exists.
        # CN: 测地直径 = 全矩阵最大值，不需要额外计算。
        diameter = float(np.max(geo))
        log.line(f"[{geometry}] sqrt(area)={norm_area:.6f}  diameter={diameter:.6f}  "
                 f"diameter/sqrt(area)={diameter / norm_area:.6f}")

        o_sub = np.sort(np.random.default_rng(0).permutation(n)[: int(config.n_sub)])
        inner_oracle = fit_theta(SquaredDifferenceFeatures(da[o_sub], db[o_sub]), rel=rel,
                                 max_iterations=int(config.max_iterations))
        thetas = {
            "position_only": np.asarray([1.0, 1.0, 1.0] + [0.0] * (da.shape[1] - 3)),
            "uniform_theta": np.full(da.shape[1], 1.0 / da.shape[1]),
            "learned": theta,
        }
        if inner_oracle["theta"] is not None:
            thetas["oracle"] = inner_oracle["theta"]

        for arm, th in thetas.items():
            pred = match(cost_matrix(da, db, th), num_itermax=20_000_000)
            raw = np.asarray(geo)[np.arange(n), np.asarray(pred, dtype=np.int64)]
            # EN: both normalisations from the SAME raw distances.
            # CN: 同一批原始距离的两种归一化。
            by_area = raw / norm_area
            by_diameter = raw / diameter
            check = float(np.max(np.abs(by_area * (norm_area / diameter) - by_diameter)))
            metrics = {
                "mean_area": float(by_area.mean()),
                "median_area": float(np.median(by_area)),
                "miss5pct_area": float(np.mean(by_area > 0.05)),
                "mean_diameter": float(by_diameter.mean()),
                "median_diameter": float(np.median(by_diameter)),
                "miss5pct_diameter": float(np.mean(by_diameter > 0.05)),
                "relation_max_abs_error": check,
            }
            metrics.update({f"diam_{key}": value
                            for key, value in accuracy_curve(by_diameter, THRESHOLDS).items()})
            metrics.update({f"area_{key}": value
                            for key, value in accuracy_curve(by_area, THRESHOLDS).items()})
            records.append(RunRecord(study=str(config.study), geometry=geometry, arm=str(arm),
                                     k=int(da.shape[1]), m=n, draw=0, metrics=metrics,
                                     config=config.to_dict(),
                                     extra={"sqrt_area": norm_area, "diameter": diameter,
                                            "ratio_diameter_over_sqrt_area": diameter / norm_area,
                                            "theta": [float(v) for v in np.asarray(th)]}))
            log.line(f"[{geometry}] {arm:<14} mean(area)={metrics['mean_area']:.4f} "
                     f"mean(diam)={metrics['mean_diameter']:.4f} "
                     f"miss5(area)={metrics['miss5pct_area']:.3f} "
                     f"miss5(diam)={metrics['miss5pct_diameter']:.3f} "
                     f"acc@5%(diam)={metrics['diam_acc@0.05']:.3f} "
                     f"relation_err={check:.2e}")

    write_records(run_dir / "records.json", records, provenance=PROVENANCE)
    log.line(f"wrote {run_dir / 'records.json'}")
    log.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
