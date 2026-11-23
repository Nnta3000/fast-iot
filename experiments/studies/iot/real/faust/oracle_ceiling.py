"""
EN: **Oracle ceiling for richer cost dictionaries — the gate before any kernel claim.**

    Question, fixed before the measurement: does a richer `FeatureMap` raise the achievable `miss@5 %` on FAUST,
    when `theta` is fitted on the very pair being evaluated?

    This is deliberately an **oracle** experiment: `theta` is fitted on the evaluated pair, so the number is not a
    result of any transferable method. It answers only one thing — *is the 13-feature family the binding
    constraint, or is it something else?* If a rich dictionary cannot beat the existing numbers even with the
    answer visible, no amount of kernel machinery will help the transferable case, and the axis stops here.

    **Baselines to beat** (same pairs, same `n = 800`, same vertex subsamples, 24 rows each), from
    `out/faust_correspondence_box/`:

        learned_selected  miss@5% = 0.1855     <- the best number actually achieved, by TRANSFER
        oracle_theta      miss@5% = 0.1945     <- the oracle of the 13-feature family
        uniform_theta     miss@5% = 0.4202

    The gate is stated against **0.1855**, not 0.1945: a transferred fit already beats the 13-feature oracle, so
    the oracle is the optimum of a different objective (the max-margin regret), not an upper bound on quality.

    Dictionaries compared, all `FeatureMap`s, so the solver / scan / loop / master are untouched:

      `diag13`     `DiagonalWeightFeatures` on the 13 descriptors — reproduces the baseline
      `gauss`      Gaussian kernels on the descriptor vectors at several bandwidths (`-k(x_i,y_j)`)
      `gauss+cos`  the same plus a cosine kernel, i.e. a two-shape radial profile
      `profile8`   hat-basis radial profiles per coordinate, `K = 13 * 8 = 104`

    Reported per dictionary: `miss@5 %`, `miss@1 %`, mean geodesic error, and `L1(theta, uniform)` — because a
    large basis dilutes the simplex and a fit that is numerically uniform has not learned anything.

CN: **更丰富代价字典的 oracle 上限——任何核主张之前的门槛。** 在**被评估的那一对**上拟合 `theta`，只回答一个问题：
    13 特征族是不是瓶颈。若丰富字典连"看得见答案"都赢不了现有数字，核方向到此为止。门槛定在 0.1855（我们真正
    达到过的最好值，来自迁移），而不是 0.1945（13 特征族的 oracle）。
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
import time
from pathlib import Path

import numpy as np
from jaxtyping import Float, Int

_REPO_ROOT = Path(__file__).resolve().parents[5]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from iot.core.features import (  # noqa: E402
    ConcatFeatures,
    DiagonalWeightFeatures,
    HingeProfileFeatures,
    FeatureMap,
    KernelMixtureFeatures,
    RadialProfileFeatures,
)
from experiments.harness.config import RunConfig, build_parser  # noqa: E402
from experiments.harness.faust import default_registrations_root, geodesic_matrix, read_ply, shared_index, surface_area  # noqa: E402
from experiments.harness.faust_match import geodesic_errors, match  # noqa: E402
from experiments.harness.preflight import check as preflight_check  # noqa: E402
from experiments.harness.records import RunLog, RunRecord, write_records  # noqa: E402
from experiments.studies.iot.real.faust.correspondence import (  # noqa: E402
    descriptors,
    fit_theta,
    standardise,
)

DEFAULT_ROOT = str(default_registrations_root())

PROVENANCE: dict[str, object] = {
    "question": "does a richer cost dictionary raise the achievable miss@5% on FAUST at all (oracle theta)?",
    "oracle": "theta is fitted on the EVALUATED pair itself; this is not a transferable result",
    "gate": "stop the kernel axis unless the oracle beats 0.1855 (the best transferred number we have)",
    "dictionaries": ["diag13 (baseline)", "gauss (multi-bandwidth)", "gauss+cos", "profile8 (hat basis)"],
    "solver": "the existing regret loop, simplex only (rel=None); no solver change",
    "metric": "geodesic on the target / sqrt(area), miss@5% and miss@1%",
    "why_not_0.1945": "a transferred fit (learned_selected, 0.1855) already beats the 13-feature oracle, so the "
                       "oracle is the optimum of a different objective, not a quality ceiling",
}


@dataclasses.dataclass(frozen=True)
class CeilingConfig(RunConfig):
    """EN: knobs; real dataclass fields so `from_namespace` cannot drop them. CN: 旋钮。"""

    section = "faust_oracle_ceiling"
    study = "faust_oracle_ceiling"
    root: str = DEFAULT_ROOT
    n_sub: int = 800
    seeds: tuple[int, ...] = (0, 1, 2)
    test_pairs: tuple[int, ...] = (50, 66, 53, 85, 56, 71, 59, 99,
                                  61, 86, 64, 72, 67, 91, 72, 85)
    max_iterations: int = 5
    bands: int = 5
    basis_size: int = 8
    families: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """EN: normalise and validate. CN: 归一化与校验。"""
        object.__setattr__(self, "seeds", tuple(int(v) for v in self.seeds))
        object.__setattr__(self, "families", tuple(str(v) for v in self.families))
        object.__setattr__(self, "test_pairs", tuple(int(v) for v in self.test_pairs))
        if len(self.test_pairs) % 2 != 0 or not self.test_pairs:
            raise ValueError("--test-pairs needs an even, non-empty list")
        if int(self.n_sub) < 2:
            raise ValueError("--n-sub must be at least 2")
        if int(self.bands) < 1 or int(self.basis_size) < 2:
            raise ValueError("--bands >= 1 and --basis-size >= 2")

    @classmethod
    def add_study_arguments(cls, parser: argparse.ArgumentParser) -> None:
        """EN: only this study's flags. CN: 只加本研究特有的参数。"""
        parser.add_argument("--root", default=DEFAULT_ROOT)
        parser.add_argument("--n-sub", type=int, default=800)
        parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
        parser.add_argument("--test-pairs", type=int, nargs="+", default=None)
        parser.add_argument("--max-iterations", type=int, default=5)
        parser.add_argument("--bands", type=int, default=5)
        parser.add_argument("--basis-size", type=int, default=8)
        parser.add_argument("--families", nargs="+", default=None,
                            help="subset of dictionaries to run (default: all)")


def _nearest_neighbour_scale(source: Float[np.ndarray, "m d"], target: Float[np.ndarray, "n d"]
                             ) -> tuple[float, Float[np.ndarray, "d"]]:
    """
    EN: two scales taken from **plausible matches**, not from random pairs.

        `s_match` is the median distance from a source descriptor to its nearest target descriptor — i.e. the
        typical distance between a point and its best candidate. A Gaussian bandwidth must be on this order to
        discriminate at all: with `sigma` set from *random* pair distances (my first version) every plausible
        match has `k ≈ 1` and every implausible one `k ≈ 0`, so the cost saturates and carries no ranking.

        `error_spread` is the per-coordinate 99th percentile of `|a_k - b_nn,k|` — the spread of the *residual*
        for good matches. That is the range a radial basis must resolve; using the spread of random pairs instead
        puts every match in the first hat function and makes the cost constant across candidates (measured:
        `theta` did not move at all, `L1(uniform) = 0.000`).

    CN: 两个尺度都取自**可能的匹配**，而不是随机对。`s_match` 是"源点到最近目标点"的中位距离，高斯带宽必须
        在这个量级才能区分；`error_spread` 是良好匹配残差的逐坐标 99 分位，是径向基必须分辨的范围。
    """
    row_nn_distance, row_nn_index = _row_nearest(source, target)
    s_match = float(np.median(row_nn_distance))
    s_match = max(s_match, 1e-9)
    residual = np.abs(np.asarray(source, dtype=np.float64)
                      - np.asarray(target, dtype=np.float64)[row_nn_index])
    error_spread = np.maximum(np.percentile(residual, 99.0, axis=0), 1e-12)
    return s_match, np.ascontiguousarray(error_spread, dtype=np.float64)


def _row_nearest(source: Float[np.ndarray, "m d"], target: Float[np.ndarray, "n d"]
                 ) -> tuple[Float[np.ndarray, "m"], Int[np.ndarray, "m"]]:
    """EN: for each source row, the distance and index of its nearest target row. CN: 每个源行的最近目标行。"""
    src = np.asarray(source, dtype=np.float64)
    tgt = np.asarray(target, dtype=np.float64)
    best_distance = np.full(src.shape[0], np.inf, dtype=np.float64)
    best_index = np.zeros(src.shape[0], dtype=np.int64)
    # EN: blocked so the (m, chunk, d) temporary stays small for large n.
    # CN: 分块计算，避免大 n 时的临时张量。
    chunk = max(1, int(4_000_000 / max(1, tgt.shape[0] * tgt.shape[1])))
    for start in range(0, src.shape[0], chunk):
        block = src[start:start + chunk]
        difference = block[:, None, :] - tgt[None, :, :]
        squared = np.einsum("ijd,ijd->ij", difference, difference)
        index = np.argmin(squared, axis=1)
        best_index[start:start + chunk] = index
        best_distance[start:start + chunk] = np.sqrt(squared[np.arange(block.shape[0]), index])
    return best_distance, best_index


def _bandwidths(source: Float[np.ndarray, "m d"], target: Float[np.ndarray, "n d"], *, bands: int,
                sample: int = 4096, seed: int = 0) -> tuple[float, ...]:
    """
    EN: Gaussian bandwidths as multiples of the nearest-neighbour scale: `sigma in {0.5, 1, 2, 4, 8} x s_match`.

        A mixture spanning "tighter than a match" to "much looser" is the radial-profile dictionary; the point of
        the mixture is that `theta` chooses where on that profile the cost lives, not that any single `sigma` is
        right.

    CN: 高斯带宽取最近邻尺度的倍数，覆盖"比一次匹配更紧"到"宽松得多"；由 `theta` 决定代价落在剖面的哪一段。
    """
    s_match, _error_spread = _nearest_neighbour_scale(source, target)
    multiples = np.geomspace(0.5, 8.0, int(bands))
    return tuple(float(value) for value in np.unique(np.round(multiples * s_match, 9)))


def _dictionaries(a: Float[np.ndarray, "m d"], b: Float[np.ndarray, "n d"], *,
                  bands: int, basis_size: int) -> list[tuple[str, FeatureMap]]:
    """
    EN: the dictionaries under test. `diag13` is the incumbent and must reproduce the baseline; the rest are the
        candidate richer families. Every one is a plain `FeatureMap`.
    CN: 待测字典。`diag13` 是现有基线（必须复现），其余是候选的更丰富族；全部都是普通 `FeatureMap`。
    """
    sigmas = _bandwidths(a, b, bands=bands)
    _s_match, error_spread = _nearest_neighbour_scale(a, b)
    gaussian_specs = tuple(f"gaussian:{value:g}" for value in sigmas)

    # EN: **the dictionary that can actually change the ranking.** Kernels on DIFFERENT signals are not monotone
    #     transforms of one distance, so `theta` can express a genuine preference. Groups match the descriptor
    #     layout in `faust.descriptors`: xyz (3), log 1-ring area (1), log curvature (1), 8 HKS columns.
    # CN: **真正能改变排序的字典。** 不同信号上的核不是同一距离的单调变换，`theta` 才能表达真实偏好。
    def group(start: int, stop: int, name: str) -> tuple[str, FeatureMap]:
        """EN: Gaussian mixture on one descriptor group. CN: 单个描述子组上的高斯混合。"""
        part_a, part_b = a[:, start:stop], b[:, start:stop]
        part_sigmas = _bandwidths(part_a, part_b, bands=max(2, int(bands)))
        return (name, KernelMixtureFeatures(part_a, part_b,
                                            tuple(f"gaussian:{value:g}" for value in part_sigmas)))

    groups = [group(0, 3, "xyz"), group(3, 4, "area"), group(4, 5, "curv"), group(5, a.shape[1], "hks")]
    return [
        ("diag13", DiagonalWeightFeatures(a, b)),
        ("gauss", KernelMixtureFeatures(a, b, gaussian_specs)),
        ("gauss+cos", KernelMixtureFeatures(a, b, gaussian_specs + ("cosine",))),
        ("profile8", RadialProfileFeatures(a, b, basis_size=int(basis_size), extent=error_spread)),
        # EN: the same radial profile with the constant (null) direction removed -- see HingeProfileFeatures.
        # CN: 同一径向剖面，去掉了常数（零）方向。
        # CN: 铰链基**不能**用最近邻残差尺度归一化：那个尺度太小，几乎所有候选距离都被截断到 1，代价饱和成常数
        #     （实测 tied=1.000，共享 θ 解码 miss 0.92）。用随机对的 99 分位才覆盖候选距离的真实范围。
        # EN: the hinge basis must NOT use the NN residual spread as its scale -- that is tiny, so nearly every
        #     candidate distance clips to 1 and the cost saturates to a constant (measured tied=1.000). Its own
        #     default (99th percentile over RANDOM pairs) spans the real range: 0.0325 vs 0.1163 on pair 050-066.
        ("hinge8", HingeProfileFeatures(a, b, basis_size=int(basis_size))),
        ("group_gauss", ConcatFeatures(tuple(item for _name, item in groups))),
        ("group+diag", ConcatFeatures((DiagonalWeightFeatures(a, b),
                                       *(item for _name, item in groups)))),
    ]


def _score(features: FeatureMap, theta: Float[np.ndarray, "k"], *, n: int,
           geo: Float[np.ndarray, "n n"], norm: float) -> dict[str, float]:
    """
    EN: score a `FeatureMap` on its own cost. This cannot go through `faust_match.map_metrics`, because that
        helper builds the cost as a diagonal `theta`-weighted distance and would silently ignore any richer
        feature family — which is exactly the bug this function replaces (it crashed with a shape mismatch, so
        at least it failed loudly).

        The cost is materialised for all `n*n` edges and decoded with the same exact OT + row-argmax used
        everywhere else, so the metric is identical across dictionaries.

    CN: 用 `FeatureMap` 自己的代价来评分。不能用 `faust_match.map_metrics`——它按对角 `theta` 加权距离构造代价，
        会忽略任何更丰富的特征族。这里显式物化 `n*n` 边的代价，用同一套精确 OT + 行 argmax 解码。
    """
    rows = np.repeat(np.arange(int(n)), int(n)).astype(np.int64)
    cols = np.tile(np.arange(int(n)), int(n)).astype(np.int64)
    cost = np.asarray(features.costs(theta, rows, cols), dtype=np.float64).reshape(int(n), int(n))
    pred = np.asarray(match(np.ascontiguousarray(cost)), dtype=np.int64)
    return geodesic_errors(pred, geo, norm)


def main(argv: list[str] | None = None) -> int:
    """EN: entry point. CN: 入口。"""
    parser = build_parser(CeilingConfig)
    args = parser.parse_args(list(argv) if argv is not None else None)
    config = CeilingConfig.from_namespace(args, parser)
    for flag in ("n_sub", "seeds", "max_iterations", "bands", "basis_size"):
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
    log.line(f"oracle ceiling: n_sub={config.n_sub} seeds={config.seeds} "
             f"pairs={len(config.test_pairs) // 2} bands={config.bands} basis={config.basis_size}")
    preflight = preflight_check(run_tiny=False, need_gpu=False)
    preflight.write(run_dir / "preflight.json")
    log.line(f"preflight ok={preflight.ok} mem={preflight.mem_available_gb:.1f} GB")
    log.line("GATE: a richer dictionary must beat 0.1855 miss@5% (transferred), not 0.1945 (13-feature oracle)")

    root = Path(config.root)
    records: list[RunRecord] = []
    for start in range(0, len(config.test_pairs), 2):
        a_id, b_id = int(config.test_pairs[start]), int(config.test_pairs[start + 1])
        geometry = f"pair{a_id:03d}-{b_id:03d}"
        a_mesh = read_ply(root / f"tr_reg_{a_id:03d}.ply")
        b_mesh = read_ply(root / f"tr_reg_{b_id:03d}.ply")
        for seed in config.seeds:
            index = shared_index(a_mesh, int(config.n_sub), seed=int(seed))
            da, db = standardise(descriptors(a_mesh)[index], descriptors(b_mesh)[index])
            geometry_id = index.astype(np.int64)
            geo = geodesic_matrix(b_mesh.vertices, b_mesh.faces, geometry_id)
            norm = float(np.sqrt(surface_area(b_mesh.vertices, b_mesh.faces)))
            available = _dictionaries(da, db, bands=int(config.bands),
                                      basis_size=int(config.basis_size))
            if config.families:
                missing = set(config.families) - {name for name, _ in available}
                if missing:
                    raise SystemExit(f"unknown --families {sorted(missing)}; known "
                                     f"{sorted(name for name, _ in available)}")
                available = [(name, features) for name, features in available
                             if name in set(config.families)]
            for name, features in available:
                started = time.perf_counter()
                fit = fit_theta(features, rel=None, max_iterations=int(config.max_iterations))
                theta = fit["theta"]
                if theta is None:
                    records.append(RunRecord(
                        study=str(config.study), geometry=geometry, arm=str(name), k=int(features.n_features),
                        m=int(da.shape[0]), draw=int(seed), metrics={}, config=config.to_dict(),
                        excluded=True, status="FAILED fit", extra={"regret": fit.get("regret")}))
                    log.line(f"[{geometry} s{seed}] {name:>10} FAILED fit")
                    continue
                theta = np.asarray(theta, dtype=np.float64)
                metrics = _score(features, theta, n=int(da.shape[0]), geo=geo, norm=norm)
                uniform_l1 = float(np.abs(theta - 1.0 / theta.size).sum())
                records.append(RunRecord(
                    study=str(config.study), geometry=geometry, arm=str(name), k=int(features.n_features),
                    m=int(da.shape[0]), draw=int(seed),
                    metrics={"mean_error": metrics["mean"], "median_error": metrics["median"],
                             "miss_1pct": metrics["miss_1pct"], "miss_5pct": metrics["miss_5pct"]},
                    config=config.to_dict(),
                    extra={"n_features": int(features.n_features), "l1_to_uniform": uniform_l1,
                           "regret": fit.get("regret"), "oracle": True,
                           "seconds": round(time.perf_counter() - started, 2)}))
                log.line(f"[{geometry} s{seed}] {name:>10} K={features.n_features:>4} "
                         f"miss5%={metrics['miss_5pct']:.4f} miss1%={metrics['miss_1pct']:.4f} "
                         f"mean={metrics['mean']:.4f} L1(unif)={uniform_l1:.3f} "
                         f"{time.perf_counter() - started:.1f}s")

    write_records(run_dir / "records.json", records, provenance=PROVENANCE)

    # EN: the verdict, computed from the record rather than eyeballed.
    # CN: 结论由记录计算得出，不靠目测。
    import collections
    import statistics as stats
    pooled: dict[str, list[float]] = collections.defaultdict(list)
    for record in records:
        if not record.excluded and "miss_5pct" in record.metrics:
            pooled[str(record.arm)].append(float(record.metrics["miss_5pct"]))
    log.line("--- pooled over pairs x seeds ---")
    best_other = float("inf")
    lines: dict[str, float] = {}
    for arm, values in sorted(pooled.items(), key=lambda kv: stats.mean(kv[1])):
        mean = stats.mean(values)
        lines[arm] = mean
        log.line(f"  {arm:>10} miss@5% = {mean:.4f}  (n={len(values)})")
        if arm != "diag13":
            best_other = min(best_other, mean)
    baseline = lines.get("diag13", float("nan"))
    log.line(f"baseline diag13 = {baseline:.4f}; best richer = {best_other:.4f}")
    log.line(f"GATE vs 0.1855: {'PASS — there is headroom' if best_other < 0.1855 else 'FAIL — the dictionary is not the constraint'}")
    log.line(f"wrote {run_dir / 'records.json'}")
    log.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
