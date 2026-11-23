"""
EN: **The two finalists, at full resolution, with the grid extended downward.**

    §4f compared four objectives at `n = 1500`, one seed, and every learned arm selected its
    hyperparameter at the *boundary* of its grid. That makes the relaxed LP's 0.026 miss@5 % advantage an
    upper bound rather than a measurement. This module removes both limits for the two arms that could be a
    submission baseline:

      | arm | objective | selected by |
      | --- | --- | --- |
      | `row_softmax` | `mean_i CE(-C_ij/tau, i)` | `tau` on validation pairs |
      | `rowcol_softmax` | `mean_i -log P_ii`, `P = Sinkhorn(exp(-C/tau))` | `tau` on validation pairs |
      | `relaxed_lp` | the regret master (simplex only) | no hyperparameter |

    What changes relative to §4f:

    1. **`n = 6890`** (the full registration, no vertex subsample) instead of 1500.
    2. **`tau` grid extended downward** to `{1e-3, 3e-3, 1e-2, 3e-2}`, because every arm in §4f picked the
       smallest value it was offered -- the optimum was outside the grid.
    3. **600 Adam steps** instead of 250, so the supervised arms are not handicapped by an iteration cap.
    4. **3 vertex-subsample seeds**, so the number carries a spread instead of being one draw.
    5. **Paired bootstrap over the 8 held-out pairs**, which is the only comparison the claim can rest on.

    Memory. The Sinkhorn is wrapped in one `torch.utils.checkpoint` segment: without it, 10 unrolled
    iterations retain ~30 `n x n` tensors (5.7 GB at `n = 6890`) and the probe died with a CUDA OOM while a
    foreign service held 4.6 GB of the 8 GB card. With the single checkpoint the retained footprint is one
    `n x n` tensor plus the recomputation workspace (~1.2 GB), and the fits run on the CPU if the GPU does
    not have that much free.

    CN: **两个 finalist 在全分辨率下的正面对决**：`n=6890`、`tau` 网格向下扩展（§4f 里每个臂都选在网格边界）、
        600 步 Adam、3 个顶点子采样种子、8 个留出对上的配对 bootstrap。Sinkhorn 用单段 checkpoint 包裹，否则
        10 次展开会保留约 30 个 `n×n` 张量（n=6890 时 5.7 GB），探测时已因此 CUDA OOM。

    Run:
        .venv/bin/python -u -m experiments.studies.iot.real.faust.finalists --n-sub 6890
"""

from __future__ import annotations

import dataclasses
import sys
import time
from pathlib import Path

import numpy as np
import torch
from jaxtyping import Float, Int

_REPO_ROOT = Path(__file__).resolve().parents[4]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from iot.core.features import SquaredDifferenceFeatures  # noqa: E402
from experiments.harness.config import RunConfig, build_parser  # noqa: E402
from experiments.harness.faust import default_registrations_root, read_ply  # noqa: E402
from experiments.harness.preflight import check as preflight_check  # noqa: E402
from experiments.harness.records import (  # noqa: E402
    RunLog,
    RunRecord,
    format_table,
    summarise,
    write_records,
)
from experiments.studies.iot.real.faust.correspondence import (  # noqa: E402
    _sample_index,
    default_pairs,
    descriptors,
    fit_theta,
    mirror_indices,
    standardise,
)
from experiments.harness.faust import geodesic_matrix, surface_area  # noqa: E402
from experiments.harness.faust_match import (  # noqa: E402
    bootstrap_contrast,
    fit_softmax,
    free_vram_gb,
    map_metrics,
)

DEFAULT_ROOT = str(default_registrations_root())

PROVENANCE: dict[str, object] = {
    "question": "at full resolution and with the grid extended downward, does the relaxed LP still beat the "
                "two supervised reweightings of the identical features?",
    "hypothesis_class": "c_ij = sum_k theta_k (a_ik - b_jk)^2, theta on the simplex, the same standardised "
                        "13 descriptors in every arm",
    "difference_from_arms_study": "n=6890 (full registration) instead of 1500; tau grid extended down to 1e-3; "
                                  "600 Adam steps instead of 250; 3 vertex-subsample seeds; paired bootstrap "
                                  "over the 8 held-out pairs",
    "selection": "tau chosen per arm on validation pairs drawn from the TRAINING subjects (2 pairs, seed 0); "
                 "the relaxed LP is simplex-only and has no hyperparameter",
    "solver": "softmax arms: torch autograd (single-checkpointed Sinkhorn, 10 iterations), Adam; relaxed_lp: "
              "the regret loop (CuPDLPx); evaluation: exact POT (ot.emd) at n=6890",
    "metric": "geodesic error on the target, normalised by sqrt(surface area) -- identical to the main study",
    "memory": "fits keep one n x n tensor plus workspace (~1.2 GB); evaluation is CPU, peak ~1.6 GB host",
}


@dataclasses.dataclass(frozen=True)
class FinalistsConfig(RunConfig):
    """CN: 旋钮。EN: knobs."""

    study: str = "faust_finalists"
    section: str = "faust_finalists"
    root: str = DEFAULT_ROOT
    train_pair: tuple[int, ...] = (0, 13)
    train_subjects: tuple[int, ...] = (0, 1, 2, 3, 4)
    test_subjects: tuple[int, ...] = (5, 6, 7, 8, 9)
    test_pair_count: int = 8
    val_pair_count: int = 2
    n_sub: int = 6890
    seeds: tuple[int, ...] = (0, 1, 2)
    steps: int = 600
    lr: float = 0.05
    tau_grid: tuple[float, ...] = (1e-3, 3e-3, 1e-2, 3e-2)
    sinkhorn_iterations: int = 10
    max_iterations: int = 30
    fit_lp: int = 1

    @classmethod
    def add_study_arguments(cls, parser) -> None:
        """CN: 只加本研究特有的参数。EN: only this study's flags."""
        parser.add_argument("--root", default=DEFAULT_ROOT)
        parser.add_argument("--train-pair", type=int, nargs=2, default=[0, 13])
        parser.add_argument("--train-subjects", type=int, nargs="+", default=[0, 1, 2, 3, 4])
        parser.add_argument("--test-subjects", type=int, nargs="+", default=[5, 6, 7, 8, 9])
        parser.add_argument("--test-pair-count", type=int, default=8)
        parser.add_argument("--val-pair-count", type=int, default=2)
        parser.add_argument("--n-sub", type=int, default=6890)
        parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
        parser.add_argument("--steps", type=int, default=600)
        parser.add_argument("--lr", type=float, default=0.05)
        parser.add_argument("--tau-grid", type=float, nargs="+", default=[1e-3, 3e-3, 1e-2, 3e-2])
        parser.add_argument("--sinkhorn-iterations", type=int, default=10)
        parser.add_argument("--max-iterations", type=int, default=30)
        parser.add_argument("--fit-lp", type=int, default=1)


def main(argv: list[str] | None = None) -> int:
    """CN: 入口。EN: entry point."""
    parser = build_parser(FinalistsConfig)
    args = parser.parse_args(list(argv) if argv is not None else None)
    config = FinalistsConfig.from_namespace(args, parser)
    run_dir = config.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    log = RunLog(run_dir / "run.log")
    pre = preflight_check(run_tiny=False, need_gpu=False)
    pre.write(run_dir / "preflight.json")
    free = free_vram_gb()
    # CN: 单段 checkpoint 需要在显存里留一个 `n×n` 张量加工作区；低于 2.5 GB 就退回 CPU。
    # EN: the single checkpoint needs one `n x n` tensor plus workspace; below 2.5 GB free we use the CPU.
    device = "cuda" if free > 2.5 else "cpu"
    log.line(f"preflight ok={pre.ok} mem={pre.mem_available_gb:.1f} GB host, {free:.1f} GB free VRAM "
             f"-> fits on {device}")

    root = Path(config.root)
    train_pair = (int(config.train_pair[0]), int(config.train_pair[1]))
    seeds = tuple(int(v) for v in config.seeds)
    test_pairs = default_pairs(tuple(int(v) for v in config.test_subjects), int(config.test_pair_count))
    pool = default_pairs(tuple(int(v) for v in config.train_subjects), 4 * int(config.val_pair_count) + 16)
    val_pairs = [p for p in pool if p != train_pair][: int(config.val_pair_count)]
    log.line(f"train pair {train_pair}; validation {val_pairs}; test {test_pairs}; seeds {seeds}")

    a_tr = read_ply(root / f"tr_reg_{train_pair[0]:03d}.ply")
    b_tr = read_ply(root / f"tr_reg_{train_pair[1]:03d}.ply")

    # CN: **第一步：选 `tau`**，只用训练受试者的验证对，只用种子 0（选择不该消耗三个种子）。
    # EN: **step 1: select `tau`** on training-subject validation pairs, with seed 0 only (selection should
    #     not spend three seeds).
    log.line("\n--- tau selection on training-subject validation pairs (seed 0, n=%d) ---" % int(config.n_sub))
    idx0 = _sample_index(a_tr, config, seeds[0])
    a0, b0 = standardise(descriptors(a_tr)[idx0], descriptors(b_tr)[idx0])
    val_cache: list[tuple[Float[np.ndarray, "n k"], Float[np.ndarray, "n k"],
                          Float[np.ndarray, "n n"], float, Int[np.ndarray, "n"]]] = []
    for pair in val_pairs:
        va = read_ply(root / f"tr_reg_{pair[0]:03d}.ply")
        vb = read_ply(root / f"tr_reg_{pair[1]:03d}.ply")
        v_idx = _sample_index(va, config, seeds[0])
        va_desc, vb_desc = standardise(descriptors(va)[v_idx], descriptors(vb)[v_idx])
        geo = geodesic_matrix(vb.vertices, vb.faces, v_idx)
        norm = float(np.sqrt(surface_area(vb.vertices, vb.faces)))
        mirror, _res = mirror_indices(vb.vertices, v_idx)
        val_cache.append((va_desc, vb_desc, geo, norm, mirror))
        log.line(f"   validation pair {pair}: geodesic matrix {geo.shape[0]}x{geo.shape[1]} ready")

    selected: dict[str, float] = {}
    for name, two_sided in (("row_softmax", False), ("rowcol_softmax", True)):
        scored: list[tuple[float, float]] = []
        for tau in config.tau_grid:
            started = time.perf_counter()
            theta = fit_softmax(a0, b0, tau=float(tau), two_sided=two_sided, steps=int(config.steps),
                                lr=float(config.lr), sinkhorn_iterations=int(config.sinkhorn_iterations),
                                device=device)
            scores = [float(map_metrics(va, vb, theta, geo, norm, mirror)["miss_5pct"])
                      for va, vb, geo, norm, mirror in val_cache]
            mean = float(np.mean(scores))
            scored.append((float(tau), mean))
            log.line(f"   {name:<16} tau={tau:<8g} val miss@5%={mean:.4f} "
                     f"(fit {time.perf_counter() - started:.1f}s)")
        selected[name] = min(scored, key=lambda item: item[1])[0]
    log.line(f"selected tau: row={selected['row_softmax']:g}, rowcol={selected['rowcol_softmax']:g}")

    # CN: **第二步：按选中的 `tau` 在三个种子上重新拟合**，以及每个种子的松弛 LP。
    # EN: **step 2: refit at the selected `tau` for every seed**, plus the relaxed LP per seed.
    log.line("\n--- fits on the training pair (n=%d, seeds=%s) ---" % (int(config.n_sub), seeds))
    thetas: dict[tuple[str, int], Float[np.ndarray, "k"]] = {}
    for seed in seeds:
        idx = _sample_index(a_tr, config, seed)
        a_seed, b_seed = standardise(descriptors(a_tr)[idx], descriptors(b_tr)[idx])
        thetas[("row_softmax", seed)] = fit_softmax(
            a_seed, b_seed, tau=selected["row_softmax"], two_sided=False, steps=int(config.steps),
            lr=float(config.lr), sinkhorn_iterations=int(config.sinkhorn_iterations), device=device)
        thetas[("rowcol_softmax", seed)] = fit_softmax(
            a_seed, b_seed, tau=selected["rowcol_softmax"], two_sided=True, steps=int(config.steps),
            lr=float(config.lr), sinkhorn_iterations=int(config.sinkhorn_iterations), device=device)
        if int(config.fit_lp):
            started = time.perf_counter()
            fit = fit_theta(SquaredDifferenceFeatures(a_seed, b_seed), rel=None,
                            max_iterations=int(config.max_iterations))
            if fit["theta"] is not None:
                thetas[("relaxed_lp", seed)] = np.asarray(fit["theta"], dtype=np.float64)
            log.line(f"   [seed {seed}] relaxed_lp regret={fit['regret']:.4g} "
                     f"converged={fit['converged']} ({time.perf_counter() - started:.1f}s)")
        for name in ("row_softmax", "rowcol_softmax"):
            theta = thetas[(name, seed)]
            log.line(f"   [seed {seed}] {name:<16} L1-to-uniform="
                     f"{float(np.abs(theta - 1.0 / theta.size).sum()):.3f}  "
                     f"max={float(theta.max()):.3f}  exact_zeros={int(np.sum(theta <= 1e-12))}")
        if device == "cuda":
            torch.cuda.empty_cache()

    # CN: **第三步：8 个留出对上的测试**，每对每种子只算一次测地矩阵，然后对所有 θ 复用。
    # EN: **step 3: the 8 held-out pairs**, with the geodesic matrix computed once per (pair, seed) and reused
    #     across every theta.
    log.line("\n--- evaluation on held-out pairs ---")
    rows: list[RunRecord] = []
    for pair in test_pairs:
        ta = read_ply(root / f"tr_reg_{pair[0]:03d}.ply")
        tb = read_ply(root / f"tr_reg_{pair[1]:03d}.ply")
        geometry = f"pair{pair[0]:03d}-{pair[1]:03d}"
        for seed in seeds:
            idx = _sample_index(ta, config, seed)
            a_test, b_test = standardise(descriptors(ta)[idx], descriptors(tb)[idx])
            geo = geodesic_matrix(tb.vertices, tb.faces, idx)
            norm = float(np.sqrt(surface_area(tb.vertices, tb.faces)))
            mirror, _res = mirror_indices(tb.vertices, idx)
            k = int(a_test.shape[1])
            candidates = {name: thetas[(name, seed)] for name in ("row_softmax", "rowcol_softmax",
                                                                 "relaxed_lp") if (name, seed) in thetas}
            candidates["uniform"] = np.full(k, 1.0 / k)
            for name, theta in candidates.items():
                metrics = map_metrics(a_test, b_test, theta, geo, norm, mirror)
                rows.append(RunRecord(study=str(config.study), geometry=geometry, arm=name, k=k,
                                      m=int(idx.size), draw=int(seed), metrics=metrics,
                                      config=config.to_dict()))
            log.line(f"   {geometry} seed {seed}: " + "  ".join(
                f"{name}={rows[-len(candidates) + i].metrics['miss_5pct']:.3f}"
                for i, name in enumerate(candidates)))
            del geo
    log.line(f"selection: {selected}")

    # CN: **配对 bootstrap**：先在 (对, 种子) 上取均值，再按对重采样，于是 CI 反映的是"对"的变异。
    # EN: **paired bootstrap**: average over seeds within a pair first, then resample pairs, so the CI
    #     reflects pair-to-pair variation rather than seed noise.
    per_pair: dict[str, dict[str, float]] = {}
    for row in rows:
        per_pair.setdefault(row.geometry, {})[row.arm] = row.metrics["miss_5pct"]
    arms = sorted({arm for values in per_pair.values() for arm in values})
    contrasts = bootstrap_contrast(per_pair, base="relaxed_lp",
                                   others=[arm for arm in arms if arm != "relaxed_lp"])
    for other, stats in contrasts.items():
        log.line(f"   {'relaxed_lp':<14} - {other:<16} mean {stats['mean']:+.4f}  "
                 f"95% CI [{stats['ci_low']:+.4f}, {stats['ci_high']:+.4f}]  "
                 f"wins {int(stats['wins'])}/{int(stats['pairs'])}")
        rows.append(RunRecord(study=str(config.study), geometry="contrast",
                              arm=f"relaxed_lp_minus_{other}", k=int(rows[0].k), m=int(config.n_sub),
                              draw=-1, metrics=stats, config=config.to_dict()))

    table = summarise(rows, group=("arm",), metrics=("mean", "miss_1pct", "miss_5pct"))
    log.line("\n" + format_table(table, ("arm", "n_draws", "mean_mean", "miss_1pct_mean", "miss_5pct_mean")))
    write_records(run_dir / "records.json", rows, provenance=PROVENANCE)
    log.line(f"wrote {run_dir / 'records.json'} ({len(rows)} rows)")
    log.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
