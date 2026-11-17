"""
EN: **Can inverse OT be a teacher instead of the deployed fitter?** Hard-negative mining from the fitted cost.

    §4f showed the relaxed LP beats a one-sided cross-entropy reweighting of the identical features by ~0.03
    miss@5 %. That is a small margin for a great deal of machinery -- an LP master, a GPU solver, a
    constraint-generation loop. This module asks whether the machinery can be spent **once** instead: fit the
    cost with inverse OT, read off which targets the fitted cost considers nearly as good as the true one,
    and use those as the hard negatives of a cheap supervised loss. The deployed model is then a plain
    cross-entropy arm; the LP is offline.

    Why the *fitted* cost is a sensible negative miner: the master certifies the true partner as optimal up
    to a margin, so the edges whose cost sits just above it are exactly the confusions the cost cannot rule
    out. Uniformly sampled negatives almost never land there.

    Four arms, all the same features and the same training pair:

      | arm | negatives in the loss |
      | --- | --- |
      | `plain` | all `n - 1` other targets |
      | `hard_lp` | 32 negatives chosen by the **inverse-OT** cost |
      | `mixed_lp` | those 32 **plus** 32 uniform random negatives |
      | `hard_self` | 32 negatives chosen by the **supervised** arm's own cost (the control) |

    `hard_self` is what makes the comparison honest: if a cost can mine its own hard negatives as well as the
    LP can, inverse OT adds nothing here.

    CN: **逆 OT 能否只当一次"老师"**：用拟合出的代价挑出"几乎和真值一样好"的目标作为硬负样本，交给廉价的
        交叉熵臂去训练。`hard_self` 是诚实的对照：若监督臂能自己挑出同样好的负样本，逆 OT 在此没有增量价值。

    Run:
        .venv/bin/python -u -m iot_experiments.studies.real.faust.curriculum --n-sub 2000
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
from iot_experiments.harness.config import RunConfig, build_parser  # noqa: E402
from iot_experiments.harness.faust import default_registrations_root, geodesic_matrix, read_ply, surface_area  # noqa: E402
from iot_experiments.harness.preflight import check as preflight_check  # noqa: E402
from iot_experiments.harness.records import (  # noqa: E402
    RunLog,
    RunRecord,
    format_table,
    summarise,
    write_records,
)
from iot_experiments.studies.real.faust.correspondence import (  # noqa: E402
    _sample_index,
    default_pairs,
    descriptors,
    fit_theta,
    geodesic_errors,
    match,
    mirror_indices,
    standardise,
)
from iot_experiments.harness.faust_match import (  # noqa: E402
    bootstrap_contrast,
    cost_matrix,
    cost_matrix_torch,
    fit_softmax,
)

DEFAULT_ROOT = str(default_registrations_root())

PROVENANCE: dict[str, object] = {
    "question": "can the inverse-OT fit be spent once, as a hard-negative miner, and give a cheap supervised "
                "arm the LP's advantage?",
    "hypothesis_class": "c_ij = sum_k theta_k (a_ik - b_jk)^2, theta on the simplex, the same standardised 13 "
                        "descriptors in every arm",
    "arms": "plain (all negatives), hard_lp (32 negatives by the inverse-OT cost), mixed_lp (32 LP negatives + "
            "32 uniform), hard_self (32 negatives by the supervised arm's own cost -- the control)",
    "selection": "tau per arm on validation pairs from the TRAINING subjects",
    "solver": "negative miner: the relaxed LP (CuPDLPx) or a first supervised fit; deployed arm: torch autograd "
              "on a masked cross-entropy, Adam, 600 steps",
    "metric": "geodesic error on the target, normalised by sqrt(surface area) -- identical to the main study",
    "memory": "the masked loss keeps one n x n cost and one n x n bool mask; ~192 MB at n=2000 in float64",
}


@dataclasses.dataclass(frozen=True)
class CurriculumConfig(RunConfig):
    """CN: 旋钮。EN: knobs."""

    study: str = "faust_curriculum"
    section: str = "faust_curriculum"
    root: str = DEFAULT_ROOT
    train_pair: tuple[int, ...] = (0, 13)
    train_subjects: tuple[int, ...] = (0, 1, 2, 3, 4)
    test_subjects: tuple[int, ...] = (5, 6, 7, 8, 9)
    test_pair_count: int = 8
    val_pair_count: int = 2
    n_sub: int = 2000
    hard_negatives: int = 32
    random_negatives: int = 32
    steps: int = 600
    lr: float = 0.05
    tau_grid: tuple[float, ...] = (1e-3, 3e-3, 1e-2)
    max_iterations: int = 30

    @classmethod
    def add_study_arguments(cls, parser) -> None:
        """CN: 只加本研究特有的参数。EN: only this study's flags."""
        parser.add_argument("--root", default=DEFAULT_ROOT)
        parser.add_argument("--train-pair", type=int, nargs=2, default=[0, 13])
        parser.add_argument("--train-subjects", type=int, nargs="+", default=[0, 1, 2, 3, 4])
        parser.add_argument("--test-subjects", type=int, nargs="+", default=[5, 6, 7, 8, 9])
        parser.add_argument("--test-pair-count", type=int, default=8)
        parser.add_argument("--val-pair-count", type=int, default=2)
        parser.add_argument("--n-sub", type=int, default=2000)
        parser.add_argument("--hard-negatives", type=int, default=32)
        parser.add_argument("--random-negatives", type=int, default=32)
        parser.add_argument("--steps", type=int, default=600)
        parser.add_argument("--lr", type=float, default=0.05)
        parser.add_argument("--tau-grid", type=float, nargs="+", default=[1e-3, 3e-3, 1e-2])
        parser.add_argument("--max-iterations", type=int, default=30)


def _hard_negative_mask(cost: Float[np.ndarray, "n n"], *, count: int, seed: int,
                        random_count: int = 0) -> torch.Tensor:
    """
    EN: boolean `n x n` mask marking the allowed classes: the diagonal, the `count` cheapest non-diagonal
        targets of each row, and (optionally) `random_count` uniform extras.
    CN: `n×n` 布尔掩码，标出允许的类别：对角、每行最便宜的 `count` 个非对角目标，以及（可选）`random_count`
        个均匀随机补位。
    """
    n = int(cost.shape[0])
    masked = np.array(cost, dtype=np.float64, copy=True)
    np.fill_diagonal(masked, np.inf)
    hard = np.argpartition(masked, int(count), axis=1)[:, : int(count)]
    allow = np.zeros((n, n), dtype=bool)
    rows = np.arange(n)
    allow[rows, rows] = True
    allow[rows[:, None], hard] = True
    if random_count > 0:
        rng = np.random.default_rng(int(seed))
        extra = rng.integers(0, n, size=(n, int(random_count)))
        allow[rows[:, None], extra] = True
    return torch.from_numpy(allow)


def fit_masked(a: Float[np.ndarray, "n k"], b: Float[np.ndarray, "n k"], *, tau: float, steps: int,
               lr: float, allow: torch.Tensor | None, device: str) -> Float[np.ndarray, "k"]:
    """
    EN: fit `theta = softmax(u)` with the cross-entropy restricted to `allow` (None = all classes). The masked
        entries are `-inf`, so the loss is a softmax over the allowed set only.
    CN: 在 `allow` 限定的类别上做交叉熵拟合 `theta = softmax(u)`（None = 全类别）；被屏蔽的位置取 `-inf`，
        因此损失只在允许集合上做 softmax。
    """
    dtype = torch.float32 if device == "cuda" else torch.float64
    xa = torch.tensor(np.asarray(a, dtype=np.float64), dtype=dtype, device=device)
    xb = torch.tensor(np.asarray(b, dtype=np.float64), dtype=dtype, device=device)
    n, k = int(xa.shape[0]), int(xa.shape[1])
    mask = None if allow is None else allow.to(device)
    u = torch.zeros(k, dtype=dtype, device=device, requires_grad=True)
    optimiser = torch.optim.Adam([u], lr=float(lr))
    target = torch.arange(n, device=device)
    for _ in range(int(steps)):
        theta = torch.softmax(u, dim=0)
        logits = -cost_matrix_torch(xa, xb, theta) / float(tau)
        if mask is not None:
            logits = logits.masked_fill(~mask, float("-inf"))
        loss = torch.nn.functional.cross_entropy(logits, target)
        optimiser.zero_grad(set_to_none=True)
        loss.backward()
        optimiser.step()
    with torch.no_grad():
        theta = torch.softmax(u, dim=0).detach().to("cpu", torch.float64).numpy()
    return np.asarray(theta, dtype=np.float64)


def main(argv: list[str] | None = None) -> int:
    """CN: 入口。EN: entry point."""
    parser = build_parser(CurriculumConfig)
    args = parser.parse_args(list(argv) if argv is not None else None)
    config = CurriculumConfig.from_namespace(args, parser)
    run_dir = config.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    log = RunLog(run_dir / "run.log")
    pre = preflight_check(run_tiny=False, need_gpu=False)
    pre.write(run_dir / "preflight.json")
    free_vram = torch.cuda.mem_get_info()[0] / 2**30 if torch.cuda.is_available() else 0.0
    device = "cuda" if free_vram > 1.5 else "cpu"
    log.line(f"preflight ok={pre.ok} mem={pre.mem_available_gb:.1f} GB vram_free={free_vram:.1f} GB "
             f"-> fits on {device}")

    root = Path(config.root)
    train_pair = (int(config.train_pair[0]), int(config.train_pair[1]))
    test_pairs = default_pairs(tuple(int(v) for v in config.test_subjects), int(config.test_pair_count))
    pool = default_pairs(tuple(int(v) for v in config.train_subjects), 4 * int(config.val_pair_count) + 16)
    val_pairs = [p for p in pool if p != train_pair][: int(config.val_pair_count)]
    a_tr = read_ply(root / f"tr_reg_{train_pair[0]:03d}.ply")
    b_tr = read_ply(root / f"tr_reg_{train_pair[1]:03d}.ply")
    idx = _sample_index(a_tr, config, int(config.seed))
    a, b = standardise(descriptors(a_tr)[idx], descriptors(b_tr)[idx])
    k = int(a.shape[1])
    log.line(f"train pair {train_pair}, n={idx.size}, K={k}; validation {val_pairs}; test {test_pairs}")

    # CN: **两个负样本挖掘器**：逆 OT 的代价，和监督臂自己的代价（对照）。
    # EN: **two negative miners**: the inverse-OT cost, and the supervised arm's own cost (the control).
    started = time.perf_counter()
    lp_fit = fit_theta(SquaredDifferenceFeatures(a, b), rel=None, max_iterations=int(config.max_iterations))
    theta_lp = None if lp_fit["theta"] is None else np.asarray(lp_fit["theta"], dtype=np.float64)
    log.line(f"negative miner 1: relaxed_lp regret={lp_fit['regret']:.4g} "
             f"({time.perf_counter() - started:.1f}s)")
    started = time.perf_counter()
    theta_sm = fit_softmax(a, b, tau=float(config.tau_grid[len(config.tau_grid) // 2]), two_sided=False,
                           steps=int(config.steps), lr=float(config.lr), sinkhorn_iterations=1, device=device)
    log.line(f"negative miner 2: row_softmax ({time.perf_counter() - started:.1f}s, {device})")
    for name, theta in (("relaxed_lp", theta_lp), ("row_softmax", theta_sm)):
        if theta is not None:
            log.line(f"   {name:<12} theta={np.round(theta, 4).tolist()}")

    # CN: 掩码只允许 {对角, 硬负样本(, 随机负样本)}；`plain` 用 None（全类别）。
    # EN: the mask allows {diagonal, hard negatives (, random negatives)}; `plain` uses None (all classes).
    masks: dict[str, torch.Tensor | None] = {"plain": None}
    if theta_lp is not None:
        masks["hard_lp"] = _hard_negative_mask(cost_matrix(a, b, theta_lp), count=int(config.hard_negatives),
                                               seed=int(config.seed))
        masks["mixed_lp"] = _hard_negative_mask(cost_matrix(a, b, theta_lp), count=int(config.hard_negatives),
                                                seed=int(config.seed),
                                                random_count=int(config.random_negatives))
    masks["hard_self"] = _hard_negative_mask(cost_matrix(a, b, theta_sm), count=int(config.hard_negatives),
                                             seed=int(config.seed))

    val_cache: list[tuple[Float[np.ndarray, "n k"], Float[np.ndarray, "n k"],
                          Float[np.ndarray, "n n"], float, Int[np.ndarray, "n"]]] = []
    for pair in val_pairs:
        va = read_ply(root / f"tr_reg_{pair[0]:03d}.ply")
        vb = read_ply(root / f"tr_reg_{pair[1]:03d}.ply")
        v_idx = _sample_index(va, config, int(config.seed))
        va_desc, vb_desc = standardise(descriptors(va)[v_idx], descriptors(vb)[v_idx])
        geo = geodesic_matrix(vb.vertices, vb.faces, v_idx)
        norm = float(np.sqrt(surface_area(vb.vertices, vb.faces)))
        mirror, _res = mirror_indices(vb.vertices, v_idx)
        val_cache.append((va_desc, vb_desc, geo, norm, mirror))

    log.line("\n--- tau selection per arm (training-subject validation pairs) ---")
    selected: dict[str, float] = {}
    for arm, mask in masks.items():
        scored: list[tuple[float, float]] = []
        for tau in config.tau_grid:
            theta = fit_masked(a, b, tau=float(tau), steps=int(config.steps), lr=float(config.lr),
                               allow=mask, device=device)
            scores = []
            for va, vb, geo, norm, mirror in val_cache:
                pred = match(cost_matrix(va, vb, theta))
                scores.append(geodesic_errors(pred, geo, norm, mirror)["miss_5pct"])
            scored.append((float(tau), float(np.mean(scores))))
            log.line(f"   {arm:<10} tau={tau:<8g} val miss@5%={np.mean(scores):.4f}")
        selected[arm] = min(scored, key=lambda item: item[1])[0]
    log.line(f"selected: {selected}")

    log.line("\n--- refit at the selected tau and evaluate on held-out pairs ---")
    thetas: dict[str, Float[np.ndarray, "k"]] = {
        arm: fit_masked(a, b, tau=selected[arm], steps=int(config.steps), lr=float(config.lr), allow=mask,
                        device=device)
        for arm, mask in masks.items()}
    thetas["uniform"] = np.full(k, 1.0 / k)
    for arm, theta in thetas.items():
        log.line(f"   {arm:<10} theta={np.round(theta, 4).tolist()}")

    rows: list[RunRecord] = []
    for pair in test_pairs:
        ta = read_ply(root / f"tr_reg_{pair[0]:03d}.ply")
        tb = read_ply(root / f"tr_reg_{pair[1]:03d}.ply")
        geometry = f"pair{pair[0]:03d}-{pair[1]:03d}"
        t_idx = _sample_index(ta, config, int(config.seed))
        a_test, b_test = standardise(descriptors(ta)[t_idx], descriptors(tb)[t_idx])
        geo = geodesic_matrix(tb.vertices, tb.faces, t_idx)
        norm = float(np.sqrt(surface_area(tb.vertices, tb.faces)))
        mirror, _res = mirror_indices(tb.vertices, t_idx)
        for arm, theta in thetas.items():
            pred = match(cost_matrix(a_test, b_test, theta))
            metrics = geodesic_errors(pred, geo, norm, mirror)
            rows.append(RunRecord(study=str(config.study), geometry=geometry, arm=arm, k=k, m=int(t_idx.size),
                                  draw=int(config.seed), metrics=metrics, config=config.to_dict()))
        log.line(f"   {geometry}: " + "  ".join(
            f"{arm}={rows[-len(thetas) + i].metrics['miss_5pct']:.3f}" for i, arm in enumerate(thetas)))
        del geo

    per_pair: dict[str, dict[str, float]] = {}
    for row in rows:
        per_pair.setdefault(row.geometry, {})[row.arm] = row.metrics["miss_5pct"]
    contrasts = bootstrap_contrast(per_pair, base="hard_lp",
                                   others=[o for o in ("plain", "hard_self", "uniform")
                                           if o in {arm for v in per_pair.values() for arm in v}])
    for other, stats in contrasts.items():
        log.line(f"   {'hard_lp':<10} - {other:<10} mean {stats['mean']:+.4f}  "
                 f"95% CI [{stats['ci_low']:+.4f}, {stats['ci_high']:+.4f}]  "
                 f"wins {int(stats['wins'])}/{int(stats['pairs'])}")
        rows.append(RunRecord(study=str(config.study), geometry="contrast", arm=f"hard_lp_minus_{other}",
                              k=k, m=int(config.n_sub), draw=-1, metrics=stats, config=config.to_dict()))

    table = summarise(rows, group=("arm",), metrics=("mean", "miss_5pct"))
    log.line("\n" + format_table(table, ("arm", "n_draws", "mean_mean", "miss_5pct_mean")))
    write_records(run_dir / "records.json", rows, provenance=PROVENANCE)
    log.line(f"wrote {run_dir / 'records.json'} ({len(rows)} rows)")
    log.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
