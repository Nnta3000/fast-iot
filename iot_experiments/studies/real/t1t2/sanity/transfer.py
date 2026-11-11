"""
EN: **Sanity check D -- pool-size transfer.**

    Hypothesis: OT pairing depends on competition inside the pool, so a cost trained on batches of size
    `s` need not transfer to a pool of size `s'`. This check measures it for every arm, cosine included:
    one cost per `(K, m)`, then ONE EMD solve on each pool size, so the only thing that changes between
    cells is the pool. The scoring function is fixed up front: the true partner is placed in the pool and
    the score is `1 / number of candidates` for a random pick.

      pool sizes   50, 100, 200, 400, 800, 1000   (mini-pool placed inside the 1000 subjects)
      arms         COSINE (label-free), UNIFORM (theta_ref), iot (F2's arm; relax/reduce on small `m`)
      metric       `hit` = exact-EMD recovery, `nn_hit` = free per-row argmin (no OT, for contrast)

    **Reading rule (fixed before the run).** A method whose `hit` decays faster than a random pick's
    `1/C` is degrading with pool size; a method whose advantage over COSINE shrinks as the pool grows is
    not transferring. Both are reported per `(K, m, C)`, never pooled.

    CN: **合理性检查 D —— 池大小迁移**：同一个代价在 50…1000 的池上分别解一次 EMD，看命中率随池的变化；
        余弦也在同一池上测量，因此衰减本身是可预期的，要看的是**相对余弦的优势**是否随池缩小。

    Cost: `sum_C C^2` EMD solves per arm, ~3e5 at `C=400` and ~1e6 at `C=1000`; a 1000x1000 POT solve is
    ~0.2 s on 4 threads, so the 1000-pool costs minutes per arm. Use `--pool-sizes` to trade scope.
"""

from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

import numpy as np
import ot
from jaxtyping import Float, Int

_REPO_ROOT = Path(__file__).resolve().parents[4]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from iot_experiments.harness.config import RunConfig, build_parser  # noqa: E402
from iot_experiments.harness.preflight import check as preflight_check  # noqa: E402
from iot_experiments.harness.records import (  # noqa: E402
    RunLog,
    RunRecord,
    format_table,
    summarise,
    write_records,
)
from iot_experiments.harness.registry import apply_geometry, features, fit_geometry  # noqa: E402
from iot_experiments.harness.solver import diagonal_cost  # noqa: E402
from iot_experiments.studies.real.t1t2.metric_learning.run import select_iot  # noqa: E402

FOLDS: int = 5
_SCORING_DRAWS: tuple[int, ...] = (0, 1, 2)

PROVENANCE: dict[str, object] = {
    "question": "does a cost trained on m labelled pairs transfer to a pool of a different size?",
    "protocol": "F2 features and geometry; the true partner is always present in the pool; one EMD solve "
                "per (arm, K, m, draw, pool size), columns shuffled, three shuffles averaged",
    "reading": "compare each arm against COSINE on the SAME pool size; raw hit decays with pool size by "
               "construction (random pick is 1/C), so the raw curve alone says nothing",
    "chance": "1 / C for a random pick",
    "nested_pools": "each pool is a prefix of the subject order, so C in {50,100,200} are nested in "
                    "C=400 -- a cell that flips in the smaller pool but not the larger is a small-pool "
                    "artefact, not a transfer failure",
}


def _hit_at(cost: Float[np.ndarray, "candidates candidates"], *, draw: int) -> float:
    """
    CN: 方阵 EMD 命中率：打乱列、用 `argsort(perm)` 跟踪真伙伴；先跑一次 oracle 断言。
    EN: square EMD hit: shuffle columns, track the true partner via `argsort(perm)`; an oracle is run
        first and the result is refused unless it is exactly 1.
    """
    c = int(cost.shape[0])
    oracle = np.ones((c, c), dtype=np.float64)
    np.fill_diagonal(oracle, 0.0)
    if abs(_plan_hit(oracle, draw=0) - 1.0) > 1e-12:
        raise RuntimeError("square-EMD oracle failed; hit rates would be meaningless")
    return _plan_hit(np.asarray(cost, dtype=np.float64), draw=int(draw))


def _plan_hit(cost: Float[np.ndarray, "candidates candidates"], *, draw: int) -> float:
    """CN: 实际计算（方阵）。EN: the actual computation (square assumed)."""
    c = int(cost.shape[0])
    perm: Int[np.ndarray, "candidates"] = np.random.default_rng(int(draw)).permutation(c)
    plan: Float[np.ndarray, "candidates candidates"] = ot.emd(np.ones(c) / c, np.ones(c) / c,
                                                              cost[:, perm], numItermax=10_000_000)
    return float(np.mean(np.argmax(plan, axis=1) == np.argsort(perm)))


def _nn_hit(cost: Float[np.ndarray, "candidates candidates"], *, draw: int) -> float:
    """CN: 自由逐行 argmin（不解 OT），跟踪列打乱。EN: free per-row argmin, no OT; shuffle tracked."""
    c = int(cost.shape[0])
    perm: Int[np.ndarray, "candidates"] = np.random.default_rng(int(draw)).permutation(c)
    return float(np.mean(np.argmin(cost[:, perm], axis=1) == np.argsort(perm)))


@dataclasses.dataclass(frozen=True)
class TransferConfig(RunConfig):
    """CN: 旋钮。EN: knobs."""

    study: str = "sanity_transfer"
    section: str = "sanity_transfer"
    feature: str = "dinov2-pool4"
    ks: tuple[int, ...] = (64, 128, 256)
    ms: tuple[int, ...] = (100, 300)
    pool_sizes: tuple[int, ...] = (50, 100, 200, 400, 800, 1000)

    @classmethod
    def add_study_arguments(cls, parser) -> None:
        """CN: 只加本研究特有的参数。EN: only this study's flags."""
        parser.add_argument("--feature", default="dinov2-pool4", choices=["dinov2-pool4"])
        parser.add_argument("--ks", type=int, nargs="+", default=[64, 128, 256])
        parser.add_argument("--ms", type=int, nargs="+", default=[100, 300])
        parser.add_argument("--pool-sizes", type=int, nargs="+", default=[50, 100, 200, 400, 800, 1000])


def main(argv: list[str] | None = None) -> int:
    """CN: 入口。EN: entry point."""
    parser = build_parser(TransferConfig)
    args = parser.parse_args(list(argv) if argv is not None else None)
    config = TransferConfig.from_namespace(args, parser)
    run_dir = config.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    log = RunLog(run_dir / "run.log")
    pre = preflight_check(run_tiny=False, need_gpu=False)
    pre.write(run_dir / "preflight.json")
    log.line(f"preflight ok={pre.ok} mem={pre.mem_available_gb:.1f} GB")
    log.line(f"ks={config.ks} ms={config.ms} pools={config.pool_sizes} draws={config.draws}")

    rows: list[RunRecord] = []
    for k in config.ks:
        x, y = features(str(config.feature), dim=int(k))
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        order = np.random.default_rng(10_000).permutation(int(x.shape[0]))
        largest = int(max(config.pool_sizes))
        pool = np.sort(order[:largest])                      # nested prefix pools
        train = np.sort(order[largest:])                     # never in any pool
        stats = fit_geometry(x[pool], y[pool], "G2")
        x_pool_full, y_pool_full = apply_geometry(x[pool], y[pool], stats)
        log.line(f"[K={k}] pool cap {largest}; {len(train)} subjects available for labelling")

        for m in config.ms:
            for draw in config.draws:
                select = np.random.default_rng(int(draw)).choice(len(train), size=int(m), replace=False)
                labelled = train[np.sort(select)]
                x_fit, y_fit = apply_geometry(x[labelled], y[labelled], stats)
                fit = select_iot(np.asarray(x_fit), np.asarray(y_fit), solver=str(config.solver),
                                 solve_limit_s=float(config.solve_limit_s), seed=int(draw))
                theta = None if fit.status != "ok" else np.asarray(fit.theta, dtype=np.float64)
                for c in config.pool_sizes:
                    c = int(c)
                    xs, ys = np.asarray(x_pool_full)[:c], np.asarray(y_pool_full)[:c]
                    for arm, th in (("COSINE", np.ones(int(k)) / float(k)),
                                    ("iot", theta)):
                        if th is None:
                            rows.append(RunRecord(study="sanity_transfer", geometry=f"pool{c}",
                                                  arm="iot", k=int(k), m=int(m), draw=int(draw),
                                                  metrics={}, config=config.to_dict(),
                                                  excluded=True, status=str(fit.status)))
                            continue
                        cost = diagonal_cost(xs, ys, th)
                        met = {"hit": float(np.mean([_hit_at(cost, draw=d) for d in _SCORING_DRAWS])),
                               "nn_hit": float(np.mean([_nn_hit(cost, draw=d) for d in _SCORING_DRAWS])),
                               "chance": 1.0 / c}
                        rows.append(RunRecord(study="sanity_transfer", geometry=f"pool{c}", arm=arm,
                                              k=int(k), m=int(m), draw=int(draw), metrics=met,
                                              config=config.to_dict(),
                                              extra={"pool_size": c, "selected": fit.selected or {}
                                                     if arm == "iot" else {}}))
                        log.line(f"[K={k} m={m} d={draw}] pool={c:<4} {arm:<6} hit={met['hit']:.4f} "
                                 f"nn={met['nn_hit']:.4f} chance={met['chance']:.4f}")

    table = summarise(rows, group=("arm", "k", "m", "geometry"), metrics=("hit", "nn_hit"))
    log.line("\n" + format_table(table, ("arm", "k", "m", "geometry", "n_draws", "hit_mean", "nn_hit_mean")))
    write_records(run_dir / "records.json", rows, provenance=PROVENANCE)
    log.line(f"wrote {run_dir / 'records.json'} ({len(rows)} rows)")
    log.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
