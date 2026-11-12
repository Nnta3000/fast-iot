"""
EN: **What does the 30 seconds buy? Solver-precision and round-count knobs, measured end to end.**

    The relaxed fit at n = 2000 runs ~12 rounds of (sparse LP over 2n duals + full-grid scan + repair) and
    lands 100x above its own `eps_r` target while reporting `converged=False`. So the natural question is
    whether the precision it pays for is used at all. Four modes, same training pair, same held-out pairs:

      | mode | outer rounds | CuPDLPx |
      | --- | --- | --- |
      | `normal` | 30 (the shipped cap) | defaults |
      | `few_rounds` | 5 | defaults |
      | `loose` | 30 | `iter_limit=200, eps_feas=1e-3, eps_opt=1e-3` |
      | `dirty` | 5 | `iter_limit=200, eps_feas=1e-3, eps_opt=1e-3` |

    plus `numpy_scan` (the scan default before 2026-11-12) to price the scan backend change on its own.

    Reported per mode: wall-clock, rounds, the loop's own regret, the **exact full-grid** regret and the
    number of training vertices the decoded map gets wrong, and `miss@5 %` on the 8 held-out pairs -- so a
    speedup only counts if it does not cost accuracy.

    CN: **那 30 秒买到了什么**：四种模式（轮数上限 × 求解器精度）加一个 numpy 扫描对照，端到端比较时间、
        外层轮数、循环自报 regret、**精确全网格** regret、训练集错配顶点数，以及 8 个留出对的 miss@5%。

    Run:
        .venv/bin/python -u -m iot_experiments.studies.real.faust.modes --n-sub 2000
"""

from __future__ import annotations

import dataclasses
import sys
import time
from pathlib import Path

import numpy as np
from jaxtyping import Float

_REPO_ROOT = Path(__file__).resolve().parents[4]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from iot.core.features import SquaredDifferenceFeatures  # noqa: E402
from iot_experiments.harness.config import RunConfig, build_parser  # noqa: E402
from iot_experiments.harness.faust import pair_geometry, read_ply, shared_index  # noqa: E402
from iot_experiments.harness.faust_match import (  # noqa: E402
    bootstrap_contrast,
    duality_gap,
    map_metrics,
)
from iot_experiments.harness.preflight import check as preflight_check  # noqa: E402
from iot_experiments.harness.records import RunLog, RunRecord, summarise, write_records  # noqa: E402
from iot_experiments.studies.real.faust.correspondence import (  # noqa: E402
    default_pairs,
    fit_theta,
)

DEFAULT_ROOT = "/home/agent/Documents/fast-iot/data/faust_extract/MPI-FAUST/training/registrations"

LOOSE: dict[str, object] = {"iter_limit": 200, "eps_feas": 1e-3, "eps_opt": 1e-3}

MODES: dict[str, dict[str, object]] = {
    "normal": {"rounds": 30, "params": None, "scan": None},
    "normal_5rounds": {"rounds": 5, "params": None, "scan": None},
    "loose": {"rounds": 30, "params": LOOSE, "scan": None},
    "dirty": {"rounds": 5, "params": LOOSE, "scan": None},
    "numpy_scan": {"rounds": 30, "params": None, "scan": "numpy"},
}

PROVENANCE: dict[str, object] = {
    "question": "does the fit's precision (CuPDLPx iteration cap/tolerances) and outer round cap change the "
                "decoded correspondence, and what wall-clock does each setting save?",
    "modes": "normal (30 rounds, default tolerances), normal_5rounds, loose (iter_limit=200, eps_feas=1e-3, "
             "eps_opt=1e-3), dirty (loose + 5 rounds), numpy_scan (30 rounds, numpy scan)",
    "training_pair": "(0,13), n=2000, subject-disjoint from the test subjects 5-9",
    "regrets": "the loop's own regret AND the exact full-grid POT gap, because they differ by ~100x and the "
               "loop's value is the one the stopping rule sees",
    "metric": "geodesic miss@5% on the 8 held-out pairs; paired bootstrap vs `normal`",
    "scan_default": "since 2026-11-12 the separation scan defaults to the torch backend (bit-identical theta "
                    "to numpy on this instance, measured); `numpy_scan` prices that change",
}


@dataclasses.dataclass(frozen=True)
class ModesConfig(RunConfig):
    """CN: 旋钮。EN: knobs."""

    study: str = "faust_modes"
    section: str = "faust_modes"
    root: str = DEFAULT_ROOT
    train_pair: tuple[int, ...] = (0, 13)
    test_subjects: tuple[int, ...] = (5, 6, 7, 8, 9)
    test_pair_count: int = 8
    n_sub: int = 2000
    backend: str = "cupdlpx"
    modes: tuple[str, ...] = tuple(MODES)

    @classmethod
    def add_study_arguments(cls, parser) -> None:
        """CN: 只加本研究特有的参数。EN: only this study's flags."""
        parser.add_argument("--root", default=DEFAULT_ROOT)
        parser.add_argument("--train-pair", type=int, nargs=2, default=[0, 13])
        parser.add_argument("--test-subjects", type=int, nargs="+", default=[5, 6, 7, 8, 9])
        parser.add_argument("--test-pair-count", type=int, default=8)
        parser.add_argument("--n-sub", type=int, default=2000)
        parser.add_argument("--backend", choices=("cupdlpx", "highs"), default="cupdlpx")
        parser.add_argument("--modes", nargs="+", default=list(MODES))


def main(argv: list[str] | None = None) -> int:
    """CN: 入口。EN: entry point."""
    parser = build_parser(ModesConfig)
    args = parser.parse_args(list(argv) if argv is not None else None)
    config = ModesConfig.from_namespace(args, parser)
    run_dir = config.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    log = RunLog(run_dir / "run.log")
    pre = preflight_check(run_tiny=False, need_gpu=False)
    pre.write(run_dir / "preflight.json")
    log.line(f"preflight ok={pre.ok} mem={pre.mem_available_gb:.1f} GB backend={config.backend}")

    root = Path(config.root)
    train_pair = (int(config.train_pair[0]), int(config.train_pair[1]))
    test_pairs = default_pairs(tuple(int(v) for v in config.test_subjects), int(config.test_pair_count))
    ma = read_ply(root / f"tr_reg_{train_pair[0]:03d}.ply")
    mb = read_ply(root / f"tr_reg_{train_pair[1]:03d}.ply")
    index = shared_index(ma, int(config.n_sub), 0)
    a, b = pair_geometry(ma, mb, index)[:2]
    k = int(a.shape[1])
    log.line(f"train pair {train_pair}, n={index.size}, K={k}")

    thetas: dict[str, Float[np.ndarray, "k"]] = {}
    info: dict[str, dict[str, float]] = {}
    for name in config.modes:
        spec = MODES[str(name)]
        started = time.perf_counter()
        fit = fit_theta(SquaredDifferenceFeatures(a, b), rel=None, max_iterations=int(spec["rounds"]),
                        backend=str(config.backend), scan_backend=spec["scan"],
                        solver_params=spec["params"])  # type: ignore[arg-type]
        elapsed = time.perf_counter() - started
        if fit["theta"] is None:
            log.line(f"   {name}: FAILED")
            continue
        theta = np.asarray(fit["theta"], dtype=np.float64)
        thetas[name] = theta
        exact, wrong = duality_gap(theta, a, b)
        info[name] = {"seconds": float(elapsed), "rounds": float(fit["iterations"]),
                      "loop_regret": float(fit["regret"]), "exact_regret": float(exact),
                      "train_wrong": float(wrong), "eps_r": float(fit["eps_r"]),
                      "solver_floor": float(fit["solver_floor"])}
        log.line(f"   {name:<15} {elapsed:6.1f}s  rounds={int(fit['iterations']):2d}  "
                 f"loop_regret={fit['regret']:.3g}  exact={exact:.3g}  train_wrong={wrong}/{index.size}  "
                 f"stall={fit['stall_status']}")

    # CN: θ 之间的差异（相对 normal），说明"快"是否换了另一个点。
    if "normal" in thetas:
        for name, theta in thetas.items():
            if name != "normal":
                log.line(f"   L1(theta_{name}, theta_normal) = {float(np.abs(theta - thetas['normal']).sum()):.3f}")

    log.line("\n--- held-out evaluation ---")
    per_pair: dict[str, dict[str, float]] = {}
    for pair in test_pairs:
        ta = read_ply(root / f"tr_reg_{pair[0]:03d}.ply")
        tb = read_ply(root / f"tr_reg_{pair[1]:03d}.ply")
        geometry = f"pair{pair[0]:03d}-{pair[1]:03d}"
        t_index = shared_index(ta, int(config.n_sub), 0)
        da, db, geo, norm, mirror = pair_geometry(ta, tb, t_index)
        for name, theta in thetas.items():
            per_pair.setdefault(geometry, {})[name] = float(
                map_metrics(da, db, theta, geo, norm, mirror)["miss_5pct"])
        del geo
        log.line(f"   {geometry}: " + "  ".join(f"{n}={per_pair[geometry][n]:.3f}" for n in thetas))

    rows: list[RunRecord] = []
    for name in thetas:
        held = float(np.mean([values[name] for values in per_pair.values()]))
        metrics = dict(info[name])
        metrics["miss_5pct"] = held
        log.line(f"   {name:<15} held-out {held:.4f}  ({info[name]['seconds']:.1f}s, "
                 f"{int(info[name]['rounds'])} rounds)")
        rows.append(RunRecord(study=str(config.study), geometry="summary", arm=name, k=k, m=int(index.size),
                              draw=0, metrics=metrics, config=config.to_dict(),
                              extra={"theta": thetas[name].tolist()}))
    contrasts = bootstrap_contrast(per_pair, base="normal", others=[n for n in thetas if n != "normal"])
    for other, stats in contrasts.items():
        log.line(f"   normal - {other:<15} {stats['mean']:+.4f}  "
                 f"CI [{stats['ci_low']:+.4f}, {stats['ci_high']:+.4f}]  "
                 f"wins {int(stats['wins'])}/{int(stats['pairs'])}")
        rows.append(RunRecord(study=str(config.study), geometry="contrast", arm=f"normal_minus_{other}", k=k,
                              m=int(index.size), draw=-1, metrics=stats, config=config.to_dict()))

    table = summarise(rows, group=("arm",), metrics=("seconds", "miss_5pct"))
    log.line("\n" + str(table))
    write_records(run_dir / "records.json", rows, provenance=PROVENANCE)
    log.line(f"wrote {run_dir / 'records.json'} ({len(rows)} rows)")
    log.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
