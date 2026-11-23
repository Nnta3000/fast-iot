"""
EN: **the concluding run: joint and averaged arms taken to convergence on the cells that were capped.**

    Reuses `arms.run_cell` unchanged (same arms, same discriminating θ* rule, same recording) with a raised outer
    iteration limit. Tasks are independent (cell, seed) pairs, so they run in separate processes; the resulting
    records are written to one file, in the same schema as every other arms run.

    Cells (K, L, p): (10, 10, 0), (20, 5, 0), (20, 10, 0), (10, 5, 0.2), 5 seeds each, uniform marginals, no control.
    Big-plan arm: K = 10, p = 0, L ∈ {5, 2} (n′ = 200, 80), 5 seeds each, `only_big`.

CN: **收尾运行：在被截断的格子上把联合臂与平均臂跑到收敛。** 原样复用 `arms.run_cell`（同样的臂、同样的判别性 θ*
    规则、同样的记录），只提高外层迭代上限。任务为独立的 (格子, 种子) 对，故分进程运行；结果写入同一文件，schema 与其他
    arms 运行相同。格子与种子见英文；大计划臂 K=10、p=0、L∈{5,2}（n′ = 200、80）。

    python -m iot_experiments.studies.synthetic.multi_observation_sparse.run_final --workers 8
"""

from __future__ import annotations

import argparse
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from iot_experiments.harness.records import RunLog, RunRecord, write_records
from iot_experiments.studies.synthetic.multi_observation_sparse.arms import run_cell

OUT = Path("iot_experiments/out/multi_observation_sparse/arms/final")
CELLS: tuple[tuple[int, int, float], ...] = ((10, 10, 0.0), (20, 5, 0.0), (20, 10, 0.0), (10, 5, 0.2))
BIG_CELLS: tuple[tuple[int, int, float], ...] = ((10, 5, 0.0), (10, 2, 0.0))


def _task(payload: tuple[int, int, float, int, bool, int]) -> list[RunRecord]:
    """EN: one (cell, seed) task, in its own process. CN: 一个 (格子, 种子) 任务，在独立进程中。"""
    k, n_observations, noise, seed, big, iterations = payload
    log = RunLog(Path(f"/tmp/opencode/final_task_K{k}_L{n_observations}_p{noise:g}_s{seed}_{big}.log"))
    try:
        return run_cell(log, n=40, bumps=k, n_observations=n_observations, noise=noise,
                        marginal_mode="uniform", seed=seed, control="none", iterations=iterations,
                        one_big_plan=False, only_big=big)
    finally:
        log.close()


def main(argv: list[str] | None = None) -> int:
    """EN: entry point. CN: 入口。"""
    parser = argparse.ArgumentParser(description="concluding run to convergence")
    parser.add_argument("--iterations", type=int, default=300)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--tag", default="final")
    args = parser.parse_args(argv)
    out_dir = Path("iot_experiments/out/multi_observation_sparse/arms") / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    payloads = [(k, l, p, seed, False, args.iterations) for k, l, p in CELLS for seed in args.seeds]
    payloads += [(k, l, p, seed, True, args.iterations) for k, l, p in BIG_CELLS for seed in args.seeds]
    started = time.perf_counter()
    records: list[RunRecord] = []
    with ProcessPoolExecutor(max_workers=int(args.workers)) as pool:
        for chunk in pool.map(_task, payloads):
            records.extend(chunk)
    total = time.perf_counter() - started
    log = RunLog(out_dir / "run.log")
    log.line(f"{len(payloads)} tasks in {total:.1f}s with {args.workers} workers "
             f"(iterations={args.iterations})")
    write_records(out_dir / "records.json", records,
                  provenance={"question": "do the joint and averaged arms differ once the joint fit is converged?",
                              "family": "DistanceProfileFeatures (bump basis)",
                              "theta_star": "Dirichlet(0.5); fresh-instance neutral agreement <= 0.5",
                              "arms": "A_single (mean over plans), B_averaged, B_informative, C_joint, D_oracle, "
                                      "E_neutral; A_single_big for the big-plan cells",
                              "held_out": "fresh points, fresh marginals, exact-OT plan at theta*",
                              "convergence": f"outer cap {args.iterations}; stops early when gap <= 1e-6",
                              "cells": [[k, l, p] for k, l, p in CELLS],
                              "big_cells": [[k, l, p] for k, l, p in BIG_CELLS]})
    log.close()
    print(f"{len(payloads)} tasks in {total:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
