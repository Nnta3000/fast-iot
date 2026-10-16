# Scaling — growth in K, m, n

> **Status: HISTORICAL (CPU path) — do NOT cite as current.** These runs use the NumPy separation
> scan and the **HiGHS** master, and POT ground truth, i.e. the CPU reference path that the binding
> project framing (`AGENTS.md`) excludes from the target. For GPU-path scaling see
> `SERVER_TESTS_REPORT.md`.

**Status: CURRENT, empirical.** Study 1. Backing data:
`../scaling/out/quadratic_gmm.{json,csv}` and `../scaling/out/quadratic_gmm_ksweep.{json,csv}`
(git-ignored), produced by `../scaling/run.py` (driver) and summarised by
`../scaling/analyse.py`.

## Config
- cost family `quadratic`, marginal `gmm`;
- **square** `m = n`, `K = d(d+1)/2` (blockwise `iot.QuadraticFeatures`, no `(m,n,K)` tensor);
- solver: shipped path `RegularizedMasterSolver(epsilon=1e-4)`, `tie_break="none"`,
  hysteresis pruning, `tol=1e-9`, `tau_prune=1e-12`, `gamma=1`;
- single seed (0), `max_iterations=300`; `scaffold_seconds` (POT) excluded from the budget.

## Result (total solve seconds; master vs oracle per iteration)

| K | n=200 | 400 | 800 | 1600 | 3200 |
|---|---|---|---|---|---|
| 3 | 0.10 | 0.20 | 0.61 | 1.40 | 4.18 |
| 10 | 0.19 | 0.35 | 0.93 | 2.65 | 8.92 |
| 36 | 0.62 | 0.82 | 1.63 | 5.64 | 18.9 |
| 136 | 2.25 | 4.60 | 10.1 | 24.1 | 67.3 |

The master/oracle split flips with `K` (n=3200): at `K=3` the oracle dominates the iteration
(1.02 s/oracle vs 0.37 s/master = **2.7×**); at `K=136` the master dominates (4.44 s/master vs
1.67 s/oracle = **2.7× the other way**). Iteration count also grows with `K` (4 at `K=3` → 11
at `K=136`).

## Master-bound regime: large-K sweep at n=800
The square grid stops at `K=136`; the sweep below (`--rect 800:800 --dims 24 32 48`, same
config) pushes `K` until the master alone is tens of seconds per iteration.

| d | K | iters | master s/iter | oracle s/iter | master/oracle | total solve s |
|---|---|---|---|---|---|---|
| 24 | 300 | 14 | 2.53 | 0.146 | 17× | 37.5 |
| 32 | 528 | 15 | 9.13 | 0.177 | 52× | 140 |
| 48 | 1176 | 17 | **69.9** | 0.282 | **248×** | **1194** |

All converge (`converged=True`, `certified=True`). At fixed `n=800` and `|F|/(m+n)≈1`, the
master grows roughly as **`K^2.4`** (2.53 → 9.13 → 69.9 for K = 300 → 528 → 1176) while the
oracle stays nearly flat (0.146 → 0.282). This is the HiGHS master LP, not the oracle scan,
becoming the entire cost.

## The 2-second frontier
`../scaling/analyse.py` reports, for each `K`, the largest `n` with total solve < 2 s:

| K | largest n < 2 s | m·n·K at the boundary |
|---|---|---|
| 3 | 1600 | 7.7e6 |
| 10 | 800 | 6.4e6 |
| 36 | 800 | 2.3e7 |
| 136 | (none ≥ 2 s already at n=200) | — |

**Takeaway.** Two regimes: scan-bound (`~m·n·K`; sub-2 s for `m·n·K ≲ 10⁷`, e.g. `n≈2000` at
`K=3`, `n≈800` at `K=36`) and master-bound (super-linear in `K`). At `n=800` the 2 s crossing is
around `K≈40–50`; by `K=300` the master alone is 2.5 s/iter and at `K=1176` it is 70 s/iter
(1194 s total). The `m·n·K` rule of thumb only holds while the oracle dominates; for large `K`
the HiGHS master LP is the wall — exactly the regime a GPU LP solver would target.

## Caveats
Single seed; wall times are single-machine and not thread-pinned here (the frozen benchmark
`MEASUREMENT_NOTES.md` pins threads for cross-method numbers). `m·n·K` is a heuristic label,
not a fitted law.
