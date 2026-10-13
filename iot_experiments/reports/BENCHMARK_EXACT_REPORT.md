# Exact (constraint-generation) benchmark — clean and noisy plans

**Method label: `exact-constraint-generation` — empirical, not proved.** Shipped configuration: hysteresis pruning ON, perturbed one-solve master `RegularizedMasterSolver(ε=1e-4)`, K=8, γ=1, tol=1e-9, τ=1e-12. Median [IQR] over seeds; per-run budget 600 s; threads pinned to 8; peak RSS = real process RSS.

Regime 1 — clean plans (`π̂ = π*`):

| n | seeds | converged | iterations | wall s | LP s/solve | max |F|/(m+n) | peak RSS MB | final maxviol | G | R |
|---|---|---|---|---|---|---|---|---|---|---|
| 200 | 10 | 10/10 | 12 [12, 13] | 0.5458 [0.522, 0.562] | 0.02036 [0.02021, 0.02076] | 3.303 [3.274, 3.318] | 712.6 [712.4, 712.8] | -0.0005796 [-0.001288, -0.0001723] | 0 [0, 0] | 0 [0, 0] |
| 800 | 10 | 10/10 | 17 [16, 17] | 5.666 [5.577, 6.043] | 0.2028 [0.1979, 0.2079] | 3.384 [3.372, 3.398] | 808 [807.9, 808.2] | -8.858e-05 [-0.0001653, -2.351e-05] | 0 [0, 0] | 0 [0, 0] |
| 1600 | 10 | 10/10 | 19 [18.25, 19] | 23.67 [22.95, 24.3] | 0.884 [0.8136, 0.9208] | 3.419 [3.405, 3.423] | 1106 [1102, 1114] | -5.079e-05 [-8.684e-05, -1.478e-05] | 0 [0, 0] | 0 [0, 0] |
| 3200 | 5 | 4/5 | 19.5 [19, 20.75] | 99.94 [97.31, 104.8] | 3.781 [3.744, 3.835] | 3.436 [3.421, 3.447] | 2327 [2302, 2374] | -1.381e-05 [-1.95e-05, -8.284e-06] | 0 [0, 0] | 0 [0, 0] |

Regime 2 — noisy plans (`M = multiplier · n` samples):

no noisy rows yet

Reference rows (uniform θ, and θ*):

| n | noise | uniform G | uniform R | theta* G | theta* R |
|---|---|---|---|---|---|
| 200 | clean | 0.2453 | 0.1355 | 0 | 0 |
| 800 | clean | 0.2495 | 0.1004 | 0 | 0 |
| 1600 | clean | 0.2512 | 0.08658 | 0 | 0 |
| 3200 | clean | 0.2651 | 0.07751 | 0 | 0 |

## Notes and confounds

- **Label:** the shipped exact config is an empirical safeguard; convergence at scale is not proved (see `LEX_GENERIC_REPORT.md` and the proof's open problems).
- **Noisy regime is near-trivial here:** `π*` is a permutation, so the sampled plan has (almost) the same support; the exact method can certify it. Zero-marginal rows/columns are dropped per PROTOCOL §3.1, so `n'` may be slightly below `n`.
- **Evaluator cap fixed:** `benchmark/evaluate.py` now passes `numItermax=DEFAULT_NUM_ITERMAX` (2e6) to every `ot.emd`. Before the fix the `n=3200` θ\* reference row was nonzero (`G≈−0.0017`, which is impossible); it is now exactly 0, so the `n=3200` `G`/`R` rows are trustworthy. (The θ-independent `uniform` row is unchanged.)
- Noisy rows are intentionally absent: the previous noisy runs used the retired generator and are quarantined in `../legacy/scratch/results/benchmark_runs/noisy_superseded/`; the canonical noisy re-run (`noisy.py`) is pending.
- `G`/`R` are computed by the shared evaluator (`score_theta`), never inside a timed run.
