# scaling — growth in K, m, n

**Question.** How does the shipped exact solver's cost grow with the problem size, and where
does it stop fitting a 2-second budget?

## Axes held fixed
- cost family (default `quadratic`), marginal (default `gmm`), solver config (the shipped
  `RegularizedMasterSolver(epsilon=1e-4)` path, `tie_break="none"`, hysteresis pruning).

## What varies
- **square**: `m = n` over `--sizes`, for each point dimension in `--dims` (so `K = d` for
  diagonal, `K = d(d+1)/2` for quadratic);
- **rectangular**: explicit `m:n` pairs via `--rect` (default off).

## Run
```bash
python iot_experiments/scaling/run.py --family quadratic --marginal gmm \
  --sizes 200 400 800 1600 3200 --dims 2 4 8 16
python iot_experiments/scaling/analyse.py iot_experiments/scaling/out/quadratic_gmm.json
```

Outputs land in `out/` (git-ignored): `<family>_<marginal>.json` (records) and `.csv`.

## Metrics (per record, from `core.records.RunRecord`)
`iterations`, `solve_seconds`, **`mean_master_seconds` vs `mean_oracle_seconds`** (the split
matters: the oracle scan dominates at small `K`, the HiGHS master dominates at large `K`),
`max_support_over_m_plus_n`, `converged`, `certified`, and `is_optimal` when `--validate`.

`scaffold_seconds` (POT ground truth) is reported separately and is **not** part of the
algorithm's budget.

## Takeaway to date
Two regimes: scan-bound (`~m·n·K`, sub-2 s for `m·n·K ≲ 10⁷`) and master-bound (super-linear
in `K`; at `n=800` the 2 s crossing is around `K≈40–50`). Regenerate the numbers with the
commands above; see `../reports/SCALING_REPORT.md`.
