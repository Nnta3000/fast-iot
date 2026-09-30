# iot_experiments

Experiments for the `iot` subpackage (inverse optimal transport by constraint
generation). Run each script from the repository root after `pip install -e .`.

## Layout

| path | purpose |
| --- | --- |
| `e0…e3_*.py`, `inverse_ot_method_comparison.py`, `entropic_psd_moment_fit.py` | runnable experiment drivers |
| `benchmark/` | shared cross-method benchmark contract (`PROTOCOL.md`, instance + noisy generators, checkpoint schema, offline evaluator, runners). **Import it; do not fork it.** |
| `scratch/` | development diagnostics: scripts at the top level, reports in `scratch/reports/`, generated outputs in `scratch/results/`, retired scripts in `scratch/archive/` |
| `theory/` | the convergence proof (not code) |


| experiment | what it shows |
| --- | --- |
| `e0_fixed_fg.py` | Fix the HELLO-OT dual pair `(f*, g*)` and report the feasible `theta` range. No unknown `(f, g)`, so this isolates identifiability and is the simplest sanity check. |
| `e1_full_master.py` | Solve the master on **all** edges with correlated (non-product) marginals, averaged over seeds. Report `delta*`, the `theta` identifiability range with `(f, g)` free, and `dist(theta_maxmargin, theta_star)` as `m = n` grows. |
| `e2_loop_vs_full.py` | Compare the constraint-generation loop with the full-edge master in `lex` mode: `delta`, `theta`, and `(f, g)` equality, plus `|F|` over iterations. |
| `e3_scaling.py` | Scaling of per-iteration time, `|F|`, and iterations up to `m = n = 800`, `K <= 20`, split into LP vs scoring time. Reports the OT-free `delta` certificate. |
| `entropic_psd_moment_fit.py` | Fit a PSD metric from a cross-moment with entropic OT + Adam + Polyak averaging + ε-annealing (the exact-OT subgradient zig-zags). Reports the true L0 gap and `is_optimal_plan`. |

## Running

```bash
python iot_experiments/e0_fixed_fg.py --m 16 --n 16 --features 4 --seed 0
python iot_experiments/e1_full_master.py --sizes 6 8 10 12 16 20 --features 4
python iot_experiments/e2_loop_vs_full.py --sizes 8 10 12 16 --features 3
python iot_experiments/e3_scaling.py --sizes 100 200 400 800 --features 8
```

Each script prints a table and, with `--out PATH`, writes a JSON report. They exit
non-zero when a check fails.

## Reading the results

- `is_optimal` comes from `iot.validation.is_optimal_plan` (exact POT OT), not from
  `theta == theta_star`. Discrete data is generically non-identifiable, so `e1` reports
  a range and a distance, never an equality.
- `is_optimal_plan` now forwards a POT iteration cap (`num_itermax`) and **raises** if POT
  does not reach optimality, instead of silently comparing against an iteration-capped cost;
  `dual_optimality_certificate` is the independent dual-feasibility check.
- `e2`'s `delta_difference` is the relaxation gap between the loop (`|F|` small) and the
  full-edge master; it is at machine precision when the loop has added all violations.
- `e3`'s per-iteration cost grows like `O(m n)` because the oracle scans all edges
  blockwise; the master itself stays near `O(m + n)` edges.
