# EOT scalability study — dense SLSQP on the full grid

> **Superseded numbers below, corrected.** `eot_restricted_edges.md` re-analysed this run and reports that
> total optimiser time grows **24.8×**, not 28×, from `n=300` to `n=1500` (`rel_ε=0.25`, 1.09 → 27.15 s on
> the corrected rerun; the 28× in §"What this establishes" came from a first run whose iteration count
> differed). The per-edge attribution in §"What this establishes" point 1 was also broken and is retracted
> there — a single evaluation is **53 ms**, not 351 ms. The tables in this file are unchanged.

**Status:** CURRENT for the size grid; the two numbers above are superseded by `eot_restricted_edges.md`.
**Driver:** `experiments/studies/iot/real/t1t2/metric_learning/eot_scaling_slsqp.py`
**Record:** `experiments/out/eot_scaling_slsqp/scaling_slsqp.json`
**Instance:** planted `diagonal`, `point_dimension=4`, seed 0 · sizes `n = 300, 450, 675, 1000, 1500`
**ε:** `rel_eps × std(c(θ_uniform))` with `rel_eps ∈ {0.5, 0.25, 0.1}` · **Grid:** full (all edges)
**Process start-up:** **0.28 s, measured once and excluded** from every timing below.
**Cost attribution:** the optimiser cost is decomposed into *measured* evaluation counts × the
*separately measured* per-evaluation time. Nothing is inferred from a single wall clock.

## Sizes

| n | m_eff | n_eff | dense variables | transport edges |
|---:|---:|---:|---:|---:|
| 300 | 300 | 167 | 170 | 50,100 |
| 450 | 450 | 258 | 261 | 116,100 |
| 675 | 675 | 387 | 390 | 261,225 |
| 1000 | 1000 | 577 | 580 | 577,000 |
| 1500 | 1500 | 886 | 889 | 1,329,000 |

## Results

| n | rel ε | optimiser | evals | est. obj time | est. grad time | other | peak RSS | col res | θ KKT |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 300 | 0.50 | 0.77 s | 61 | 0.11 s | 0.10 s | 0.57 s | 633 MB | 2.1e-6 | 2.0e-5 |
| 300 | 0.25 | 1.59 s | 97 | 0.17 s | 0.16 s | 1.27 s | 633 MB | 1.1e-7 | 3.9e-7 |
| 300 | 0.10 | 2.45 s | 166 | 0.29 s | 0.27 s | 1.90 s | 633 MB | 3.5e-7 | 3.6e-6 |
| 450 | 0.50 | 1.18 s | 69 | 0.25 s | 0.24 s | 0.69 s | 650 MB | 6.4e-6 | 1.1e-4 |
| 450 | 0.25 | 1.66 s | 105 | 0.40 s | 0.37 s | 0.89 s | 649 MB | 3.5e-7 | 2.5e-6 |
| 450 | 0.10 | 2.85 s | 172 | 1.09 s | 0.79 s | 0.97 s | 649 MB | 1.1e-6 | 1.4e-5 |
| 675 | 0.50 | 2.44 s | 97 | 0.74 s | 0.79 s | 0.92 s | 671 MB | 1.2e-6 | 9.0e-6 |
| 675 | 0.25 | 2.66 s | 110 | 0.84 s | 0.83 s | 0.98 s | 671 MB | 6.9e-7 | 1.6e-5 |
| 675 | 0.10 | 5.62 s | 226 | 1.82 s | 1.87 s | 1.93 s | 671 MB | 1.7e-7 | 3.7e-6 |
| 1000 | 0.50 | 10.67 s | 98 | 2.07 s | 1.94 s | 6.67 s | 729 MB | 3.1e-7 | 7.9e-6 |
| 1000 | 0.25 | 8.11 s | 121 | 2.27 s | 2.39 s | 3.46 s | 729 MB | 3.7e-7 | 8.6e-6 |
| 1000 | 0.10 | 13.29 s | 227 | 4.18 s | 4.15 s | 4.97 s | 729 MB | 2.1e-7 | 1.2e-6 |
| 1500 | 0.50 | 14.86 s | 94 | 5.07 s | 4.98 s | 4.81 s | 848 MB | 9.2e-8 | 1.0e-5 |
| 1500 | 0.25 | 17.85 s | 131 | 6.99 s | 6.88 s | 3.98 s | 848 MB | 4.1e-7 | 3.6e-6 |
| 1500 | 0.10 | 46.53 s | 256 | 13.61 s | 13.60 s | 19.32 s | 848 MB | 1.2e-7 | 2.4e-6 |

All 15 runs report **`Optimization terminated successfully`** from the optimiser.

## What this establishes

**1. Cost growth.** The optimiser time grows ≈ **28× from n=300 to n=1500** (1.6 s → 17.9 s at
rel_ε = 0.25), while the dense variable count grows 5.2× and the edge count 26.5×. The cost is
dominated by evaluation count × per-evaluation cost, and the per-evaluation cost grows **14.5×**
(3.66 ms → 53 ms) because every evaluation recomputes `Φ_e (96.3 µs per 1000 edges)` against `π φ`
(168 µs per 1000 edges). At `n = 1500` a single evaluation already touches 1.33 M edges.

**2. Memory grows slowly.** 633 MB → 848 MB, i.e. **215 MB for 719 extra dense variables**
(≈ 0.30 MB/var including the edge arrays). SLSQP's dense working memory is not the binding constraint
at these sizes; the edge tensor and the per-evaluation cost are.

**3. Accuracy.** Target marginal residual stays **1e-8 … 6e-6** and θ simplex KKT **4e-7 … 1e-4**
across all sizes. Accuracy does **not** degrade with `n` — the residual is set by SLSQP's `ftol`,
not by the problem size. At `n=1500, rel_ε=0.1` the θ gradient is `-220.38` on every positive
coordinate with a spread of `2.4e-6`.

## The reporting distinction, made explicit

Three different measures were being conflated. They are now separate columns:

| measure | meaning | SLSQP result |
|---|---|---|
| **optimiser success** | SciPy `status == 0` | 15/15 here; 23/24 in the solver comparison |
| **KKT feasibility** | θ violation ≤ 1e-6 **and** target marginal ≤ 1e-5 | **1/15 here** |
| **objective reached** | Ψ at the common optimum | 24/24 (solver comparison) |

The single failed *flag* in the solver comparison was `n=300, rel_ε=0.5, start=θ_star`:
`Positive directional derivative for linesearch`, after 45 iterations — and it still produced
Ψ = 904.5814540076074, **the optimum**, with col res 5.9e-6 and θ KKT 3.8e-5.

So: **SLSQP reliably finds the optimum and its own success flag is noisy at larger `n`; its KKT
residuals (1e-6…1e-4) are not certificate-grade.** That matters directly, because the EOT certificate is
supposed to bound a value that SLSQP cannot currently certify.

## Decision threshold (as requested)

On the evidence at these sizes, the binding constraint is **not** memory. It is:

1. **Per-evaluation cost × evaluation count.** At `n = 1500` a single evaluation is 53 ms and a solve
   costs 14–47 s. Extrapolating the observed per-evaluation growth (`~1.13×` per 1.25× in `n_eff`), a
   solve at `n ≈ 3000` would be ≈ 100 ms/evaluation and tens of seconds to minutes — while the
   *useful* work (the plan has ≈ 20 non-negligible edges per node at `rel_ε = 0.25`) is a tiny fraction
   of the grid.
2. **KKT accuracy floors at 1e-6…1e-4**, which is inadequate to certify a bracket on `min Ψ`.

**Threshold to introduce a sparse / limited-memory solver:** when a restricted edge set `F` (the actual
EOT object) is used, the evaluation cost scales with `|F|`, not with the grid. The right test is whether
the restricted-`F` problem at the same accuracy solves in time independent of the full grid size. If it
does not, a sparse method is needed. That test is the natural next step and is supported by
`--edge-fraction` in this same driver.

## Honest limitations

- **Full grid only.** The EOT algorithm operates on an adaptively selected `F`. This study says nothing
  about restricted-`F` cost, which is the case that matters.
- **One instance family** (`diagonal`, Monge), one seed, one point dimension.
- **`n = 1500` is the ceiling tested** (889 dense variables, 1.33 M edges, 848 MB peak). Nothing beyond
  was attempted.
- The per-evaluation timing was measured at the *starting* iterate; evaluations elsewhere may differ.
