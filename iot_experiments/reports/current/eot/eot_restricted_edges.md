# EOT restricted-`F` report — sparse master accuracy, cost, and full-grid agreement

**Status: CURRENT. This report corrects `eot_scaling_slsqp.md` (the 28× figure and the per-edge
attribution) and defines the current picture of the restricted-`F` question. Read it after the other
four.**
**Driver:** `iot_experiments/studies/iot/real/t1t2/metric_learning/eot_restricted_edges.py`
**Record:** `iot_experiments/out/eot_restricted_edges/restricted_f.json` · 60/60 cases completed
**Instance:** planted `diagonal`, `point_dimension=4`, seeds {0,1} · `n ∈ {70,120,300}` ·
`rel_eps ∈ {0.5,0.25}` · fractions {1,2,5,10,20 %}`
**Driver:** `iot_experiments/studies/iot/real/t1t2/metric_learning/eot_restricted_edges.py`
**Grid:** `n ∈ {70, 120, 300}` × `seeds {0,1}` × `rel_ε ∈ {0.5, 0.25}` × fractions {1, 2, 5, 10, 20 %} × 2

## Two corrections to the previous report, made first

**(a) The 28× figure was wrong.** Total optimiser time grows **24.8×** from n=300 to n=1500 at
`rel_ε = 0.25` (1.09 s → 27.15 s), not 28×; and overall it is **11–25×** depending on `rel_ε`
(1.59 → 17.85 s in the first run, 1.09 → 27.15 in the corrected rerun — SLSQP's iteration count varies
with the starting iterate). The report now states the measured decomposition:

| quantity | n=300 → n=1500 (rel_ε=0.25) |
|---|---|
| edges | 50,100 → 1,329,000 (**26.5×**) |
| `evaluate()` cost | 1.70 → 52.45 ms (**30.9×**) |
| measured evaluation count | 97 → 131 (**1.35×**) |
| evaluation work (count × cost) | 0.16 → 6.87 s (**41.7×**) |
| **total optimiser time** | 1.09 → 27.15 s (**24.8×**) |

**(b) The per-edge attribution was broken.** The driver timed `evaluate(...)[0]` and `evaluate(...)[1]`
separately and added them — but `evaluate()` returns value **and** gradient from one pass, so this
double-counted (~2.7×). That is why "96 µs + 168 µs per 1000 edges" implied 351 ms for a 53 ms
evaluation. Fixed: the driver now times the single `evaluate` call the optimiser makes, plus an
independent edge sweep (`np.add.reduce` over the edge features). Re-measured, the edge sweep is
**14.2–16.3 µs per 1000 edges**, remarkably stable (±4 %) from 50 k to 1.33 M edges, and the dominant
share of an evaluation is the row log-sum-exp (57 %) and the plan construction (23 %).

## A code defect found before benchmarking (your item 4)

`joint_reference.evaluate` computed the row log-sum-exp by reshaping to `(n_source, n_target)` — valid
only on a rectangular grid. On a restricted `F` with ragged rows it **silently mixed rows**.

Fixed with an independent `segmented_logsumexp` (`np.maximum.at` / `np.add.at`, max-shift stabilised)
that supports ragged groups and **raises on any empty group**. Verified: matches an explicit per-row
loop to `4.4e-16` on an intentionally ragged set, and the new regression test
`test_reference_evaluate_handles_a_ragged_edge_subset` **fails the old implementation** and passes the
new one. The whole sparse benchmark would otherwise have been run against a wrong reference.

## Result 1 — the restricted solve itself is cheap and accurate

| | value |
|---|---|
| restricted target marginal residual | ≤ `3.4e-6` (most ≤ `1e-6`) |
| restricted θ KKT violation | ≤ `8.6e-6` |
| SciPy status | `0` in **60/60** |
| time ratio F/U | 0.04 – 1.9 (median ≈ 0.15) |

At 5 % of the grid the restricted solve takes **0.03–0.05 s** against 0.3–0.4 s for the full grid, and
it solves to the same accuracy. **Sparse masters are cheaper and accurately solvable.** That part of the
milestone is met.

## Result 2 — the restricted solution does *not* approximate the full grid at these fractions

`Ψ_F ≤ Ψ_U` pointwise, and the restricted optimum is a different point. Evaluating the restricted
solution on the **full grid**:

| n | rel ε | \|F\|/\|U\| | Ψ gap vs full-grid optimum | θ L1 vs full | **plan mass kept in F** |
|---:|---:|---:|---:|---:|---:|
| 70 | 0.25 | 0.20 | 1.54 | 0.11 | 0.709 |
| 120 | 0.25 | 0.20 | 1.37 | 0.09 | 0.769 |
| 300 | 0.25 | 0.20 | **1.92** | **0.04** | **0.801** |
| 300 | 0.25 | 0.10 | 105.2 | 0.14 | 0.292 |
| 300 | 0.25 | 0.05 | 361.6 | 0.27 | 0.069 |
| 300 | 0.50 | 0.20 | 220.9 | 0.07 | 0.223 |

At **5–10 %** the restricted solution misses **70–99 % of the full-grid plan mass**: it is a solution to
a different, much easier problem. At **20 %** it keeps 71–80 % of the mass and the objective gap collapses
to `O(1)` in **absolute** terms while Ψ itself is `O(10²–10³)` — which is `O(0.1–1 %)` relative.

## The decisive caveat: 1–20 % is an order of magnitude too small for these instances

Measured from the full-grid solutions themselves, the number of edges carrying 99 % of the plan mass:

| n | rel ε | grid | edges for 99 % mass | % of grid | per node |
|---:|---:|---:|---:|---:|---:|
| 70 | 0.5 | 3,220 | 2,527 | **78.5 %** | 21.8 |
| 120 | 0.5 | 9,120 | 6,966 | **76.4 %** | 35.5 |
| 300 | 0.5 | 50,100 | 37,321 | **74.5 %** | 79.9 |
| 300 | 0.25 | 50,100 | 25,902 | **51.7 %** | 55.5 |

At `n = 300, rel_ε = 0.25` an oracle that keeps 99 % of the mass needs **~52 % of the grid**, not 5 %.
**These instances cannot be approximated by a 20 % edge set at all** — so the fractions 1–20 % are not a
hard test of EOT; they are a test of a regime in which the restricted problem is simply a different problem.

This also corrects an earlier claim of mine. I previously reported "at ε = 0.05 the plan's 99 % mass sits
on 1.9 edges per node". That was measured on the `m_eff = 120, n_eff = 70` instance at `ε = 0.05 × std`,
where `std` was computed on the FULL grid of a **different** point dimension. The numbers above are on the
current instances at the current ε normalisation, and they say the opposite: the mass is spread, not
concentrated, at `rel_ε = 0.25–0.5`.

## What this means for EOT

1. **Sparse masters are cheap and accurate.** Confirmed.
2. **Sparsity of edges does not solve node scaling.** The dense variable count stays `K + n_eff - 1`
   regardless of `|F|`: at n=300 that is 170 variables whether `F` has 500 or 50,100 edges. A sparse
   edge set with a dense optimiser improves evaluation cost, not optimisation cost. Your two scaling
   questions must stay separate, and only the first is answered here.
3. **The fractions should be revisited.** A meaningful restricted-`F` test on these instances needs
   fractions near the measured 99 %-mass frontier (≈ 50–80 %), or larger `ε` / different cost families
   where the plan really is sparse. The gap between "what a certificate would need" and "what a 5 % edge
   set can deliver" is the central risk to the EOT proposition, and it is now quantified.

## Still blocked

Adaptive edge-generation loop, full-grid objective certificate, P2/P3 claims, production solver choice.

## Honest limitations

- One instance family (`diagonal`, Monge), two seeds, one point dimension.
- `F` is selected using the **full-grid plan** (an oracle). It measures the best case for a given
  `|F|`; a real adaptive loop would have to *discover* those edges and would do no better.
- The budget rule (top mass then connectors) is one choice; a row/column-threshold rule may differ.
