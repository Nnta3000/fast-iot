# Multi-observation relaxed inverse OT with sparse plans and linear memory

**Status: implemented and measured (2026-11-20).** Code `src/iot/sparse/`, study
`iot_experiments/studies/synthetic/multi_observation_sparse/`, tests `tests/iot_sparse_*_test.py`.

## Contribution

The claim of this work is **scalability**: the same objective as the dense reference — shared-θ minimum regret,
`F(θ) = Σ_l [⟨c(θ), π̂_l⟩ − OT_l(c(θ))]` — the same answer up to a stated tolerance, with memory linear in `n + m`
instead of quadratic, run on sizes where the dense reference cannot. Time is **not** claimed: pricing visits every
pair, so time stays quadratic.

### What is NOT claimed

No claim about approximation power, and no claim of beating the state of the art on any task. Whether a richer cost
class or a different formulation improves on the state of the art is a separate question and out of scope. The taxi
results are not part of this claim. The planted-window run is a **sanity check** (the solver recovers plans when the
data come from its own model), not a result about the method.

## What exists

| piece | where | what it is |
| --- | --- | --- |
| dense reference | `src/iot/relaxed/multi_plan.py` | shared-θ Frank–Wolfe with dense `(n,n,K)` tensors and dense POT. **Marked as the reference; not exported by `iot.relaxed`; not moved.** Used to check the bundle. |
| dense references | `studies/synthetic/multi_observation_sparse/reference.py` | (i) the exact problem as **one joint LP** in `(θ, {u_l,v_l})` — the ground truth `F*`; (ii) dense FW over an arbitrary `FeatureMap`. |
| inner oracle | `src/iot/sparse/sparse_ot.py` | exact OT on a growing edge set; blockwise pricing of the complement via `exact/oracle.scan_violations`; certificate `max σ` outside `E`. |
| outer loop | `src/iot/sparse/bundle_multi.py` | Kelley cuts in θ (`solve_cut_master(mode="kelley")`, `K+1` variables), bracket `[master value, best exact F]`. |
| features | `src/iot/sparse/features.py` | `s, s², log1p(s), s³` computed on the fly from two point clouds, subsample-RMS normalised; **no constant and no same-zone column**. |

Reused, not reimplemented: `FeatureMap`, `EdgeSet`, `SimplexTheta`/`LinearThetaSet`, `scan_violations`,
`solve_cut_master`. The only new solver is the restricted-edge exact OT with pricing.

## Algorithm

```
repeat                                  # outer: Kelley in θ
   for l = 1..L:
      E_l ← warm set (observed support ∪ previous E_l)
      repeat                            # inner: exact OT on E_l
         (P, u, v) ← sparse LP on E_l  (scipy HiGHS, sparse A_eq, m+n rows)
         σ_max ← pricing over all pairs outside E_l, blockwise
         if σ_max ≤ tol_inner: break    # dual feasible everywhere ⇒ P is globally optimal
         E_l ← E_l ∪ top-γ violators (per row and per column)
      g += M(π̂_l) − M(P_l);  F += ⟨M(π̂_l),θ⟩ − ⟨c,P_l⟩
   cuts ← cuts ∪ {g};  lower ← master value;  upper ← min(upper, F)
   if upper − lower ≤ tol·max(1,|upper|): stop
   θ ← master θ
```

`F(θ) = ⟨g, θ⟩` holds exactly because `F` is positively homogeneous of degree one in the cost; the bundle checks this
identity every iteration and raises if it fails.

## What is stored

| object | size |
| --- | --- |
| θ, cut bank | `K`, `t×K` |
| duals, marginals | `L·(m+n)` |
| edge sets `E_l` | `Σ_l |E_l|`; **peak measured below** |
| plans | ≤ `L·(m+n−1)` non-zeros |
| pricing tile | `max_tile_elements × K` |
| features | `O((m+n)d)` for point clouds; `O(mn)` if the feature is a stored matrix (not used here) |

Measured peak `|E|` for one cold solve, `γ=2` (from `--mode inner`):

| n | peak `|E|` | peak `/(n+m)` | kept after `prune_slack=1e-3` | sweeps | LP s | pricing s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 200 | 4 613 | 11.5 | 438 (1.09×) | 10 | 0.30 | 0.05 |
| 800 | 26 138 | 16.3 | 2 123 (1.33×) | 12 | 3.34 | 0.96 |
| 3200 | 139 310 | 21.8 | 10 109 (1.58×) | 16 | 45.8 | 22.0 |

**This is the main risk, stated early:** the inner loop accumulates violators and the **peak** edge count is
`11.5–21.8 ×(n+m)`, still rising with `n` (the number of sweeps grows). Pruning at the end returns the *warm* set to
1.1–1.6×, but pruning happens after the peak, so it does not reduce the peak. Memory is therefore `O(|E|)` with a
constant that is not yet bounded by the measurements. Time stays `O(m n K)` per sweep.

## Θ and stopping

`θ` on the simplex over unit-RMS-normalised columns; **no constant column and no same-zone-only column** (both were
measured to make the regret trivially small in the dense studies). Outer stop: `upper − lower ≤ tol·max(1,|upper|)`,
with `lower` the Kelley master value and `upper` the best exact `F`; inner stop: pricing gap ≤ `tol_inner`. Every row
records θ, both gaps, the diagonal mass of observed and predicted plans, and the peak/kept edge counts.

## Correctness (measured)

`n=30, L=5`, planted θ*: bundle `F = 4.27e-08`, bracket `[-9.19e-07, 4.27e-08]`, gap `9.6e-07`, converged;
dense joint LP `F* = -2.3e-16` (machine zero); dense FW (100 iterations) `F = 1.05e-05`. Plan agreement 0.9994 (one
window differs by a tie). So the bundle matches the exact LP to `5e-8` absolute and is closer than the dense FW at the
same budget; the bracket contains `F*`.

## Scaling (measured, `L=3`, dense arm capped per size)

| n | dense FW F / time / peak | bundle F / time / peak |
| ---: | --- | --- |
| 200 | 2.33e-06 / 0.4 s / 628 MB | 1.40e-07 / 0.8 s / 629 MB |
| 800 | 6.86e-05 / 3.4 s / 724 MB | 4.08e-06 / 8.3 s / 709 MB |
| 3200 | 6.98e-05 / 13.0 s / 1801 MB | 6.98e-05 / 24.9 s / 1570 MB |

**Read this carefully.** (i) The dense arm was capped (10 / 5 / 1 iterations), so the `F` columns are not a quality
comparison; at n=3200 the dense arm did one iteration and both report the same `F`. (ii) The bundle is **slower** at
every size here: POT's dense EMD is C code, while each inner solve is a certified sparse LP plus pricing. (iii) The
memory difference is small because the measured peak includes ~600 MB of library baseline and the instance
generator's own dense cost (≈250 MB at n=3200). The separation is asymptotic, not yet dominant at n ≤ 3200.

Where the method stops: at `n=3200` a cold inner solve takes 66 s, of which the LP is 67 %. Time grows like `n²`
(4.3 s → 66 s for 4× `n`), so `n=12800` would be ≈17 min **per window per outer iteration**; the **memory limit is
never reached** — time fails first. The array that would fail first is the **peak edge set** (and HiGHS's internals
on it), not the plan and not the features.

## Inner solver decision

`scipy.optimize.linprog(method="highs")` on a sparse `A_eq`, **labelled a reference**. AGENTS.md says: "no NumPy scan
and no HiGHS master on the production path"; HiGHS is "CPU-only and retired from the production path
(reference/validation only)"; the production master is CuPDLPx. The tension is real: the inner transport LP has no
existing exact sparse alternative in the environment (POT dense, HELLO-OT squared-Euclidean, networkx integer-only).
Measured: it is 67 % of the runtime at n=3200. **Per the decision rule, that is the evidence to propose a primal-dual
/ auction inner solver — not to build it in this pass.**

## Not done

The single-plan relaxed loop at `L=1` was not run as a third arm: it solves the same objective, and the bundle at
`L=1` already matches both dense references; it would add a third solver, not a new answer. The large `L` sweep was
reduced to `L ∈ {1,5,20}` in the sanity mode. `n=12800` was not run (estimated 17 min per inner solve).
