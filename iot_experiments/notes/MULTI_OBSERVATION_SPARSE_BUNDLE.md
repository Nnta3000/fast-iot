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

---

## Addendum: the K = 2n reversal is a degeneracy, not a result (2026-11-20)

The arms pilot showed `C_joint` far below `B_averaged` at K = 80 = 2n. Before treating that as a finding, the fitted
cost was checked against the observed support.

**Degeneracy (K=80, n=40, 10 seeds, both L).** For each fitted θ, every support edge of the training plans was
assigned to its nearest bump centre; the weighted bins (θ_k > 1e-6) carry **0.0000 ± 0.0000** of the observed mass.
Examples from the log: L=5 seed 0 puts θ = 1.0 on bin 35 with **zero observed mass** and reaches `F = −3.3e-11`;
L=20 seed 0 spreads over bins 38/28/37, all zero-mass, `F = −1.9e-10`. Zero-mass bins: 47.8/80 (L=5), 37.9/80
(L=20). At K=10 the same statistic is 0.478 ± 0.275 (L=5) and 0.638 ± 0.219 (L=20) with 3.7/10 and 2.3/10 zero-mass
bins, and `F ≈ 3.6e-3` — a real fit.

So at K = 2n the cost is ~0 on every observed edge and the regret is zero **by construction**. The K=2n rows of the
pilot measure this degeneracy, not the value of joint fitting.

**Fixes (K=80, L ∈ {5,20}, 10 seeds, 10 outer iterations).** Same Θ and same rule for every arm.

| fix | L | B_averaged agreement | C_joint agreement | B cost error | C cost error |
| --- | ---: | ---: | ---: | ---: | ---: |
| reference | 5 | 0.200 ± 0.122 | 0.050 ± 0.040 | 0.749 | 0.931 |
| reference | 20 | 0.310 ± 0.111 | 0.060 ± 0.034 | 0.584 | 0.924 |
| floor τ=0.1 | 5 | 0.095 ± 0.060 | 0.025 ± 0.025 | 0.715 | 0.838 |
| floor τ=0.1 | 20 | 0.170 ± 0.040 | 0.018 ± 0.016 | 0.560 | 0.821 |
| floor τ=0.3 | 20 | 0.120 ± 0.019 | 0.038 ± 0.036 | 0.584 | 0.643 |
| stage-2 η=0.05 | 5 | 0.303 ± 0.065 | 0.113 ± 0.053 | 0.727 | 0.896 |
| stage-2 η=0.05 | 20 | 0.415 ± 0.169 | 0.145 ± 0.061 | 0.468 | 0.875 |
| stage-2 η=0.2 | 20 | 0.340 ± 0.145 | 0.195 ± 0.074 | 0.437 | 0.862 |
| oracle | — | 1.000 | 1.000 | 0.000 | 0.000 |

Stage-2 lifts C (0.06 → 0.15–0.20) and the floor barely moves it; **neither closes the gap to averaging**, which is
2–8× higher on agreement at both L. θ spread across seeds is 1.5–1.85 (L1 over 80 coordinates) in every
configuration, i.e. the cone is wide and the fit is not stable.

**Defects of this diagnostic, stated:** the fits were capped at 10 outer iterations, and the fix rows do **not**
record the training objective or the outer gap (the values were computed but not written) — so a capped fit cannot be
distinguished from a converged one in that table. The degeneracy run took 7.8 min and the fix run 15 min, both above
the estimate given before launching.

**Consequence:** the mid-size grid is **not** run. The K=2n regime is not informative under this family until the
basis is prevented from placing weight where no observation lies (e.g. a support-restricted basis or a width tied to
the observed distance range), and that is a design decision, not a rerun.
