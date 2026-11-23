# Multi-observation relaxed inverse OT with sparse plans and linear memory

**Status: implemented and measured (2026-11-20).** Code `src/iot/sparse/`, study
`iot_experiments/studies/iot/synthetic/multi_observation_sparse/`, tests `tests/iot_sparse_*_test.py`.

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
| dense references | `studies/iot/synthetic/multi_observation_sparse/reference.py` | (i) the exact problem as **one joint LP** in `(θ, {u_l,v_l})` — the ground truth `F*`; (ii) dense FW over an arbitrary `FeatureMap`. |
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

### Addendum 2: a monotone (hinge) basis — **corrected**: the first version of this addendum was wrong

Tried `RampFeatures`: `phi_k(s) = max(0, s - t_k)` with thresholds between the 1st and 99th percentile of the
sampled distance, planted weights `theta*_k ∝ exp(-t_k/0.3)`, unit-RMS columns, simplex Θ, n = 40, K = 10, L = 5.

**The run stops** with `RuntimeError: cut bank has no informative (nonzero) cuts`.

**Correction (2026-11-20).** The first version of this addendum said the summed gradient was zero at the initial θ,
that every monotone cost therefore induces the *same* plan, and that a monotone family is unidentifiable from plans
at all. That is false, and it came from reading one window as if it were the sum. The three checks below were run at
the user's request; they contradict the claim.

(a) **Monotone reweighting changes the plan.** On one fixed instance (n = 40, same two clouds, uniform marginals),
the exact-OT matchings under cost `s`, `s²`, `s³` differ in 15/40, 24/40 and 18/40 row targets respectively
(`s` vs `s²`, `s` vs `s³`, `s²` vs `s³`). In 2-D the magnitudes matter, not only the ordering of distances — the
1-D argument (where any increasing cost gives the same monotone matching) does not transfer.

(b) **The basis is severely collinear, and its top column is nearly empty.** At K = 10 the thresholds are
0.0598 … 1.1142 while the sampled distance spans 0.0117 … 1.3300; the fraction of pairs with `phi_k > 0` falls from
0.988 (k = 0) to 0.013 (k = 9); the maximum off-diagonal correlation is 0.9976 and the smallest eigenvalue of the
correlation matrix is 6.76e-04. Hinges on a fixed range are nested, so they are close to linearly dependent.

(c) **The observed plans are not all the uniform-cost plans.** Comparing the planted plan with the exact-OT plan at
uniform θ, per window: 0/40, 6/40, 0/40, 2/40 and 9/40 row targets differ.

**What actually raised the error.** `F(uniform)` summed over the five windows is `5.3e-4`, not zero; per window it
is `−1.7e-18`, `1.5e-4`, `0`, `1.3e-6`, `3.8e-4`. The zero was window 0's contribution alone, which is exactly the
window whose observed plan is already the uniform-cost plan. The crash comes from the **per-window fits of arm B**:
a window that is already optimal at uniform has a zero subgradient at the initial θ, so the cut bank is empty and the
master LP (correctly) refuses it. It is not a statement about the family.

**How common is it:** over 10 seeds, n = 40, a window is already optimal at uniform in 19/50 (K = 10), 29/50
(K = 40) and 30/50 (K = 80) cases. The **joint** `F(uniform)` is positive in 9/10 seeds at every K, and is `≈ 0` in
1/10 seeds at K = 80 (all five windows optimal at uniform) — so the joint fit can hit the same empty bank.

**Fix (a real bug in the new code, not a tuning choice).** `solve_bundle_multi` now stops when the subgradient is
zero: `F(θ) = 0` is the global minimum because `F ≥ 0`, and the cut would be the always-true halfspace
`0·θ' ≤ 0`. It records that iteration with `lower = upper = value` and reports a zero-gap bracket; it no longer hands
an empty bank to `solve_cut_master`. Covered by `tests/iot_bundle_multi_test.py`.

**Not claimed.** This says nothing yet about whether the ramp family identifies θ better than the bump family; the
K ∈ {10, 40, 80}, L ∈ {5, 20} rerun is the measurement for that. It also does not remove the K = 2n zero-mass-bin
hole, which was found on the bump basis and is untouched here.

#### Addendum 2b: the first rerun (hinge basis, planted θ*) — **inadequate, superseded by 2c**

That run used the hinge basis with evenly spaced thresholds and the analytic planted θ* `∝ exp(-t_k/0.3)`. It is kept
only to record why it cannot answer the question: **the neutral plan was already near the ceiling**. Measured on the
same cells, neutral (uniform θ) held-out agreement was **0.925 on K = 10, seed 0** (mean **0.875** over the 10 seeds),
while the joint fit reached 0.975–0.995. The apparent gain was therefore at most +0.05, inside the seed spread, and
agreement could not separate the arms. The user's reading of ~0.91 is confirmed (0.925 measured; the difference is
seed/cell choice).

#### Addendum 2c: the discriminating rerun — neutral baseline, three bases, and B with/without skipped windows

Everything below was fixed **before** any arm was fitted in this configuration.

**Basis (chosen on conditioning only, no arm result seen).** Rule: among {hinge, first difference} × {range,
quantile} thresholds, take the variant with the largest smallest correlation eigenvalue at K = 20, provided no column
is positive on fewer than 4 % of pairs; ties broken at K = 10. Measured (n = 40, full pair grid, subsample-RMS
columns):

| variant | K=10 max abs corr / min eig | K=20 max abs corr / min eig | min column positivity |
| --- | --- | --- | ---: |
| hinge / range (old) | 0.9976 / 6.76e-04 | 0.9996 / 5.15e-05 | 0.013 |
| hinge / quantile | 0.9957 / 2.54e-04 | 0.9994 / 3.11e-05 | 0.047 |
| difference / range | 0.8289 / 6.69e-02 | 0.9180 / 2.65e-02 | 0.013 |
| **difference / quantile** | 0.8921 / 3.73e-02 | 0.9425 / **1.75e-02** | **0.047** |

`difference / range` has the better K = 20 eigenvalue but a column positive on 1.3 % of pairs, so it fails the
guard; `difference / quantile` is 340× better conditioned than the old basis at K = 20 and 55× at K = 10. Chosen:
**first-difference hinge columns with quantile thresholds** (`RampFeatures(threshold_rule="quantile",
differences=True)`). Each column is `max(0, s-t_k) - max(0, s-t_{k+1})`: zero below `t_k`, rising on
`(t_k, t_{k+1})`, saturating at `t_{k+1}-t_k`; the columns telescope to the first hinge. Tests:
`tests/iot_sparse_ramp_test.py`.

**θ* (chosen on discrimination only).** `θ* ~ Dirichlet(0.5·1_K)`, accepted iff on a **fresh** instance the neutral
plan reproduces at most 50 % of the `θ*` plan (`Σ min(P_neutral, P*) / Σ P* ≤ 0.5`); at most 200 draws, else the
cell raises. Measured acceptance 10–25 % (K = 10/20/40/80); mean draws to accept 5.2–7.9 in the run below. Rule and
acceptance are identical for every arm, so no arm sees a different instance.

**Run.** `--mode ramp --n 40 --seeds 0 … 9 --ks 10 40 80 --ls 5 20 --iterations 40 --tag ramp4`
(695 s, `out/multi_observation_sparse/diagnose/ramp4/`, table by `analyse_ramp.py --tag ramp4`). Paired design:
θ* and the held-out instance depend only on (K, seed), so L = 5 and L = 20 within a K are compared on the same
held-out instance. Gain = mean ± sd of the **per-seed difference** `arm − neutral`. `B` is the unweighted mean of
the per-window fits (each sums to 1, so the simplex projection is a no-op); `B_informative` drops windows whose fit
stopped at iteration 1 (observed plan already optimal at uniform θ). `D_oracle` is the planted θ*.

| cell | neutral | C | C − neutral | B all | B all − neutral | B informative | B inf − neutral | oracle |
| --- | ---: | ---: | --- | ---: | --- | ---: | --- | ---: |
| K10 L5 | 0.698±0.160 | 0.570±0.287 | **−0.128±0.313** | 0.828±0.137 | +0.130±0.126 | 0.715±0.284 | +0.018±0.278 | 1.000 |
| K10 L20 | 0.698±0.160 | 0.855±0.113 | **+0.158±0.148** | 0.812±0.137 | +0.115±0.161 | 0.787±0.162 | +0.090±0.141 | 1.000 |
| K40 L5 | 0.520±0.161 | 0.758±0.155 | +0.237±0.253 | 0.753±0.188 | +0.233±0.152 | 0.753±0.188 | +0.233±0.152 | 1.000 |
| K40 L20 | 0.520±0.161 | 0.762±0.171 | +0.242±0.220 | 0.798±0.205 | +0.277±0.153 | 0.798±0.205 | +0.277±0.153 | 1.000 |
| K80 L5 | 0.613±0.225 | 0.733±0.084 | +0.120±0.195 | 0.743±0.152 | +0.130±0.103 | 0.743±0.152 | +0.130±0.103 | 1.000 |
| K80 L20 | 0.613±0.225 | 0.735±0.076 | +0.123±0.202 | 0.763±0.160 | +0.150±0.170 | 0.763±0.160 | +0.150±0.170 | 1.000 |

Skipped windows (fit stopped at iteration 1, i.e. observed plan already optimal at uniform θ), per cell, out of L:
7.2/20 and 1.9/5 at K = 10; 0.1/20 and 0.1/5 at K = 40; 0.0 at K = 80. Mean draws to accept θ*: 7.9, 6.1, 5.2 at
K = 10, 40, 80.

**Measurements (no interpretation yet).**
- The neutral baseline is now 0.52–0.70, not 0.87–0.93, so the readings have room for a gain.
- At **K = 10**, C goes from **−0.128** (L = 5) to **+0.158** (L = 20) relative to neutral; B stays at +0.130 /
  +0.115. The spreads are 0.13–0.31, so this is a measurement, not a claim.
- At **K ≥ 40 the joint fit is not converged**: `hit_iteration_cap` is 1.0 (10/10 seeds) at L = 5 and L = 20, with a
  certified gap of 1e-2–7e-2. Its gain column is the best *queried* θ, so those rows cannot compare C with B.
- Including the skipped windows **raises** B at K = 10 (0.828 vs 0.715 at L = 5; 0.812 vs 0.787 at L = 20); at
  K ≥ 40 the two variants coincide because almost nothing is skipped. So at K = 10 "B" is not "B without the
  uniform windows", and the two must be reported separately.
- Support readings: `weight_on_zero_mass_ramps` is **0.350–0.595 for the neutral arm**, 0.177–0.439 for C and
  0.077–0.157 for `B_informative`. So the *neutral* θ is the arm that leans hardest on ramps with no observed mass —
  the hole is not something the fit introduces. The pre-registered mass-share flag (< 0.2) still fires nowhere (min
  0.376), because every arm including neutral weights all K columns at K = 80; `weight_on_zero_mass_ramps` is the
  reading that separates them.
- `cost_rel`: C 0.240–0.323, B 0.250–0.372. B is not uniformly better on cost either.

**Defects, stated.**
1. The acceptance rule is checked on **one** fresh instance per candidate, not on the held-out instance. Measured
   held-out neutral agreement is 0.52–0.70 except **one cell at 0.95** (K = 80, L = 20, seed 6), so the rule
   improves but does not guarantee discrimination out of sample. Requiring two fresh instances would.
2. K ≥ 40 is uncertified at 40 iterations (above). A 200-iteration check would be needed before using those rows.
3. `K10 L5`: C is worse than neutral with a gap of 0, i.e. it genuinely converged to a θ that generalises worse than
   uniform. That is a property of this basis at K = 10 with 5 windows, not a solver failure.

**Interpretation, separated.** At K = 10 the paired L = 5 → L = 20 movement is large for C (−0.128 → +0.158) and
small for B (+0.130 → +0.115). With 10 seeds and spreads of 0.13–0.31 this is consistent with "more windows help the
joint fit" but does not establish it; the K ≥ 40 rows that could have tested it are uncertified. θ recovery is **not**
a headline reading for this family (see the note's standing caveat): it is recorded per row and printed only by
`analyse_ramp.py --theta`.

### Addendum 3: the G3 arms grid and the bounded big-plan arm

**Config (fixed before running).** `n = 40`, `K ∈ {10, 20}`, `L ∈ {1, 5, 10}`, corruption `p ∈ {0, 0.2}`, uniform
marginals, no control, 5 seeds, 40 outer iterations, bump basis (`DistanceProfileFeatures`), `θ*` by the same
rejection rule as Addendum 2c (`Dirichlet(0.5)`, fresh-instance neutral agreement ≤ 0.5, ≤ 200 draws). On this basis
the rule accepts almost immediately (neutral held-out agreement 0.040 at K = 10, 0.015 at K = 20), so the instances
are strongly discriminating and `E_neutral` is far from the ceiling. Commands and records:
`arms --tag g3` (809 s), `arms --only-big --tag g3_big` (1005 s), `arms --iterations 200 --tag g3_deep` (62 s),
tables by `analyse_arms.py`. Gain over neutral is the mean ± sd of the **per-seed** difference; for `A_single` the
per-seed value is the mean over that seed's plans.

| cell | neutral | A single | B averaged | B − neutral | C joint | C − neutral | C − B | C capped |
| --- | ---: | ---: | ---: | --- | ---: | --- | --- | ---: |
| K10 L1 p0 | 0.040 | 0.650 | 0.650 | +0.610±0.215 | 0.650 | +0.610±0.215 | 0.000 | 0.20 |
| K10 L5 p0 | 0.040 | 0.663 | 0.870 | +0.830±0.165 | 0.980 | +0.940±0.082 | +0.110±0.121 | 0.80 |
| K10 L10 p0 | 0.040 | 0.702 | 0.880 | +0.840±0.151 | 0.955 | +0.915±0.119 | +0.075±0.120 | 1.00 |
| K20 L1 p0 | 0.015 | 0.660 | 0.660 | +0.645±0.219 | 0.660 | +0.645±0.219 | 0.000 | 1.00 |
| K20 L5 p0 | 0.015 | 0.514 | 0.720 | +0.705±0.203 | 0.535 | +0.520±0.400 | −0.185±0.563 | 1.00 |
| K20 L10 p0 | 0.015 | 0.544 | 0.795 | +0.780±0.132 | 0.570 | +0.555±0.356 | −0.225±0.418 | 1.00 |
| K10 L1 p0.2 | 0.040 | 0.260 | 0.260 | +0.220±0.212 | 0.260 | +0.220±0.212 | 0.000 | 1.00 |
| K10 L5 p0.2 | 0.040 | 0.200 | 0.280 | +0.240±0.245 | 0.515 | +0.475±0.324 | +0.235±0.159 | 1.00 |
| K10 L10 p0.2 | 0.040 | 0.174 | 0.360 | +0.320±0.357 | 0.535 | +0.495±0.248 | +0.175±0.210 | 1.00 |
| K20 L1 p0.2 | 0.015 | 0.455 | 0.455 | +0.440±0.129 | 0.455 | +0.440±0.129 | 0.000 | 1.00 |
| K20 L5 p0.2 | 0.015 | 0.278 | 0.465 | +0.450±0.298 | 0.415 | +0.400±0.205 | −0.050±0.310 | 1.00 |
| K20 L10 p0.2 | 0.015 | 0.295 | 0.560 | +0.545±0.351 | 0.320 | +0.305±0.259 | −0.240±0.213 | 1.00 |

C diagnostics per cell (objective, outer gap, iterations, cap hit, skipped windows) are recorded per row in
`g3/records.json`; the outer gap is 4.6e-05 (K10 L1 p0) and **2.4e-03 – 1.2e-01** everywhere else, with the cap hit
in 9/12 cells. No window is skipped for a zero gradient on this basis (`skipped_windows = 0.0` everywhere), so
`B_averaged` and `B_informative` coincide at every cell and the latter is not tabulated.

**The cap makes the C column unusable as a comparison.** Only `K10 L1 p0` is converged. With `theta*` genuinely
discriminating, 40 Kelley iterations are far from enough (gap up to 0.12 at K20 L10 p0.2). The C−B column is
therefore the difference between a *capped* joint fit and a converged average, not between two optima.

**One certified cell (the only C-vs-B statement that survives).** Re-running `K10 L5 p0` with 200 allowed iterations
(`g3_deep`) lets C converge: 67.6 iterations, gap ≈ 0, objective ≈ 0. C = **0.960**, B = 0.855, neutral = 0.040,
`C − B = +0.105 ± 0.182` per seed (individual seeds −0.05, −0.025, 0, +0.25, +0.35) and `cost_rel` 0.223 (C) versus
0.339 (B). So at this cell the joint fit is ahead when it is actually solved, but 3 of 5 seeds show no difference and
the spread is larger than the effect.

**Bounded big-plan arm** (`A_single_big`: one plan of `L·n` points, `L ∈ {1, 2, 5}`, i.e. n′ = 40/80/200; same fresh
held-out instance and θ*):

| cell | n′ | agreement | cost_rel | iterations | capped | gap |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| K10 p0 L1/2/5 | 40/80/200 | 0.695 / 0.600 / 0.700 | 0.392 / 0.384 / 0.362 | 40 / 36 / 40 | 3/5 / 2/5 / 4/5 | 1.2e-04 / 0 / 1.2e-04 |
| K10 p0.2 L1/2/5 | 40/80/200 | 0.185 / 0.115 / 0.120 | 0.423 / 0.425 / 0.441 | 40 / 40 / 40 | 5/5 / 5/5 / 5/5 | 4.0e-04 / 5.2e-04 / 2.7e-04 |
| K20 p0 L1/2/5 | 40/80/200 | 0.640 / 0.290 / 0.180 | 0.585 / 0.715 / 0.795 | 34 / 40 / 40 | 2/5 / 5/5 / 5/5 | 0 / 5.2e-03 / 7.3e-03 |
| K20 p0.2 L1/2/5 | 40/80/200 | 0.250 / 0.230 / 0.240 | 0.839 / 0.841 / 0.596 | 40 / 40 / 40 | 5/5 / 5/5 / 5/5 | 1.6e-04 / 2.4e-03 / 4.5e-03 |

The apparent collapse at K = 20 with more points (0.64 → 0.18) is **not** a result: those cells hit the cap (5/5)
with gap 5e-3–1.8e-2, and at K20 L2 p0 the training objective is 8.6e-06 — the fit has found a θ at which the single
observed plan is optimal, but the θ is not identified by one plan, so its held-out plan differs. Report as "one plan
does not identify θ", not as "more points hurt".

**Cost accounting, measured.** G3 was estimated at ~25–30 min for 60 cells plus the big pass; actual 809 s (13.5 min)
and 1005 s (16.7 min). The big pass exceeded its ~8 min estimate because each n′ = 200 fit runs to the 40-iteration
cap (~30–48 s per cell). The deep check cost 62 s.
