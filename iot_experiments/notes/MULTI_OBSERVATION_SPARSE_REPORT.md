# Multi-observation relaxed inverse OT with sparse plans — report

Status: current. Scope: the `src/iot/sparse` solver and the `multi_observation_sparse` study. Measurement and
interpretation are separated throughout; every number cites the record it was read from. Generated records live under
`iot_experiments/out/multi_observation_sparse/` (git-ignored, as the repository convention requires).

## 1. Question and claim

**Question.** Given `L` observed transport plans that are exactly optimal at one shared cost parameter θ*, can the
relaxed inverse-OT objective `F(θ) = Σ_l [⟨c(θ), π̂_l⟩ − OT_l(c(θ))]` be minimised with memory linear in `n + m`, and
does more than one observation help at fixed feature family and Θ?

**Claim (the contribution).** A **scalability** contribution: the same objective and the same answer as the dense
formulation, with plans kept sparse and the feature tensor never stored, so memory is `O(Σ_l|E_l| + K·t + (n+m)L)`
instead of `O(L·m·n)`. There is **no accuracy claim** against any state-of-the-art method, **no** claim on dense-flow
tasks, and no claim that θ* is recovered.

## 2. What was built, and what was reused

1. `src/iot/sparse/features.py` — cost families evaluated **on the fly** (point distances, radial profiles, hinge
   ramps), each column scaled by a subsample RMS. No `m×n` array is stored.
2. `src/iot/sparse/sparse_ot.py` — exact OT on a **growing edge set**: sparse LP, blockwise pricing, its own
   certificate, north-west-corner repair. **Time stays quadratic**; only memory is claimed linear.
3. `src/iot/sparse/bundle_multi.py` — Kelley cuts over θ from exact per-window solves, bracket
   `[master value, best exact F]`, homogeneity identity checked every iteration, stop at a zero subgradient.
4. Drivers: `run.py`, `arms.py`, `diagnose.py`, `run_final.py`, with `reference.py` (dense joint LP, dense Frank–Wolfe)
   as the independent check.
5. Reused unchanged: `solve_cut_master` (the Kelley LP), `SimplexTheta`, the harness record/log/table utilities, POT.

## 3. Correctness

`iot_experiments/out/multi_observation_sparse/correctness/records.json` (`n = 30`, `L = 5`, one record):

| quantity | value |
| --- | ---: |
| bundle exact `F` at its θ | 4.272e-08 |
| certified bracket | [−9.193e-07, 4.272e-08] |
| certified gap | 9.620e-07 |
| dense joint LP optimum | −0.0 (reported as `-0.0`) |
| dense Frank–Wolfe best | 1.049e-05 |
| held-out plan agreement | 0.99942 |

The reported value lies inside a bracket that contains the dense LP optimum. The oracle matches a dense OT to machine
precision at `n = 200` and `n = 800` (`.../inner/records.json`, difference ≤ 2e-16); at `n = 3200` no dense reference
exists (`dense_value = nan`), so only its own certificate applies (`inner_gap = 0`).

## 4. Scalability, stated honestly

From `.../inner/records.json` (one cold solve, `γ = 2`):

| n | peak `|E|` | peak / (n+m) | sweeps | LP s | pricing s | total s | LP share |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 200 | 4 613 | 11.53 | 10 | 0.292 | 0.056 | 0.358 | 81.7 % |
| 800 | 26 138 | 16.34 | 12 | 3.351 | 0.956 | 4.337 | 77.3 % |
| 3200 | 139 310 | 21.77 | 16 | 44.274 | 21.694 | 66.153 | 66.9 % |

**Measured.** The peak edge set is 11.5–21.8× `n+m` and **grows** with `n` (the sweep count grows), so the memory
constant is not bounded by the current pricing schedule. The inner LP is 67–82 % of the solve. A cold solve takes
0.36 s at `n = 200` and 66 s at `n = 3200`.

**End-to-end against the dense implementation** (`.../benchmark/records.json`, `L = 3`, dense arm capped at 10/5/1
iterations, so `F` is not a quality comparison):

| n | dense FW: F / s / peak MB | bundle: F / s / peak MB |
| ---: | --- | --- |
| 200 | 2.326e-06 / 0.470 / 630.6 | 1.401e-07 / 0.776 / 631.3 |
| 800 | 6.861e-05 / 3.452 / 726.1 | 4.079e-06 / 8.228 / 711.5 |
| 3200 | 6.976e-05 / 12.780 / 1805.2 | 6.976e-05 / 24.114 / 1573.8 |

**NOT shown — and this is the honest summary of the section.** There is **no crossover**: the bundle is slower than the
dense Frank–Wolfe at every measured size (1.6×, 2.4×, 1.9×). The peak-memory difference is small and not monotone in sign: the bundle's
peak is **0.1 % larger** at `n = 200`, 2.0 % smaller at `n = 800`, and 12.8 % smaller at `n = 3200` and is **polluted** by a ~600 MB library baseline plus the generator's own dense
cost, so it does not measure the solver's asymptotic memory. Nothing above `n = 3200` was measured, and no timing was
measured with the production master: the LP here is SciPy HiGHS, the repository's **reference** path. Whether the
first-order GPU master changes the wall-clock picture is an open question — it cannot certify a `1e-6` gap and its
first-order slacks break hysteresis pruning, so it would grow `|F|` rather than shrink it.

## 5. Does more data help? (synthetic, in-model)

Only **converged** cells are headline evidence. They come from `.../arms/final/records.json` (`n = 40`, 5 seeds,
uniform marginals, iteration cap 300, stop when the certified gap ≤ 1e-6, discriminating θ* by rejection sampling;
table `.../arms/final/table.txt`). `C − B` is the **per-seed** difference, all five values; `C − neutral` is the mean.

| cell | neutral | A single | B averaged | C joint | C − B per seed (mean) | C − neutral | iters | final gap | capped | w on zero-mass bins |
| --- | ---: | ---: | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: |
| K10 L10 p0 | 0.040 | 0.727 | 0.870 | **0.925** | +0.125, −0.125, +0.05, 0, +0.225 (**+0.055**) | +0.885 | 85.0 | 0 | no | 0.72 |
| K20 L5 p0 | 0.015 | 0.714 | 0.805 | **0.845** | −0.05, −0.025, −0.1, +0.05, +0.325 (**+0.040**) | +0.830 | 193.6 | 0 | no | 0.73 |
| K20 L10 p0 | 0.015 | 0.690 | 0.820 | **0.915** | +0.05, +0.025, +0.15, +0.05, +0.2 (**+0.095**) | +0.900 | 242.6 | 2.4e-04 (seed 0) | 1/5 | 0.69 |
| K10 L5 p0.2 | 0.040 | 0.204 | 0.385 | **0.485** | −0.075, 0, +0.2, 0, +0.375 (**+0.100**) | +0.445 | 116.2 | 0 | no | 0.03 |

K20 L10 p0 is converged on 4 of 5 seeds; seed 0 still hit the 300-iteration cap with gap 2.4e-04 and is **not**
converged — it is kept in the table, flagged, and not extended. All other cells converged.

**Measurement.** With the joint fit converged, C is ahead of B by +0.04 to +0.10 in every cell, and ahead of the
neutral plan by +0.45 to +0.90. The per-seed differences change sign (−0.125 to +0.375), so the mean effect is smaller
than the seed spread; 3 of the 20 individual differences are ≤ 0.

**One plan of many points vs many observations** (`.../arms/final/records.json`, `A_single_big`, K = 10, p0, converged
in every seed): `n′ = 80` gives 0.610 (45 iterations), `n′ = 200` gives 0.605 (58.8 iterations; per seed 0.95, 0.1,
0.95, 0.5, 0.525) against **0.925** for ten 40-point observations — same θ*, same held-out instance, both converged.

**Interpretation (labelled as such).** At this size extra *observations* carry more information about θ than extra
*points in one plan*, when both are solved to the same certificate. In-model, synthetic; no statement about real data
or `n ≥ 3200`. It is **not** evidence that θ* is recovered — the converged θ puts 0.69–0.72 of its weight on bins with
**zero observed mass** (table), the hole of 5.1, not identification.

### 5.1 Degenerate directions found, one line each

1. **Constant column** — a constant cost makes every plan optimal; excluded (`.../nyc_taxi/README.md`).
2. **Mask column** — admits `θ = e_mask` (`F_train ≈ 3` on taxi); excluded (`.../nyc_taxi/README.md`).
3. **Same-zone column** — a cost supported only there makes any diagonal-free plan optimal; diagonal mass 0.0000
   (`.../planted_shared_theta/README.md`).
4. **Bins with no data** — at `K = 2n` the fit weights bins with zero observed mass and reaches `F ≈ 0` by construction
   (`.../diagnose/degeneracy/records.json`); still visible at `K = 20` (0.69–0.73 above).
5. **Monotone-family collapse** — nested hinges put the neutral plan at 0.925 of the planted one, at the ceiling
   (`ramp2`/`ramp3`, superseded by `ramp4`).

## 6. Real data (NYC taxi)

The taxi study fits a 263×263 matrix of trip **counts**, which is not an optimal-transport plan: the exact-OT decoder
has at most 525 non-zeros against ~9 000 occupied cells, so the formulation cannot represent the observed flow and its
held-out prediction is worse than a gravity baseline
(`iot_experiments/out/related_nyc_taxi_shuffle2/`; `iot_experiments/studies/related/nyc_taxi/README.md`). The
mask arm reproduces the shuffled-distance control to three decimals — its number is set by the support, not the
geometry. This dataset **does not predict flows** and cannot test whether a cost is recovered; the clean test is a
planted instance, which section 5 uses.

## 7. What is not established

- No accuracy claim, against any method; no claim on dense-flow tasks.
- No crossover at any measured size; the bundle is slower everywhere.
- The memory constant is not bounded (peak `|E|` grows 11.5 → 21.8 × `(n+m)`).
- Nothing above `n = 3200`; nothing with the production (first-order GPU) master.
- θ* is not recovered; no bound on the distance to it. At `K = 20` the fitted θ puts most weight where no observation
  lies.
- `C > B` holds only for the four converged cells; the sign changes per seed, and the K ≥ 40 ramp cells and most of the
  40-iteration G3 grid are not converged (appendix).
- Nothing implies the sparse formulation is preferable on real workloads or beats a dense GPU solver in wall time.

## Appendix: not-converged cells (not evidence)

- `.../diagnose/ramp4/` — ramp basis, 40 iterations: C capped 10/10 at K = 40 and 80 (L = 5, 20), gap 1e-2–7e-2.
- `.../arms/g3/` — G3 at 40 iterations: C capped 9/12 (gap 2.4e-03–1.2e-01, e.g. K20 L10 p0.2 = 1.2e-01);
  four of those cells are replaced by the headline table.
- `.../arms/g3_big/` — big plan at 40 iterations: capped 5/5 at K = 20 (gap 5e-03–1.8e-02); at K20 L2 p0 the
  training objective is 8.6e-06, i.e. one plan does not identify θ.
- `.../arms/g3_deep/` — first converged cell (K10 L5 p0, 200 iterations, C 0.960 vs B 0.855, `C−B = +0.105±0.182`);
  superseded by `arms/final`, kept as the pre-registered single-cell check.

## 8. Superseded results (do not quote)

| superseded statement | where it was | why it is wrong / replaced |
| --- | --- | --- |
| A monotone (hinge) basis is rank-equivalent; every monotone cost gives the same plan | `d6f6a43`, Addendum 2 | False: `s`, `s²`, `s³` give different matchings (15/40, 24/40, 18/40 pairs). The crash was a zero subgradient at iteration 1, fixed in `85e6e31`. |
| The ramp crash proved a property of the cost family | same | Empty cut bank on a window already optimal at uniform θ (19/50, 29/50, 30/50 at K = 10/40/80). |
| Ramp readings: neutral 0.875–0.925, C 0.975–0.995 | `ramp2/`, `ramp3/` | Near-ceiling instance; gains inside the seed spread. Replaced by `ramp4`; do not quote. |
| G3 C column at 40 iterations (K10 L5 p0: C 0.980 vs B 0.870) | `.../arms/g3/` | C was capped (gap 2.4e-03), not an optimum. Replaced by `arms/final` (85 iterations, gap 0, C 0.925 vs B 0.870). |
| `B_informative` in the first `g3`/`ramp4` passes | first pass | Excluded *converged* singles; averaged the worst fits. Regenerated with "stopped at iteration 1 only". |
| "K = 2n reversal is a degeneracy of the basis" (`f609abc`) | Addendum 1 | The K = 2n grid stays uninformative; no number from it is evidence about K. |
| First NYC-taxi table (ratio 0.0045) | `nyc_taxi/README.md` | Scale artefact (`R(αc) = αR(c)`); replaced by share/fraction and the shuffled control. |
