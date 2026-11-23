# Multi-observation relaxed inverse OT with sparse plans — report

Status: current (2026-11-23). Code `src/iot/sparse/`, study `iot_experiments/studies/synthetic/multi_observation_sparse/`,
records under `iot_experiments/out/multi_observation_sparse/` (git-ignored, as the convention requires). Measurement is
separated from interpretation and every number cites its record.

## 1. Question and claim

**Question.** Given `L` observed transport plans that are exactly optimal at one shared θ*, can the relaxed inverse-OT
objective `F(θ) = Σ_l [⟨c(θ), π̂_l⟩ − OT_l(c(θ))]` be minimised with memory linear in `n + m`, and does more than one
observation help at fixed feature family and Θ?

**Claim (intended contribution).** Scalability: the same objective as the dense reference and the same answer up to a
tolerance, with sparse plans and no stored feature tensor — memory `O(Σ_l|E_l| + K·t + (n+m)L)` instead of
`O(L·m·n)`. Time is not claimed: pricing visits every pair.

**State of that claim.** Correctness against the dense reference is established (§3). Scalability is **not
demonstrated** at the sizes run (§4): the bundle is slower at every size and the memory advantage is inside the
library baseline. The statistical results (§5) are synthetic, in-model, 5 seeds. There is no accuracy claim against
any state-of-the-art method, no claim on dense-flow tasks, and no claim that θ* is recovered.

## 2. What was built, and what was reused

1. `src/iot/sparse/features.py` — costs evaluated **on the fly** from two point clouds; no `m×n` array stored.
2. `src/iot/sparse/sparse_ot.py` — exact OT on a **growing edge set** with blockwise pricing and its own certificate;
   time stays quadratic, only memory is claimed linear.
3. `src/iot/sparse/bundle_multi.py` — Kelley cuts over θ, bracket `[master value, best exact F]`, homogeneity identity
   checked every iteration, stop at a zero subgradient.
4. Drivers `run.py`, `arms.py`, `diagnose.py`, `run_final.py`; `reference.py` (dense joint LP, dense FW) checks them.
5. Reused unchanged: `solve_cut_master`, `SimplexTheta`, the harness record/log/table utilities, POT.

## 3. Correctness

`iot_experiments/out/multi_observation_sparse/correctness/records.json` (`n = 30`, `L = 5`):

| quantity | value |
| --- | ---: |
| bundle exact `F` at its θ | 4.272e-08 |
| certified bracket | [−9.193e-07, 4.272e-08] |
| certified gap | 9.620e-07 |
| dense joint LP optimum | −0.0 |
| dense Frank–Wolfe best | 1.049e-05 |
| held-out plan agreement | 0.99942 |

The reported value lies inside a bracket containing the dense LP optimum. The oracle matches a dense OT to machine
precision at `n = 200` and `n = 800` (`.../inner/records.json`, difference ≤ 2e-16); at `n = 3200` no dense reference
exists (`dense_value = nan`), so only its own certificate applies (`inner_gap = 0`).

## 4. Scalability, stated honestly

`.../inner/records.json` (one cold solve, `γ = 2`):

| n | peak `|E|` | peak / (n+m) | sweeps | LP s | pricing s | total s | LP share |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 200 | 4 613 | 11.53 | 10 | 0.292 | 0.056 | 0.358 | 81.7 % |
| 800 | 26 138 | 16.34 | 12 | 3.351 | 0.956 | 4.337 | 77.3 % |
| 3200 | 139 310 | 21.77 | 16 | 44.274 | 21.694 | 66.153 | 66.9 % |

The peak edge set is 11.5–21.8× `n+m` and **grows** with `n` (the sweep count grows), so the memory constant is not
bounded. The inner LP is 67–82 % of the solve.

End-to-end against the dense implementation (`.../benchmark/records.json`, `L = 3`, dense arm capped at 10/5/1
iterations, so `F` is not a quality comparison):

| n | dense FW: F / s / peak MB | bundle: F / s / peak MB |
| ---: | --- | --- |
| 200 | 2.326e-06 / 0.470 / 630.6 | 1.401e-07 / 0.776 / 631.3 |
| 800 | 6.861e-05 / 3.452 / 726.1 | 4.079e-06 / 8.228 / 711.5 |
| 3200 | 6.976e-05 / 12.780 / 1805.2 | 6.976e-05 / 24.114 / 1573.8 |

**What this does not show.** No **crossover**: the bundle is slower at every size (1.6×, 2.4×, 1.9×). The peak-memory
difference is small and not monotone — **0.1 % larger** at `n = 200`, 2.0 % and 12.8 % smaller at 800 and 3200 — and is
**polluted** by a ~600 MB library baseline plus the generator's dense cost, so it does not measure asymptotic memory.
Nothing above `n = 3200`, and no timing with the production master: the LP is SciPy HiGHS, the **reference** path.
Whether the first-order GPU master changes the picture is open (it cannot certify `1e-6` and its slacks break
hysteresis pruning, so it would grow `|F|`). Memory never binds first; time does (66 s per cold solve at `n = 3200`).

## 5. Does more data help? (synthetic, in-model)

Converged cells only: `.../arms/final/records.json` (`n = 40`, 5 seeds, uniform marginals, cap 300, stop at gap ≤ 1e-6,
discriminating θ*, fresh points/marginals per observation, one fresh 40-pair held-out instance per seed). Derived
table: `.../arms/final/table.txt`. Agreement is the share of a new matching reproduced; `C−B` and `C−A` are per-seed.

| cell (K, L, p) | neutral | A single | B averaged | C joint | C − B per seed (mean) | C − A: positive, mean | C − neutral | iters | final gap | capped |
| --- | ---: | ---: | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: |
| 10, 10, 0 | 0.040 | 0.727 | 0.870 | **0.925** | +0.125, −0.125, +0.05, 0, +0.225 (+0.055) | 4/5, +0.198 | +0.885 | 85.0 | 0 | no |
| 20, 5, 0 | 0.015 | 0.714 | 0.805 | **0.845** | −0.05, −0.025, −0.10, +0.05, +0.325 (+0.040) | 4/5, +0.131 | +0.830 | 193.6 | 0 | no |
| 20, 10, 0 | 0.015 | 0.690 | 0.820 | **0.915** | +0.05, +0.025, +0.15, +0.05, +0.20 (+0.095) | 5/5, +0.225 | +0.900 | 242.6 | 2.4e-04 (seed 0) | 1/5 |
| 10, 5, 0.2 | 0.040 | 0.204 | 0.385 | **0.485** | −0.075, 0, +0.20, 0, +0.375 (+0.100) | 4/5, +0.281 | +0.445 | 116.2 | 0 | no |

Oracle (true θ*) is 1.000 in every cell. `K20 L10` seed 0 hit the cap and is **not converged**; it is kept in the
table, not extended, not dropped.

**What holds.** *Combining observations beats a single one.* Both C and B are above A in every cell; C beats A in
**17 of 20** seed-cells (mean +0.131 to +0.281), and averaging beats A in every cell. This is the robust effect.

**What does not hold: joint versus averaging is not settled.** C leads on average in all four cells, but the per-seed
sign flips in three. The seeds are **shared across cells** (same θ* and held-out instance), so the four cells are not
four independent confirmations. The per-seed mean of `C − B` over the cells is +0.012, −0.031, +0.075, +0.025,
+0.281: **seed 4 carries most of it** (without it ≈ +0.02).

**θ is mostly not recovered** (diagnostic, not a headline). L1 distance to θ* (uniform θ ≈ 1.0): K10 L10 C 0.35 vs B
0.60; K20 L5/L10 C 0.87/0.83 vs B 0.83/0.82, close to neutral weights. Plans transfer at 0.85–0.92 while the cost
parameters stay unidentified.

**Why the noise-free pattern looks like this (interpretation).** Single-plan fits are exactly feasible (regret ≈ 0,
5/5 every cell): they interpolate their own plan and reproduce only ≈ 0.7 of a new one. The joint fit is also exactly
feasible (5/5, 5/5, 4/5) but on all `L` plans at once, which shrinks the consistent set. Averaging leaves that set —
never feasible (regret 0.0084/0.0067/0.0099, 0/5) — a plausible reason it falls between. With noise nothing is
feasible (A 0.0147, C 0.1157, B 0.1665) and the joint advantage over a single plan is largest (0.485 vs 0.204).

**One plan with more points versus more observations (corrected).** `A_single_big`, converged in all seeds: n′ = 80
0.610 (per seed 1.0, 0.125, 0.6, 0.775, 0.55); n′ = 200 0.605 (0.95, 0.10, 0.95, 0.5, 0.525) — not better than the
single 40-point plans (≈ 0.7). The earlier summary compared n′ = 200 with the joint fit at L = 10, which is 400 points
and **not like-for-like**. The matched cell is not in `arms/final`, but it is in `.../arms/g3_deep/records.json`
(K=10, L=5, p=0, seeds 0–4 — five 40-point plans, i.e. the same 200 points), and I re-checked it: C 0.960 (per seed
0.85, 1.0, 0.95, 1.0, 1.0; 34–135 iterations, gap 0 in every seed), B 0.855, A 0.691, neutral 0.040. Both runs call
the same `run_cell` with the same seeds, so θ* and the held-out instance coincide. On that matched comparison, five
40-point observations beat one 200-point plan by +0.355 for the joint fit (one cell, 5 seeds, n′ = 200; the two records
are separate invocations of the same code).

### 5.1 Degenerate directions found, one line each

1. **Constant column** — a constant cost makes every plan optimal; excluded (`iot_experiments/studies/related/nyc_taxi/README.md`).
2. **Mask column** — admits `θ = e_mask` (`F_train ≈ 3` on taxi); excluded (`iot_experiments/studies/related/nyc_taxi/README.md`).
3. **Same-zone column** — a cost supported only there makes any diagonal-free plan optimal; diagonal mass 0.0000
   (`iot_experiments/studies/synthetic/planted_shared_theta/README.md`).
4. **Bins with no data** — at `K = 2n` the fit weights bins carrying zero observed mass and reaches `F ≈ 0` by
   construction (`.../diagnose/degeneracy/records.json`); the K = 2n grid is therefore uninformative.
5. **Monotone-family collapse** — nested hinges put the neutral plan at 0.925 of the planted one, at the ceiling
   (`ramp2`/`ramp3`, superseded by `ramp4`).

**Retracted diagnostic: weight on zero-mass bins.** At `K ≤ n/2` it does not separate a good fit from a broken one: in
the noise-free cells C puts 0.41–0.98 and B 0.60–0.95 of its weight on bins with no observed training mass, while
agreement is 0.85–0.93 against 0.02–0.04 for neutral; at K = 80 the same figure was ≈ 1.0 with agreement 0.04. A cost
that is high where no plan goes is real information, so the statistic needs θ*'s own weight there, which is not
recorded. The reading "the zero-mass hole, not identification" is withdrawn.

## 6. Real data (NYC taxi)

The taxi study fits a 263×263 matrix of trip **counts**, not a transport plan: the exact-OT decoder has ≤ 525
non-zeros against ~9 000 occupied cells, so it cannot represent the flow and predicts worse than a gravity baseline
(`iot_experiments/out/related_nyc_taxi_shuffle2/`; `.../nyc_taxi/README.md`). The mask arm matches the shuffled-distance
control to three decimals, so its number is the support's, not the geometry's. This dataset **does not predict flows**
and cannot test cost recovery; §5 uses a planted instance instead.

## 7. What is not established

- No accuracy claim; no claim on dense-flow tasks.
- **Scalability is not demonstrated**: no crossover, bundle slower everywhere, memory advantage within the baseline,
  nothing above `n = 3200`, nothing with the production master.
- Whether joint fitting beats averaging: signs flip, cells share seeds, seed 4 carries most of the gap.
- θ* is not recovered; at K = 20 the weights are close to uniform.
- Noise beyond one cell and 5 seeds; real data.
- `K20 L10 p0` seed 0 is not converged (gap 2.4e-04 at the cap).
- One fresh 40-pair held-out instance per seed: values move in steps of 0.025 and carry instance noise.

## 8. Superseded results (do not quote)

| superseded statement | where it was | why it is wrong / replaced |
| --- | --- | --- |
| A monotone basis is rank-equivalent; every monotone cost gives the same plan | `d6f6a43`, Add. 2 | False: `s`, `s²`, `s³` differ in 15–24 of 40 pairs; the crash was a zero subgradient, fixed in `85e6e31`. |
| The ramp crash proved a property of the family | same | Empty bank on windows already optimal at uniform θ. |
| Ramp readings 0.875–0.925 / C 0.975–0.995 | `ramp2`, `ramp3` | Near-ceiling instance; replaced by `ramp4`. |
| G3 C at 40 iterations (0.980 vs 0.870) | `arms/g3` | Capped (gap 2.4e-03); replaced by `arms/final`. |
| `B_informative`, first `g3`/`ramp4` pass | first pass | Excluded converged fits; regenerated. |
| K = 2n reversal is a basis degeneracy | Add. 1 | The K = 2n grid stays uninformative. |
| First NYC-taxi table (ratio 0.0045) | `nyc_taxi/README.md` | Scale artefact; replaced by share/fraction. |
| "The zero-mass hole, not identification" at K = 20 | previous revision | Retracted; the statistic is not a valid flag. |
| One 200-point plan (0.605) is worse than the joint fit (0.925) | previous revision | Not like-for-like (400 points); contrast open. |
| "C beats B" as a settled result | previous revision | Sign flips in 3/4 cells; seed 4 dominates. |

## Appendix: not-converged cells (not evidence)

- `.../diagnose/ramp4/` — ramp basis, 40 iterations: C capped 10/10 at K = 40 and 80 (L = 5, 20), gap 1e-2–7e-2.
- `.../arms/g3/` — G3 at 40 iterations: C capped 9/12 (gap 2.4e-03–1.2e-01; e.g. K20 L10 p0.2 = 1.2e-01).
- `.../arms/g3_big/` — big plan at 40 iterations: capped 5/5 at K = 20 (gap 5e-03–1.8e-02); at K20 L2 p0 the training
  objective is 8.6e-06, i.e. the single plan is exactly optimal at the fitted θ.
- `.../arms/g3_deep/` — K10 L5 p0 at a 200-iteration cap; superseded by `arms/final` for the headline cells, but it is
  the matched cell for the big-plan comparison (C 0.960, B 0.855, gap 0).
