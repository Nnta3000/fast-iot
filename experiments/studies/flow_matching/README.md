# flow_matching — does the coupling pool size matter?

**Status: measured, v2 (2026-11-24).** This is a study of **scalable exact forward OT** with HELLO-OT as the eventual
solver. It is **not** an inverse-OT experiment and it is **not** evidence for the relaxed-iOT method; the two use the
same repository for unrelated reasons. No number here is a claim about CIFAR/FFHQ, and the paper's FFHQ figure
(global coupling vs minibatch OT) is context we have not verified — this toy tests the mechanism (coupling pool size,
straightness, NFE), not that claim.

## Question

At a fixed training batch of 256, does the **coupling** change what a flow learns: does an exact OT plan over a larger
pool, up to one global plan over the whole training set, produce a better model or the same quality with fewer
function evaluations? This is the `B = N` limit of the batch-size argument behind OT-CFM, and it is the quantity a
GPU exact-OT solver (HELLO) would change.

## Protocol (fixed before the training sweep)

| item | setting |
| --- | --- |
| data | 8-Gaussians and two-moons (d = 2), plus an eight-mode Gaussian mixture at d = 8 and d = 32; 8 192 training points |
| source | `N(0, I₂)` |
| model | MLP 2→128→128→2, SiLU, four Fourier time features; Adam 1e-3 |
| loss | CFM: `‖v_t(x_t) − (x₁ − x₀)‖²`, `x_t = (1−t)x₀ + t x₁` |
| training batch | **256, fixed**; only the coupling pool varies |
| arms | `independent` (fresh noise, i.i.d. partner); `fixed_random` (fixed 8 192 noise points + fixed random permutation — the control for `global_ot`); `ot_pool` at `B = 256` and `B = 1024` (fresh pool every step, exact POT EMD); `global_ot` (one exact plan between the fixed noise set and the whole training set, computed once) |
| seeds | 10 for `independent`, `fixed_random`, `ot_pool_B256`, `global_ot`; 3 for `ot_pool_B1024` |
| budgets | 1 000 and 10 000 steps for the cheap arms; 1 000 for `B = 1024` (POT costs 0.082 s/step there) |
| samplers | Euler NFE ∈ {1, 2, 5, 10, 50}; RK4 NFE ∈ {4, 8, 20, 48} |
| primary metric | **sliced 2-Wasserstein**, 512 random projections, 4 096 generated vs a **fresh** 4 096-point true sample; the energy distance (1 024) and the exact 1 024-point W2 are reported as secondary |
| diagnostics | flow straightness (displacement / path length, 1 = straight, 50-step Euler rollout); mode-weight L1, per-mode centre and covariance errors for the mixtures |
| floor | W2 between a **fresh sample of the true distribution** and the held-out set, averaged over 3 draws; differences are read against it |
| coupling cost | plan time and training time recorded; both equal-step and equal-wall-clock are reportable |

**Why the control.** `global_ot` fixes 8 192 noise points for the whole run; the other arms draw fresh noise every
step. A model can memorise a finite fixed noise set, so `global_ot` vs `independent` mixes coupling quality with that
finite-noise effect. `fixed_random` shares the same fixed noise points and a fixed random permutation, so
`global_ot − fixed_random` isolates the coupling.

**Metric note (v2).** The v1 exact W2 was computed against a finite 1 024-point held-out set, and three models beat
the "floor" computed between two true samples; that is a finite-sample bias, not a better-than-perfect model. v2 uses
the sliced W2 against a fresh 4 096-point sample and reports its own floor, so the comparison is not read against a
quantity the model can beat by being slightly under-dispersed. The fraction of differing pairs in the pre-check is not
interpreted: a pool plan cannot reproduce a global partner that is absent from the pool, so it is near 1 by
construction; the cost gap is the informative column. Seed spreads are per-seed standard deviations; the standard error
(`sd/√n`) is reported with them.

**Pre-registered reading.**
- *Signal*: at low NFE (Euler 10, RK4 20), `global_ot` beats `fixed_random` by more than the seed spread, and
  `ot_pool_B1024` beats `ot_pool_B256`.
- *Null*: no separation beyond the spread. The honest conclusion is then "this 2-D instance is too easy to show a
  coupling effect", **not** "OT couplings do not help"; the study stops, it is not tuned.
- A model that is straighter because it optimises faster is a different claim from a better final model; the 1 000 and
  10 000 step budgets are reported separately.

The pre-check (`--mode precheck`) measures, on the study's own targets and on the same noise points, how far a pool
pairing is from the global pairing: the fraction of different partners, the mean squared displacement of each pairing,
and the relative cost gap. It is a diagnostic, not a stop rule.

## Run

```bash
PYTHONPATH=. .venv/bin/python -m experiments.studies.flow_matching.run --mode precheck
PYTHONPATH=. .venv/bin/python -m experiments.studies.flow_matching.run --mode precompute --workers 4
PYTHONPATH=. .venv/bin/python -m experiments.studies.flow_matching.run --mode train
PYTHONPATH=. .venv/bin/python -m experiments.studies.flow_matching.run --mode evaluate
PYTHONPATH=. .venv/bin/python -m experiments.studies.flow_matching.analyse --tag coupling_v2
```

Records: `experiments/out/records/flow_matching_coupling_v2/`.

## Results (v2)

Records: `experiments/out/records/flow_matching_coupling_v2/` (`train.json`, `eval.json`, `precheck.json`, `table.csv`,
`run.log`); the driver also writes copies under `experiments/out/flow_matching/coupling_v2/`. **Note on the run:** the
background training process did not survive an interruption of the session; 214 of 246 checkpoints were already
written, the 32 missing d = 32 runs were trained afterwards, and `train.json` was rebuilt from the surviving `run.log`
timings. `eval.json` is one evaluation of all 246 checkpoints.

Primary metric: **sliced 2-Wasserstein** (512 projections, 4 096 generated vs a fresh 4 096-point true sample). Floors:
**0.0797** (8-Gaussians), **0.0110** (two-moons), **0.0373** (mixture d = 8), **0.0131** (mixture d = 32). Exact 1 024-point
W2 is still in `eval.json` but is no longer read: v1 showed it can fall below its own floor.

**Sliced W2 at 10 000 steps, Euler NFE 1 / 2 / 5 / 10 / 50:**

| target | arm | NFE 1 | NFE 2 | NFE 5 | NFE 10 | NFE 50 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 8-Gaussians | independent | 1.292 | 0.450 | 0.154 | 0.130 | 0.127 |
| 8-Gaussians | fixed_random | 1.271 | 0.429 | 0.146 | 0.120 | 0.117 |
| 8-Gaussians | `ot_pool` B256 | 0.156 | 0.126 | 0.113 | 0.111 | 0.110 |
| 8-Gaussians | `global_ot` | **0.121** | **0.105** | **0.100** | **0.098** | **0.097** |
| two-moons | independent | 0.530 | 0.355 | 0.130 | 0.074 | 0.052 |
| two-moons | fixed_random | 0.529 | 0.346 | 0.131 | 0.076 | 0.054 |
| two-moons | `ot_pool` B256 | 0.080 | 0.045 | 0.039 | 0.038 | 0.038 |
| two-moons | `global_ot` | **0.044** | **0.028** | **0.025** | **0.025** | **0.025** |
| mixture d=8 | independent | 0.497 | 0.264 | 0.132 | 0.085 | 0.064 |
| mixture d=8 | fixed_random | 0.415 | 0.243 | 0.119 | 0.080 | 0.065 |
| mixture d=8 | `ot_pool` B256 | 0.140 | 0.085 | 0.066 | 0.062 | 0.060 |
| mixture d=8 | `global_ot` | **0.121** | **0.077** | **0.062** | **0.060** | **0.058** |
| mixture d=32 | independent | 0.143 | 0.131 | 0.095 | **0.068** | **0.048** |
| mixture d=32 | fixed_random | 0.117 | 0.124 | 0.096 | 0.069 | 0.051 |
| mixture d=32 | `ot_pool` B256 | 0.073 | 0.080 | 0.080 | 0.078 | 0.074 |
| mixture d=32 | `global_ot` | **0.072** | **0.076** | **0.075** | 0.075 | 0.073 |

**Gain over the control** (`arm − fixed_random`, per seed; negative is better; sd over seeds, sem = sd/√n):

| target | NFE | B256 | global |
| --- | ---: | --- | --- |
| 8-Gaussians | 5 | −0.033±0.015 (sem 0.005) | **−0.047±0.023 (sem 0.007)** |
| 8-Gaussians | 10 | −0.009±0.016 | −0.022±0.020 |
| two-moons | 5 | −0.091±0.017 (sem 0.005) | **−0.106±0.013 (sem 0.004)** |
| two-moons | 10 | −0.038±0.014 | −0.052±0.010 |
| mixture d=8 | 5 | −0.053±0.009 | **−0.057±0.008** |
| mixture d=8 | 10 | −0.018±0.009 | −0.020±0.007 |
| mixture d=32 | 5 | −0.016±0.005 | **−0.021±0.004** |
| mixture d=32 | 10 | +0.009±0.005 | **+0.006±0.004 (worse)** |

**Flow straightness** (displacement / path length, 1 = straight) and mode diagnostics at Euler 50:

| target | independent | fixed_random | B256 | global |
| --- | ---: | ---: | ---: | ---: |
| 8-Gaussians | 0.645±0.012 | 0.652±0.015 | 0.970±0.020 | **0.980±0.014** |
| two-moons | 0.764±0.021 | 0.762±0.017 | 0.995±0.001 | **0.995±0.002** |
| mixture d=8 | 0.843±0.022 | 0.845±0.021 | 0.987±0.002 | **0.989±0.002** |
| mixture d=32 | 0.965±0.005 | 0.965±0.004 | 0.992±0.002 | **0.992±0.002** |

Mode metrics at Euler 50 (mixtures only; mode-weight L1, per-mode centre error, covariance error):
8-Gaussians independent 0.077 / 0.060 / 0.106, global 0.059 / 0.041 / 0.107; d = 8 independent 0.073 / 0.233 / 0.327,
global 0.062 / 0.196 / 0.189; d = 32 independent 0.072 / 0.331 / 0.569, global 0.059 / 0.511 / 0.528.

**Equal wall-clock** (plan + training; sliced W2 at Euler 10, best run within the budget):

| budget | target | independent | control | global |
| ---: | --- | ---: | ---: | ---: |
| 40 s | 8-Gaussians | 0.125 | 0.111 | **0.066** (36.6 s) |
| 40 s | two-moons | 0.081 | 0.068 | **0.024** (37.1 s) |
| 40 s | mixture d=8 | 0.084 | 0.076 | not within budget (60.6 s) |
| 40 s | mixture d=32 | **0.068** | 0.074 | 0.077 (38.8 s) |
| 120 s | mixture d=8 | 0.084 | 0.076 | **0.053** (60.6 s) |

Plan cost: `global_ot` 28.5–50.6 s per seed (one 8 192×8 192 POT solve); `ot_pool_B256` 73–84 s on 2-D and
167–246 s at d = 32 for 10 000 steps; `ot_pool_B1024` 89–106 s for 1 000 steps. Training itself is 8–12 s (10 000) and
~1 s (1 000).

**Readings.**
1. **The v1 statement "no NFE effect" is withdrawn.** It compared Euler 10 vs 50, both in the accurate regime. At
   **NFE 1–2** the coupling arms are 2–10× lower than the control on every target, and the learned flows are much
   straighter (0.97–0.995 vs 0.65–0.85). This is the mechanism the v1 summary missed.
2. **At high NFE the advantage shrinks.** It survives on two-moons (0.025 vs 0.054 at NFE 50) and is small on
   8-Gaussians and d = 8; on **d = 32 it reverses** (global 0.073 vs independent 0.048 at NFE 50, and +0.006 at
   NFE 10 in the paired gain). The high-dimensional instance is **not** a clean win.
3. **d = 8 behaves like 2-D** (global better at every NFE, and covariance error 0.189 vs 0.327), while **d = 32**
   trades a better mode weighting (L1 0.059 vs 0.072) for a worse per-mode centre error (0.511 vs 0.331).
4. **The fixed-noise control is again indistinguishable from independent** at every NFE and target, so the gains are
   the coupling, not the finite noise set. That control result is established in 2-D and d = 8/32 at N = 8 192; it is
   not evidence for much larger N or for image data.
5. **The global plan is the best and cheapest OT arm.** One 28–50 s solve beats per-step plans that cost 73–246 s, and
   at 40 s of total budget it wins on 2-D and d = 8; it loses on d = 32, and it is not cheaper than the independent
   arm for a single training run (the plan must be paid once and reused to amortise).
6. **What this means for a fast solver.** On this toy the case for HELLO is not "make minibatch OT cheaper" — the
   global plan already dominates that. It is "make **dataset-level** plans possible when the N×N cost matrix no longer
   fits" (FFHQ-scale N is ~70 k, i.e. ~5×10⁹ entries), and there the fixed-noise control would need re-examination.

**Superseded v1 numbers.** The exact-W2 table (floor 0.3142 / 0.0779) with models below the floor, and the sentence
"No NFE effect", are wrong or superseded and must not be quoted. `coupling_v1` records are kept for provenance only.
