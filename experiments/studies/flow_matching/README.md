# flow_matching — does the coupling pool size matter?

**Status: measured (2026-11-23).** This is a study of **scalable exact forward OT** with HELLO-OT as the eventual
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
| data | 8-Gaussians (8 modes, radius 2, σ 0.1) and two-moons, 8 192 training points, 2 048 held out |
| source | `N(0, I₂)` |
| model | MLP 2→128→128→2, SiLU, four Fourier time features; Adam 1e-3 |
| loss | CFM: `‖v_t(x_t) − (x₁ − x₀)‖²`, `x_t = (1−t)x₀ + t x₁` |
| training batch | **256, fixed**; only the coupling pool varies |
| arms | `independent` (fresh noise, i.i.d. partner); `fixed_random` (fixed 8 192 noise points + fixed random permutation — the control for `global_ot`); `ot_pool` at `B = 256` and `B = 1024` (fresh pool every step, exact POT EMD); `global_ot` (one exact plan between the fixed noise set and the whole training set, computed once) |
| seeds | 10 for `independent`, `fixed_random`, `ot_pool_B256`, `global_ot`; 3 for `ot_pool_B1024` |
| budgets | 1 000 and 10 000 steps for the cheap arms; 1 000 for `B = 1024` (POT costs 0.082 s/step there) |
| samplers | Euler NFE ∈ {1, 2, 5, 10, 50}; RK4 NFE ∈ {4, 8, 20, 48} |
| metric | exact 2-Wasserstein (squared Euclidean, uniform masses) between 1 024 generated and 1 024 held-out points |
| floor | W2 between a **fresh sample of the true distribution** and the held-out set, averaged over 3 draws; differences are read against it |
| coupling cost | plan time and training time recorded; both equal-step and equal-wall-clock are reportable |

**Why the control.** `global_ot` fixes 8 192 noise points for the whole run; the other arms draw fresh noise every
step. A model can memorise a finite fixed noise set, so `global_ot` vs `independent` mixes coupling quality with that
finite-noise effect. `fixed_random` shares the same fixed noise points and a fixed random permutation, so
`global_ot − fixed_random` isolates the coupling.

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
PYTHONPATH=. .venv/bin/python -m experiments.studies.flow_matching.analyse --tag coupling_v1
```

Records: `experiments/out/flow_matching/coupling_v1/{precheck.json,records.json,w2.csv}`.

## Results

Records: `experiments/out/records/flow_matching_coupling_v1/` (`records.json`, `precheck.json`, `w2.csv`, `run.log`);
copies are written to `experiments/out/flow_matching/coupling_v1/` by the driver. Config: 10 seeds for the cheap arms,
3 for `B = 1024`, 1 000 and 10 000 steps, Euler and RK4 grids as above. `W2` is exact, 1 024 generated vs 1 024
held-out; the floor is **0.3142** (eight-Gaussians) and **0.0779** (two-moons).

**Pre-check** (20 draws, same noise points, pool pairing vs global pairing):

| target | pool | differing partners | MSD local | MSD global | cost gap |
| --- | ---: | ---: | ---: | ---: | ---: |
| eight-Gaussians | 256 | 0.995 | 1.013 | 0.945 | +7.2 % |
| eight-Gaussians | 1024 | 0.973 | 0.953 | 0.941 | +1.2 % |
| two-moons | 256 | 0.991 | 0.834 | 0.796 | +4.8 % |
| two-moons | 1024 | 0.976 | 0.787 | 0.788 | −0.1 % |

A pool plan is a **different plan** almost everywhere even at `B = 1024`, but its cost penalty is small at `B = 1024`
and several percent at `B = 256`.

**W2 at Euler NFE = 10** (mean over seeds; floor in brackets):

| target | steps | independent | fixed_random | `ot_pool` B256 | `ot_pool` B1024 | `global_ot` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| eight-Gaussians [0.314] | 1 000 | 0.437 | 0.423 | 0.409 | 0.404 | **0.391** |
| eight-Gaussians [0.314] | 10 000 | 0.329 | 0.314 | 0.294 | not run | **0.283** |
| two-moons [0.078] | 1 000 | 0.282 | 0.280 | 0.219 | 0.206 | **0.199** |
| two-moons [0.078] | 10 000 | 0.186 | 0.186 | 0.127 | not run | **0.104** |

**Gain over the control** (`arm − fixed_random`, per seed; negative is better), Euler NFE = 10:

| target | steps | B256 | B1024 | global |
| --- | ---: | --- | --- | --- |
| eight-Gaussians | 1 000 | −0.014±0.020 | −0.022±0.033 | **−0.032±0.014** |
| eight-Gaussians | 10 000 | −0.020±0.051 | not run | −0.030±0.049 |
| two-moons | 1 000 | −0.061±0.018 | −0.088±0.007 | **−0.081±0.015** |
| two-moons | 10 000 | −0.059±0.019 | not run | **−0.082±0.021** |

**Readings.**
1. **Two-moons: the coupling helps, and more pool is better.** `global_ot` beats the control by 0.082 ± 0.021 W2 at
   10 000 steps (≈ 4σ), `B256` by 0.059 ± 0.019; `B1024` matches the global gain at 1 000 steps. This is a
   seed-consistent effect, not a single-seed outlier.
2. **Eight-Gaussians is saturated at 10 000 steps.** Every arm is at or below the floor (0.283–0.329 vs 0.314), so the
   coupling difference disappears; the only clear signal is at 1 000 steps (`global_ot` −0.032 ± 0.014, ≈ 2.3σ). Per
   the pre-registered rule this is "the instance is too easy at this budget", **not** "OT couplings do not help".
3. **The fixed-noise confound is not the cause.** `fixed_random` (same fixed 8 192 noise points, fixed random pairing)
   is indistinguishable from `independent` (0.314 vs 0.329 at 10 000 eight-Gaussians; 0.186 vs 0.186 two-moons), so the
   `global_ot` gain comes from the coupling, not from fixing the noise set.
4. **No NFE effect.** Euler 10 vs 50 differs by ≤ 0.01 in every arm; the coupling changes the final model, not the
   number of integration steps needed. The straightness argument is not visible on these 2-D targets.
5. **Coverage** (eight-Gaussians, within 0.3 of a mode, 10 000 steps): independent 0.808, control 0.813, `B256` 0.804,
   `global_ot` 0.839. The ordering is not monotone and the differences are small.

**Wall-clock, measured** (per run, from the records' `plan_seconds` and `train_seconds`): `global_ot` plan 28.3–28.9 s
(one 8 192×8 192 POT solve per seed) against 8–11 s of training for 10 000 steps; `ot_pool_B256` plan 73–84 s for
10 000 steps; `ot_pool_B1024` plan 88–105 s for 1 000 steps against ~1 s of training. On CPU POT the exact minibatch
coupling costs **8× the gradient time at B=256 and ~100× at B=1024**; the global plan is a one-off and amortises over
epochs, the per-step pools do not. This asymmetry is what a GPU solver would change, and it is why the numbers above
are **equal-step**, not equal-wall-clock.

**What this does not show.** No CIFAR/FFHQ claim, no FID, no HELLO timing (POT was used throughout); the paper's
FFHQ number is context, not a result verified here.
