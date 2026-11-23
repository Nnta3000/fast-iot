# multi_observation_sparse — one θ, many sparse plans, linear memory

**Question.** Can the shared-θ minimum-regret objective `F(θ) = Σ_l [⟨c(θ),π̂_l⟩ − OT_l(c(θ))]` be solved with memory
linear in `n + m` when the observed plans are sparse, and does the answer match the dense formulation?

**Claim.** Scalability only: same objective, same answer up to a stated tolerance, linear memory. Time stays quadratic
(pricing visits every pair), and the planted-window part is a sanity check, not a result about quality. No claim of
beating the state of the art. See `notes/MULTI_OBSERVATION_SPARSE_BUNDLE.md` for the full note, including "what is
NOT claimed".

## Protocol

- Synthetic points in the unit square; features `s, s², log1p(s), s³` computed on the fly (`s = ||x−y||`), each column
  divided by a subsample-RMS scalar. **No constant column and no same-zone column** (both make the regret trivially
  small — measured in the dense studies).
- Observed plans: exact OT plans at a planted θ* (sparse by construction, ≤ `m+n−1` non-zeros).
- Estimator: Kelley cuts in θ from exact per-window OT solves; the inner OT is solved on a growing edge set with
  blockwise pricing and its own certificate. Master and inner LP use `scipy` HiGHS, **labelled a reference**.
- Modes: `correctness` (vs a dense joint LP and a dense Frank–Wolfe), `inner` (edge growth and time per size),
  `benchmark` (dense vs sparse, separate processes), `sanity` (planted windows + an out-of-family control).

## Run

```bash
python -m iot_experiments.studies.iot.synthetic.multi_observation_sparse.run --mode correctness   # ~1 s
python -m iot_experiments.studies.iot.synthetic.multi_observation_sparse.run --mode inner         # ~75 s
python -m iot_experiments.studies.iot.synthetic.multi_observation_sparse.run --mode inner --prune-slack 1e-3
python -m iot_experiments.studies.iot.synthetic.multi_observation_sparse.run --mode benchmark --iterations 10 --max-seconds 45
python -m iot_experiments.studies.iot.synthetic.multi_observation_sparse.run --mode sanity
python -m iot_experiments.studies.iot.synthetic.multi_observation_sparse.diagnose --mode ramp \
    --n 40 --seeds 0 1 2 3 4 5 6 7 8 9 --ks 10 40 80 --ls 5 20 --iterations 40 --tag ramp4   # ~12 min
python -m iot_experiments.studies.iot.synthetic.multi_observation_sparse.analyse_ramp --tag ramp4        # the table
```

## Results (all in `iot_experiments/out/multi_observation_sparse/`)

**Correctness** (`n=30, L=5`): bundle `F = 4.27e-08`, bracket `[-9.19e-07, 4.27e-08]`, gap `9.6e-07`;
dense joint LP `F* = -2.3e-16`; dense FW `1.05e-05`; plan agreement 0.9994.

**Inner scaling** (one cold solve, `γ=2`):

| n | peak `|E|` | peak `/(n+m)` | kept (prune 1e-3) | sweeps | LP s | pricing s | total s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 200 | 4 613 | 11.5 | 1.09× | 10 | 0.30 | 0.05 | 0.36 |
| 800 | 26 138 | 16.3 | 1.33× | 12 | 3.34 | 0.96 | 4.32 |
| 3200 | 139 310 | 21.8 | 1.58× | 16 | 45.8 | 22.0 | 66.1 |

The peak edge count grows faster than `n+m` (the sweep count grows), so the memory constant is **not** yet bounded;
pruning shrinks the *warm* set but happens after the peak. The inner LP is 67 % of the time at `n=3200`.

**Benchmark** (`L=3`; dense arm capped at 10 / 5 / 1 iterations, so `F` is not a quality comparison):

| n | dense FW: F / time / peak | bundle: F / time / peak |
| ---: | --- | --- |
| 200 | 2.33e-06 / 0.4 s / 628 MB | 1.40e-07 / 0.8 s / 629 MB |
| 800 | 6.86e-05 / 3.4 s / 724 MB | 4.08e-06 / 8.3 s / 709 MB |
| 3200 | 6.98e-05 / 13.0 s / 1801 MB | 6.98e-05 / 24.9 s / 1570 MB |

The bundle is slower at every size (POT's dense EMD is C code; the bundle pays a certified sparse LP plus pricing per
window). Peak-memory differences are small at these sizes because the measurement includes ~600 MB of library
baseline and the generator's own dense cost. Time, not memory, is what stops the method: a cold solve at `n=3200`
takes 66 s and grows like `n²`, so `n=12800` is ≈17 min per window per outer iteration.

**Sanity** (planted, `n=40`, `L ∈ {1,5,20}`, two seeds): `F ≈ 0` and plan agreement `1.0000` in every planted run;
the out-of-family control gives `F = 0.04–0.71` and agreement `0.24–0.27`. θ is not recovered (`θ_l1` 0.2–0.8) while
the plans are: the identifiability cone, as expected. Not evidence about quality.

**Ramp basis, discriminating instances** (`K ∈ {10,40,80}`, `L ∈ {5,20}`, `n=40`, 10 seeds; records
`out/multi_observation_sparse/diagnose/ramp4/`, table by `analyse_ramp.py --tag ramp4`): the basis is first-difference
hinges on quantile thresholds (chosen on conditioning: min correlation eigenvalue 1.75e-02 at `K=20` versus 5.15e-05
for nested hinges), and `theta*` is rejection-sampled so that a fresh instance's **neutral** plan reproduces at most
half of it. Neutral held-out agreement is 0.52–0.70 (the earlier hinge run had 0.87–0.93, which is why its gains were
unreadable). Against neutral: at `K=10` the joint fit goes from **−0.128** (`L=5`) to **+0.158** (`L=20`), while
averaging stays at +0.130/+0.115; at `K≥40` the joint fit hits the 40-iteration cap in 10/10 seeds with a gap of
1e-2–7e-2, so those rows are **not** a certified optimum and cannot compare the arms. Averaging with versus without
the windows that are already optimal at uniform θ changes the result at `K=10` (0.828 vs 0.715 at `L=5`), so the two
are reported separately. Full table, basis/rule diagnostics and defects:
`notes/MULTI_OBSERVATION_SPARSE_BUNDLE.md`, Addendum 2c. θ error is a diagnostic only.

**Arms grid G3** (`K ∈ {10,20}`, `L ∈ {1,5,10}`, `p ∈ {0,0.2}`, uniform marginals, no control, 5 seeds; records
`arms/g3/`, `arms/g3_big/`, `arms/g3_deep/`, table by `analyse_arms.py`): `theta*` is rejection-sampled on the bump
basis (neutral held-out agreement 0.040 at `K=10`, 0.015 at `K=20`). Gains over neutral are large for every arm
(+0.13 to +0.94), `A = B = C` exactly at `L = 1` (sanity), and `A_single` is below both `B` and `C` at `L ≥ 5`. The
joint fit is **capped at 40 iterations in 9 of 12 cells** (gap 2.4e-03–1.2e-01), so its column is not a certified
optimum; the C−B contrast is therefore unreliable except where checked. Re-running the one cell `K10 L5 p0` with 200
iterations lets C converge (67.6 iterations, gap ≈ 0): C 0.960 vs B 0.855, `C−B = +0.105 ± 0.182`, `cost_rel` 0.223 vs
0.339. The bounded big-plan arm (`n′ = 40/80/200`) appears to collapse at `K=20` but is capped 5/5 there with gap
5e-03–1.8e-02; at `K20 L2 p0` the training objective is 8.6e-06, i.e. one plan does not identify θ. Details:
`notes/MULTI_OBSERVATION_SPARSE_BUNDLE.md`, Addendum 3.

## Files

`run.py` (four modes), `reference.py` (dense joint LP + dense FW), `diagnose.py` (degeneracy / fix / ramp modes),
`analyse_ramp.py` (the ramp table), `analyse_arms.py` (the arms table), `../../../../src/iot/sparse/{features,sparse_ot,bundle_multi}.py`,
tests `tests/iot_sparse_ot_test.py`, `tests/iot_bundle_multi_test.py`.
