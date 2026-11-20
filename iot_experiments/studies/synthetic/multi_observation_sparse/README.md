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
python -m iot_experiments.studies.synthetic.multi_observation_sparse.run --mode correctness   # ~1 s
python -m iot_experiments.studies.synthetic.multi_observation_sparse.run --mode inner         # ~75 s
python -m iot_experiments.studies.synthetic.multi_observation_sparse.run --mode inner --prune-slack 1e-3
python -m iot_experiments.studies.synthetic.multi_observation_sparse.run --mode benchmark --iterations 10 --max-seconds 45
python -m iot_experiments.studies.synthetic.multi_observation_sparse.run --mode sanity
python -m iot_experiments.studies.synthetic.multi_observation_sparse.diagnose --mode ramp \
    --n 40 --seeds 0 1 2 3 4 5 6 7 8 9 --ks 10 40 80 --ls 5 20 --iterations 40 --tag ramp3   # ~4.5 min
python -m iot_experiments.studies.synthetic.multi_observation_sparse.analyse_ramp                    # the table
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

**Ramp (hinge) basis** (`K ∈ {10,40,80}`, `L ∈ {5,20}`, `n=40`, 10 seeds, 40-iteration cap; records
`out/multi_observation_sparse/diagnose/ramp3/`, table from `analyse_ramp.py`): the joint fit's held-out agreement is
0.887–0.995 and averaging single fits gives 0.912–0.960; `cost_rel` is 0.140–0.193 for the joint fit and
0.096–0.157 for averaging. Going from `L=5` to `L=20` lowers `‖θ−θ*‖₁` by only 7–16 %, while the error grows with
`K` (0.61 → 1.29 → 1.54 at `L=5`). The joint fit keeps its weight on ramps that carry observed mass (0.000–0.070 on
zero-mass ramps); averaging puts 27–42 % of its weight on ramps that are zero on every observed edge. Full table and
the correction of an earlier wrong claim about monotone bases: `notes/MULTI_OBSERVATION_SPARSE_BUNDLE.md`,
Addendum 2.

## Files

`run.py` (four modes), `reference.py` (dense joint LP + dense FW), `diagnose.py` (degeneracy / fix / ramp modes),
`analyse_ramp.py` (the ramp table), `../../../../src/iot/sparse/{features,sparse_ot,bundle_multi}.py`,
tests `tests/iot_sparse_ot_test.py`, `tests/iot_bundle_multi_test.py`.
