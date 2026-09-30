# PROTOCOL — inverse-OT benchmark (shared)

This document is the contract for the shared benchmark in `iot_experiments/benchmark/`
and for the gradient (entropic-θ) runner. The other agent can read it to produce
comparable exact-method checkpoints. Nothing here edits the exact method.

## 0. Ownership and read-only boundary

- The benchmark package and the gradient runner are new files under
  `iot_experiments/benchmark/` plus the warm-start helper added to
  `src/iot/entropic.py` (`sinkhorn_plan_with_potentials`; `sinkhorn_plan` behaviour is
  unchanged).
- `src/iot/{loop,master,oracle,validation}.py` and their tests belong to the exact-method
  agent. The benchmark **imports** `solve_inverse_ot` and `dual_optimality_certificate`
  but never edits them.
- HELLO-OT source (`src/hello_ot/**`, `setup.py`, `MANIFEST.in`, …) is untouched.

## 1. Canonical generator

`iot_experiments/benchmark/instances.py::build_instance` wraps the exact-side
`build_pot_problem` semantics exactly:

1. `rng = np.random.default_rng(seed)`
2. `theta_star = SimplexTheta(k).sample(rng)` — `theta_star` is drawn **first**
3. `source = rng.normal((n, k))`, then `target = rng.normal((n, k))`
4. optional `mixing_strength` (default `0.0`; no mixed arm is run yet)
5. `cost = dense_cost_matrix(DiagonalWeightFeatures(source, target), theta_star)`
6. `plan = ot.emd(uniform, uniform, cost, numItermax=2_000_000)`

Fixed choices: `K = 8`, `numItermax = 2_000_000`. Evaluation seeds `0–9`; tuning seeds
`1000–1004`. `e3_scaling.py`'s seed convention (`seed + 1000*repeat + size`) is
**historical** and is not reproduced here.

The wrapper is byte-identical to
`iot_experiments/scratch/common.build_problem(..., num_itermax=2_000_000)` at seeds `0–9`;
this is asserted by `tests/iot_benchmark_test.py`.

### 1.1 Ground-truth certificate

The ground-truth plan (clean plan, or the clean plan re-solved on a noisy instance's
evaluation universe — see §3) must pass
`iot.validation.dual_optimality_certificate(problem, theta_star, num_itermax=2_000_000)`.
`BenchmarkInstance.__post_init__` raises `BenchmarkError` if it does not certify. The
default certificate tolerances (`feasibility_tol=1e-7`, `tightness_tol=1e-7`) are used
unchanged; if a size cannot certify, the run stops with an error and is reported rather
than silently loosened.

## 2. Clean control

`build_instance(..., noise_multiplier=None)` gives `pi_hat = pi*`. This is the control arm
for both regimes.

## 3. Noisy-plan constructor

`build_noisy_instance(n, k, seed, noise_multiplier ∈ {5, 20, 100})`:

- `M = noise_multiplier * n`
- sample `M` i.i.d. pairs from the flattened true plan `pi* / sum(pi*)` using a dedicated
  `np.random.default_rng(seed)` (so the noise is reproducible independent of mixing),
- `pi_hat = counts / M`, and the observed marginals are `pi_hat`'s own row/column sums,
- the control is the clean plan of §2.

### 3.1 Zero-marginal repair (disclosed)

`pi*` is generically a permutation, so at `M = 5n` a few rows/columns receive zero samples
(measured at `n=200`: 1–3 of 200). Both methods require strictly positive marginals
(`InverseOTProblem` and `Sinkhorn` both enforce it), and with a zero marginal the observed
gap `G` is vacuous. We therefore **drop rows/columns whose empirical marginal is zero**
and record `zero_rows_dropped` / `zero_cols_dropped` and the effective `n_source`/`n_target`.
No epsilon is added and no observation is thresholded; the plan values stay exactly
`counts / M` on the retained sub-problem.

Consequence: at `M = 5n` the noisy arm is evaluated on a slightly smaller universe
(e.g. `n' ≈ 197–199` at `n = 200`); this is a documented selection effect and is reported.
`true_plan` for a noisy instance is the clean OT plan re-solved **on the retained points**
with uniform marginals, so `R` is computed against a consistent true plan on the same
universe.

## 4. Methods

| label | what it is |
| --- | --- |
| `exact-constraint-generation` | the other agent's `solve_inverse_ot` (external; not edited here) |
| `gradient-entropic-theta` | `iot_experiments/benchmark/runner.py` |

The gradient runner minimises the entropic θ-gap
`L_ε(θ) = <C_θ, pi_hat> − <C_θ, P_ε(θ)>` by the envelope gradient
`∇L_ε(θ)_k = <Φ_k, pi_hat − P_ε(θ)>`.

## 5. Optimiser and θ parametrisation

- **Optimiser: AdamW, `weight_decay = 0`.** With decoupled weight decay set to zero the
  update is algebraically identical to the historical Adam used by
  `fit_theta_sinkhorn`, so the historical Adam columns remain comparable. Label:
  `AdamW, wd=0`. No other optimiser is run.
- **θ is the constrained variable directly** (simplex vector for `SimplexTheta`, flattened
  PSD metric for `BilinearPSDTheta`), **projected back onto Θ after every step**
  (`project_simplex` / `project_psd`). Initial θ is uniform (`1/K`).
- **No logits and no weight decay**, therefore no implicit shrinkage of θ toward uniform.
  This is a deliberate correction to the earlier instruction.

## 6. Epsilon schedule and Sinkhorn

- ε anneals **all the way to the floor `1e-3`** by geometric `epsilon_decay`; the runner
  chooses `num_outer` large enough that the floor is reached, and records
  `epsilon_floor_reached` and the full ε trace.
- Sinkhorn potentials are **warm-started** between outer steps
  (`sinkhorn_plan_with_potentials`), so lowering ε starts from the previous solution.
- The final Sinkhorn **marginal error** (`max` row/column mass deviation) is logged per
  checkpoint and at the end. If Sinkhorn has not converged at the floor, this is stated; it
  is not hidden behind a small `G`.
- `sinkhorn_iterations` and `sinkhorn_tolerance` are part of the frozen config.

Historical note: the earlier `COMPARISON_REPORT.md` stopped at ε ≈ 0.1, so its gaps mostly
reflect smoothing bias. Those numbers are **historical** (different instances, `K=4`,
default POT cap) and are never mixed into the new tables.

## 7. Checkpoints

At least **10 theta snapshots per run**, time-stamped (`elapsed_seconds` since run start),
written by `iot_experiments/benchmark/checkpoint.py`. Schema (`iot-benchmark-run-v1`):

- instance: `instance_id`, `seed`, `n`, `n_features`, `noise_multiplier` (or `null`),
  `noise_pairs` (or `null`), `mixing_strength`, `num_itermax`, effective `n_source`/`n_target`,
  `zero_rows_dropped`, `zero_cols_dropped`;
- run: `method`, `theta_parametrisation`, `optimiser`, `weight_decay`, `learning_rate`,
  `epsilon_start`, `epsilon_decay`, `epsilon_floor`, `num_outer`, `sinkhorn_iterations`,
  `hardware`, `threads`, `converged`, `peak_rss_mb`, `wall_time_seconds`,
  `final_sinkhorn_marginal_error`, `epsilon_floor_reached`, `terminal_theta`;
- per checkpoint: `iteration`, `elapsed_seconds`, `theta`, `epsilon`,
  `sinkhorn_marginal_error`, `entropic_gap`.

## 8. Metrics (offline)

`iot_experiments/benchmark/evaluate.py`; POT only, never inside a timed section. With
`sigma_star = std(entries of C_theta*)`:

- `G = (<C_theta_hat, pi_hat> − min_P <C_theta_hat, P>) / sigma_star`, `P` over `pi_hat`'s
  own marginals (the observed-plan gap).
- `R = (<C_theta*, pi(theta_hat)> − <C_theta*, pi*>) / sigma_star`, over the true marginals
  (the regret under the true cost of the plan induced by `theta_hat`).

`R` is the metric to trust on noisy plans, because θ is not identifiable there; θ matrices
are never compared directly. Both `G` and `R` are reported raw as well.

Reference rows: `theta_hat = uniform` and `theta_hat = theta*`. **If the gradient method
cannot beat uniform on `R` for noisy plans, that is a finding and is reported plainly.**
There are **no historical noisy-plan numbers**; regime R1 is being measured here for the
first time.

## 9. Tuning discipline

Hyperparameters (learning rate, ε schedule, `num_outer`, `sinkhorn_iterations`) are set
**once** on tuning seeds `1000–1004` at `n = 200` and `n = 800`, then frozen and logged.
No per-instance tuning and no tuning after seeing evaluation results.

## 10. Timings, threads, hardware, memory

- Run the gradient runner on **CPU** first, same machine and thread count as the exact
  method. Pin `OMP_NUM_THREADS = MKL_NUM_THREADS = OPENBLAS_NUM_THREADS = 8` for both
  agents and record it.
- Host: Intel i7-13620H, 16 logical CPUs, 16 GB RAM (~10 GiB free). GPU (RTX 4060 Laptop)
  exists but is a **separate labelled arm** only if the CPU `n=3200` run cannot finish; a
  port must first match the NumPy runner on `n=200` seeds `0–2` to `1e-6` after the same
  steps, and the CPU/GPU asymmetry is stated, never hidden.
- **Peak RSS** is measured as the real process RSS of each run (`resource.getrusage`), each
  run executed in its own subprocess — not `tracemalloc`. If `n=3200` risks swapping, the
  run stops and reports.
- `.cpu_lock`: create `iot_experiments/benchmark/.cpu_lock` before any heavy run and remove
  it afterwards; if it exists, wait. Timings taken while both agents run are invalid.

## 11. Decision rules (fixed)

- **R2 (large n):** at `n = 1600` and `n = 3200`, normalised `G ≤ 1e-3` within the exact
  method's median wall-clock, or reached where the exact method fails to converge.
- **R1 (noisy):** `R` clearly below the uniform reference on most seeds (~3× lower), and
  `R` falls as `M` grows.
- If neither holds, state plainly that it is not an interesting alternative.

Raw numbers (median and IQR over seeds) are reported regardless of the outcome.

## 12. Known biases / confounds

- Historical `COMPARISON_REPORT.md` used different instances (`K=4`, mixing 0.8, default
  POT cap) and stopped at ε ≈ 0.1; it is not comparable and is not mixed in.
- The exact method sees `pi_hat` through its support (and its marginals); on sampled plans
  whose support equals the true permutation support it can still certify even though the
  counts are noisy. This is a property of the method, reported as-is.
- `M = 5n` zero-marginal repair changes the universe size slightly (§3.1).
- θ is not identifiable on discrete data; `R`/`G` (not θ distance) are authoritative.
- Thread pinning fixes CPU contention but not turbo/thermal variability; medians/IQR over
  seeds are used.
