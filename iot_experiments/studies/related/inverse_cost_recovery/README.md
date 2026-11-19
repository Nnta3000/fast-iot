# inverse_cost_recovery — Ma et al.'s discrete cost-recovery benchmark, estimated with the relaxed arm

**Question.** Ma, Sun, Ye, Zha & Zhou (2021, §5.1) generate an entropic plan from a known cost and score the
inferred cost by its relative Frobenius error. What error does this repository's **relaxed** (regret) master reach
on the same generated data, and how does it move with the entropy weight?

The benchmark is theirs; the estimator is ours. Nothing here is a number from their paper except the reference
value quoted in `PROVENANCE` (`run.py`).

## Protocol

Exactly the paper's data-generating process:

- `n = 100`; truth `c*_ij = |(i − j)/n|^p`, `p ∈ {0.5, 1, 2, 3}`;
- `20` marginal pairs `μ, ν ~ Dirichlet(1, …, 1)` on the `n`-simplex (the paper says "20 random marginal pairs"
  without fixing the distribution; this is the choice, and the same 20 draws are reused in every `(p, ε)` cell);
- observed plan `π̂ = sinkhorn_log(μ, ν, c*, ε)`, `ε ∈ {10, 1, 0.1, 0.01}`, untouched — no thresholding, no
  renormalisation;
- estimator: `argmin_θ R(θ)` with `R(θ) = ⟨c(θ), π̂⟩ − OT(c(θ))`, one relaxed-master solve over all `m·n` edges
  (`--master-backend auto` → `cupdlpx` when a CUDA device is usable, else the exact CPU simplex `highs`).

Three choices the paper does not have to make, because its unknown is a free `n × n` matrix:

| choice | what we do | why |
| --- | --- | --- |
| hypothesis class | `--family`: `p_basis` (`t^q`, `t = |i−j|/n`), `asym_basis` (`|i/n − a·j/n|^q`, `a ∈ {1,2}`), `free_dense` (`K = n²`), `hat9` (hats/ramp of `t`) | our cost is `c = Σ θ_k Φ_k` with `θ` on the simplex, so the class is a feature family |
| gauge | every feature scaled to unit Frobenius norm | `R` is positively homogeneous in the cost, so a family whose members differ in magnitude is scored on magnitude too. A positive rescaling of a cost leaves the OT plan unchanged, so this removes exactly the non-identifiable part; the recovered scale is reported as `best_scale` |
| metrics | `cost_rel` (raw), `cost_rel_scaled` (after the optimal positive rescaling), `plan_rel`, `plan_cost_rel` | `plan_rel` is a Frobenius distance between plans; costs of the form `|i−j|^p` have ties, so the LP has an optimal face and POT may return a different vertex for a rescaled cost. `plan_cost_rel` is the excess cost of the induced plan under the *true* cost, relative to the true optimum: 0 exactly when the induced plan is optimal, independent of which vertex was returned |

## Measured so far

`n = 100`, 20 instances per cell, 320 rows per family, CPU (`highs`), ~6 minutes per family. Every row carries `θ`
itself in `out/<study>/records.json`. Cells are `(p, ε)`; `cost_rel_scaled` is the paper-comparable number.

**`p_basis`** (members `t^q` for the four grid powers; the truth is one of them):

| generating `p` | ε = 0.01 | ε = 0.1 | ε = 1 | ε = 10 | `θ` returned |
| ---: | ---: | ---: | ---: | ---: | --- |
| 0.5 | 0.090 | 0.602 | 0.665 | 0.655 | `p = 3` vertex in 18–20 of 20 draws (except ε = 0.01: `p = 0.5` in 2) |
| 1 | 0.308 | 0.503 | 0.499 | 0.485 | `p = 3` vertex in 18–19 of 20 (ε = 0.01: interior point in 2) |
| 2 | 0.199 | 0.218 | 0.215 | 0.208 | `p = 3` vertex in 18–20 of 20 |
| 3 | **0.000** | **0.000** | 0.006 | 0.017 | `p = 3` vertex in 18–20 of 20 |

**`asym_basis`** (members `|i/n − a·j/n|^q`, `a ∈ {1,2}`, `q ∈ {1,2}`):

| generating `p` | ε = 0.01 | ε = 0.1 | ε = 1 | ε = 10 | `θ` returned |
| ---: | ---: | ---: | ---: | ---: | --- |
| 0.5 | 0.332 | 0.571 | 0.546 | 0.541 | `(a=2, q=1)` in 8–19 of 20 |
| 1 | 0.323 | 0.605 | 0.584 | 0.581 | `(a=2, q=1)` in 12–19 of 20 |
| 2 | 0.624 | 0.679 | 0.679 | 0.679 | `(a=2, q=1)` in 8–19 of 20 |
| 3 | 0.747 | 0.751 | 0.752 | 0.753 | `(a=2, q=1)` in 13–19 of 20 |

### What these numbers show

1. **The master returns the global minimiser of `R`.** For `asym_basis`, `p = 1`, `ε = 1`, `n = 60`, the LP's `θ`
   matched a brute-force grid search over the 4-simplex to `1e-16` (`R = 1.7768632887174961e-03` in both), and a
   direct recomputation of `R` at the returned `θ` agrees to `1e-16`. The selected member is a property of the
   criterion and the family, not of the solver.
2. **In 29 of the 32 cells the same member is returned regardless of the generating one** (`p = 3` for
   `p_basis`, `(a=2, q=1)` for `asym_basis`). The observed plan does not decide which member is reported in those
   cells.
3. **Where the returned member is the generating one, recovery is exact**: `cost_rel_scaled` is `0.000` at
   `p = 3, ε = 0.01` and `0.1`, and rises monotonically with `ε` (0.006 at `ε = 1`, 0.017 at `ε = 10`). The arm
   recovers a cost exactly when `R`'s optimum is the truth.
4. **Where it is not the generating one, the cost can still be plan-equivalent.** At `p = 2` the recovered cost
   induces a plan identical to the truth's (`plan_rel ≈ 0` at `ε ≤ 0.1`) while `cost_rel_scaled = 0.199–0.218`; at
   `p = 1` the recovered cost's plan is exactly optimal under the true cost (`plan_cost_rel = 0.000` in all four
   `ε`) while `cost_rel_scaled ≈ 0.485–0.503`. The plan is preserved and the cost is not.
5. **`plan_rel` alone is not a safe reading.** For costs of the form `|i−j|^p` the LP has an optimal face, so two
   plans can both be optimal and still differ as matrices (`plan_rel ≈ 1.1` with `plan_cost_rel = 0.000` at
   `p = 1`). That is why both are reported; `plan_cost_rel` is the one that says whether a plan is optimal.

The measured statement, scoped to this benchmark and these families: minimising the exact-OT regret selects a
member of the family that is fixed by the cost's shape, not by the generating member; cost recovery is exact only
in the cells where that fixed member *is* the truth; and the recovered cost can reproduce the plan while being
wrong as a cost. The paper's own §5.3 reports the same phenomenon for the free-cost case (recovered cost differs
from the truth; plan error 0.026).

## Running it

```bash
# 20 instances, all four p and eps; about 2 minutes per family on CPU (0.2-0.5 s per fit at n=100)
python -m iot_experiments.studies.related.inverse_cost_recovery.run --family p_basis --n 100 --instances 20
python -m iot_experiments.studies.related.inverse_cost_recovery.run --family asym_basis --n 100 --instances 20

# free cost matrix: K = n^2 = 10 000 features, one (m*n, K) block = 800 MB; --allow-heavy required
python -m iot_experiments.studies.related.inverse_cost_recovery.run --family free_dense --n 100 --instances 5 --allow-heavy
```

Records: `out/related_cost_recovery_*/records.json` with `run.log` and `preflight.json`. Each row carries `θ`
itself, so a later analysis can re-score without re-solving.

## What is not in this study

- `free_dense` has been implemented but not run: the `(m·n, K) = 10⁴ × 10⁴` feature block is 800 MB and the LP
  assembly scales with it, so it is a server run.
- `hat9` has no member equal to the truth, so its error floor is the basis's approximation error; it is implemented
  and not yet run.
- The reported reference point (the paper's `≈1e-4`) comes from their Algorithm 1 with a free cost matrix and a
  likelihood objective; this study's estimator is the regret master with a feature family. The protocol section
  of `notes/RELATED_WORK_AND_COMPARABILITY.md` states the differences; the numbers above are what this estimator
  returns on this data.
