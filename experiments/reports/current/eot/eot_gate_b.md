# Gate B — independent reference report

**Status: CURRENT — the formulation question only.** Answers "does the joint EOT formulation have the
right optimum?" (yes). It does **not** answer which solver carries it: §5 recommends adopting SLSQP as
the working solver, but `eot_solver_compare.md` and `eot_scaling_slsqp.md` later declined that, and this
file's §5 should be read as the first of three statements, not the last.
**Commit:** `6f57f1d` (solver frozen at `53f3211`, git tag `t3-freeze-gate-b` — tags keep their
historical name)
**Driver:** `experiments/studies/iot/real/t1t2/metric_learning/eot_joint_reference.py`
**Instance:** `planting.plant_instance(family="diagonal", n=24, point_dimension=4, seed=0)`
**Sizes:** `m_eff = 24`, `n_eff = 15`, `K = 4`, full grid = 360 edges, `sum a = sum b = 24`
**ε:** `0.5738680483` = `0.5 × std(c(θ_uniform))` where `std = 1.1477360965`
**Constraints:** simplex `Σθ = 1, θ ≥ 0`; gauge `g₀ = 0`
**Record:** `experiments/out/eot_joint_reference/joint_reference.json`
**Tests:** `tests/iot_entropic_joint_reference_test.py` — 7 passed

The independent solver is `src/iot/entropic/joint_reference.py`. It was written from the reduced
objective directly, using `scipy.special.logsumexp` and `scipy.optimize.minimize(SLSQP)`. It does not
import `joint_master` and shares none of its gradient, Hessian-vector product, coordinate reduction,
active set or line search.

Verification of the reference itself (independent of both solvers):

| check | result |
|---|---|
| `Ψ_red` vs explicit `(θ,f,g)` objective at the eliminated `f` | **difference 0.0** |
| source marginal, for arbitrary `g` (scale 0, 1, 10) | `≤ 1e-12` |
| `∇_θ Ψ_red` vs central differences | `3.8e-9` |
| `∂Ψ_red/∂g_j` vs central differences | `4.9e-9` |

## 1. Does the independent joint formulation agree with the nested reference?

**Yes.** Four starting points, none of them inherited from the custom solver's initialisation:

| start | Ψ | θ | status |
|---|---|---|---|
| nested reference θ | 31.7731301675407 | (0.4809721, 0.4966576, 0.0223703, 0) | 0 |
| custom solver's stalled θ | 31.7731301675407 | (0.4809721, 0.4966576, 0.0223703, 0) | 0 |
| interior uniform (0.25 × 4) | 31.7731301675407 | (0.4809722, 0.4966575, 0.0223703, 0) | 0 |
| other feasible (0.7, 0.1, 0.15, 0.05) | 31.7731301675407 | (0.4809721, 0.4966576, 0.0223703, 0) | 0 |

Agreement to **1e-12** across starts, and to the nested value to **1e-12**. The starting point that the
custom solver stops at converges, in a proper solver, to the answer the custom solver misses. That
rules out the step being "too hard from there" — it was the solver.

## 2. Does the independent solution satisfy the simplex KKT conditions?

**Yes.** `simplex_kkt_violation ≤ 1.4e-6` at every start (that residual is the SLSQP convergence
tolerance, not a formulation issue). `θ₃ = 0` exactly; the multiplier on it is `+10.16`, so the
boundary is a genuine optimum, not an artefact. Target marginals:

| quantity | value |
|---|---|
| source marginal error | `3.3e-16` (exact by construction) |
| target marginal error | `7.5e-7` |
| `max_j |∂Ψ/∂g_j|` | at the same scale as the target marginal error |

## 3. Does the face enumeration support the same optimum?

**Yes**, independently, but **my earlier reported numbers were partly wrong and I am correcting them.**

| face | Ψ | excluded multipliers | KKT-optimal |
|---|---|---|---|
| {0,1,2,3} | 31.7731301678 | — | yes |
| **{0,1,2}** | **31.7731301678** | +10.158 on 3 | **yes** |
| {0,1,3} | 31.795049961 | −5.187 on 2 | no |
| {0,1} | 31.795049961 | −1.952 on 2 | no |
| all other 11 faces | ≥ 36.895 | negative | no |

Two corrections to what I reported before:

1. **The value for face `(0,1)` was 31.79505, not 31.81830.** My first version of the independent
   enumeration optimised the *full* problem and then filtered, so SLSQP moved `θ₂` off zero
   (`θ₂ = 0.0224`, with `θ₂ = 10⁻⁸` instead of exactly `0`) and I recorded a point that is not on the
   face. It also listed `(0,1)` above `(0,1,3)` with a *smaller* value, which is impossible, and that
   contradiction is what exposed the bug. With a face-restricted solver (`solve_on_face`, where the
   excluded coordinates are not variables) the face values are consistent: `{0,1}` and `{0,1,3}` give
   the same 31.795049961, as they must, since the latter's optimum sits on the former.
2. **My earlier sign convention was sloppy.** I reported the multiplier on coordinate 2 as `−10.433`;
   the correct comparison is `grad_i − λ`, which is `−1.95` on face `{0,1}`. `grad_i` itself is
   `−10.4`. The conclusion — coordinate 2 must be released — was right; the number I attached to it
   was not.

Only one face is optimal, and it contains `θ₂ > 0`, so any solver that pins `θ₂ = 0` is at a
non-optimal point.

## 4. Is the defect in the formulation or in the custom Newton-CG?

**In the custom Newton-CG.** Head-to-head, all three points evaluated in the *same* independent
objective:

| rel ε | method | Ψ | θ | row err | col err |
|---|---|---|---|---|---|
| 0.5 | B independent | **31.77313016754** | (0.4810, 0.4967, **0.0224**, 0) | 3.3e-16 | 7.5e-7 |
| 0.5 | C custom | 31.79829950532 | (0.4938, 0.5062, **0**, 0) | 2.2e-16 | **9.1e-2** |
| 0.25 | B independent | **12.81870766401** | (0.4484, 0.4929, 0.0587, 0) | 4.4e-16 | 1.9e-7 |
| 0.25 | C custom | 14.32539777403 | (0.6517, 0.2477, 0.1006, 0) | 4.4e-16 | **1.6** |

The diagnostics that locate the defect precisely:

- **The custom solver's own state at the stall is a non-optimal KKT point.** It reports a free-block
  gradient of `0.37` and stalls 16 steps in. Its `θ` gradient on the pinned coordinate is `−15.15`
  while the pivot is `−8.98`; the correct stationary comparison is `−6.16 < 0`, i.e. mass belongs on
  `θ₂`.
- **Its target marginal is badly violated** (`9.1e-2` at ε=0.5, `1.6` at ε=0.25) while the
  independent solver reaches `1e-7`. The `g` block is not converging either, so this is not a
  θ-only problem.
- **Its own gradient and Hessian are correct.** I checked both independently by central finite
  differences (`2e-9`, `5.3e-9`) and by step-size refinement (ratio `1.048 → 1.011 → 1.002 → 1.0004`
  as `α → 0`). My earlier claim of an "inconsistent Hessian" was wrong; the model was right.
- **Therefore the fault is in the control flow**, not in the calculus it calls: the projection onto
  the constrained tangent space, the step acceptance, and the release logic. The release branch
  requires `kkt_ok`, but `kkt_ok` is false exactly when a release is needed — the branch is
  unreachable in the one case it exists for.

## 5. Recommended next step

**Do not repair the custom Newton-CG as the baseline.** It is 16+ iterations to a wrong answer at
ε=0.5, 1 iteration at ε=0.25, and its diagnostics are self-contradictory (`converged=False` while the
trace is flat). The independent SLSQP solves the same problem to `1e-12` in **1 iteration and 0.01 s**
on this instance.

Concretely:

1. **Adopt the independent SLSQP as the Gate B reference and as the working solver** for this
   parameter range, and re-run Gate B with it as the passing implementation. It is already exercise-
   tested (7 tests) and does not need a bespoke active set.
2. **Keep the custom Newton-CG only if HVP counts matter at scale**, and if so, rebuild it around a
   single derived reduction rather than two special cases — but that is an optimisation question for
   P2/P3, not a correctness blocker.
3. Do **not** proceed to P2/P3 or the certificate until Gate B is re-stated against the SLSQP
   reference, with the same KKT checks used here.

One caveat worth stating now: SLSQP is dense and finite-dimensional; at larger `n` its cost grows, so
"SLSQP works" is a statement about the formulation being right, not about scale. The HVP-based solver
may still be needed for `n` in the thousands — but it must be validated against this reference, not
against itself.
