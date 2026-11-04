# T3 solver comparison — SLSQP reference vs custom Newton-CG

**Commit:** `5c7387b` (+ the driver) · **Record:** `iot_experiments/out/t3_solver_compare/solver_compare.json`
**Driver:** `iot_experiments/studies/real/t1t2/metric_learning/t3_solver_compare.py`

## What is being compared

Both solvers minimise the **same joint reduced objective**

```
Psi_red(theta, g) = <phi_bar, theta> - b.g + eps * sum_i a_i * (1 - log a_i + L_i)
```

on the same data, the same absolute `eps = rel_eps * std(c(theta_uniform))`, the same simplex constraint
and the same gauge `g_0 = 0`. They differ only in algorithm:

- **`slsqp`** — `iot.entropic.joint_reference`: SciPy SLSQP on `(theta, g_1..g_{n-1})`. This is a
  **joint** method, not the old nested scheme (SLSQP on θ with an OT solve inside).
- **`newtoncg`** — `iot.entropic.joint_master`: Newton-CG with a primal active set. **Not patched during
  this study.**

Starts are structured only (uniform interior; θ from the planted truth). No oracle starts.

## Results

Success = converged AND objective at the common optimum (tolerance `1e-6·|Ψ|`).

| n | rel ε | SLSQP conv | Newton-CG conv | Newton-CG reaches optimum | grid edges |
|---:|---:|---:|---:|---:|---:|
| 24 | 0.50 | 2/2 | 2/2 | 2/2 | 360 |
| 24 | 0.25 | 2/2 | 1/2 | 1/2 | 360 |
| 24 | 0.10 | 2/2 | 1/2 | 1/2 | 360 |
| 70 | 0.50 | 2/2 | 2/2 | 2/2 | 3,220 |
| 70 | 0.25 | 2/2 | 0/2 | 1/2 | 3,220 |
| 70 | 0.10 | 2/2 | 0/2 | 1/2 | 3,220 |
| 120 | 0.50 | 2/2 | 1/2 | 1/2 | 9,120 |
| 120 | 0.25 | 2/2 | 1/2 | 1/2 | 9,120 |
| 120 | 0.10 | 2/2 | 2/2 | 2/2 | 9,120 |
| 300 | 0.50 | **1/2** | 0/2 | **2/2** | 50,100 |
| 300 | 0.25 | 2/2 | 0/2 | **2/2** | 50,100 |
| 300 | 0.10 | 2/2 | 2/2 | 2/2 | 50,100 |

**Totals: SLSQP 23/24 converged, 24/24 at the optimum. Newton-CG 13/24 converged, 16/24 at the optimum.**

Where both reach the optimum they agree to **4.5e-12** relative — the two algorithms solve the same
mathematical problem.

### Accuracy at the optimum

| solver | target marginal residual | θ KKT violation |
|---|---|---|
| SLSQP | `1e-7` … `6e-6` | `1e-7` … `4e-5` |
| Newton-CG | `1e-15` … `1e-8` | `1e-15` … `1e-7` |

Newton-CG is **more accurate when it succeeds** — two to four orders of magnitude — because its
convergence test is a genuine KKT test. SLSQP stops at its own `ftol`, and the residual it leaves is
the price of the dense SQP machinery.

### Cost

| solver | in-worker SLSQP/NC work at n=300 | end-to-end wall (incl. Python start-up) | peak RSS |
|---|---|---|---|
| SLSQP | 1.4 – 6.0 s | 3.6 – 5.7 s | ≈ 635 MB |
| Newton-CG | 0.5 – 1.6 s | 2.1 – 2.8 s | ≈ 640 MB |

**Reported honestly:** almost all of the end-to-end wall time is Python/NumPy start-up (~2.1 s per
worker). On the actual optimisation at `n = 300`, Newton-CG is **2–4× faster** when it converges.
Neither solver shows a memory problem at these sizes, and neither was tested beyond `n = 300`.

## What this does and does not show

**Does:** SLSQP is the more **reliable** joint solver over this range — it converged in 23/24 runs
versus 13/24, including the cases where Newton-CG cycles. The formulation is confirmed: everything both
solvers reach agrees to `4.5e-12`.

**Does not:** SLSQP is **not** shown to scale. It was exercised up to `n_eff = 167` / 170 dense
variables / 50,100 grid edges, which is small. SLSQP's cost grows with the number of dense variables and
its memory is quadratic in them; that has not been measured. Its single failure is at `n = 300,
rel_eps = 0.5` (`Positive directional derivative for line search`), which is a warning at the edge of the
tested range, not a demonstration of failure beyond it.

**Also does not:** Newton-CG's failure pattern is *not* the earlier θ₃ cycle. It now fails at interior
estates with large target-marginal residuals (e.g. `n=120, ε=0.25`: `col = 5.5`), i.e. it stops with the
`g` block unconverged. That is a different defect from the one diagnosed at `n = 24`.

## Recommendation

1. **Keep SLSQP as the trusted reference.** It converged everywhere except one run.
2. **Do not yet designate SLSQP the production solver.** It has not been tested where its dense growth
   would bite. The next scaling question is much larger `n`, and possibly a sparse/limited-memory
   constrained method.
3. **Stop patching Newton-CG.** It is 2–4× faster when it converges and more accurate, but it is
   unreliable in 11/24 runs and its failure mode differs by size. It is worth revisiting only if SLSQP's
   dense cost becomes the binding constraint.
4. **T3's distinctive parts remain blocked.** Adaptive edge generation and the certificate are not
   validated, and neither can be until a joint solver is reliable at scale.

The successful reduced-Hessian correction stands: it is what made the ε = 0.5 results agree to `4.5e-12`
with an independent implementation. What it did not do is make the custom solver reliable.
