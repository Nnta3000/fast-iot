# EOT — priorities (user, 2026-11-03)

Binding order. Do not reorder. Do not merge steps.

> **Progress (2026-11-05).** `[ ]` not run · `[~]` partly done or run without a report · `[x]` done.
> The status of each step is tracked in `README.md` §2; this file keeps the user's wording unchanged.

## [~] P1 — Full-grid reference (F = 100 %)

Solve the planted problem on all edges, at each ε. **Newton-CG and Sinkhorn independently**,
recorded side by side: convergence, marginal residuals, objective value, Newton steps, CG
iterations, total time.

**Also solve the full θ master, not just the fixed-θ transport problem.**

*Progress:* the formulation half is answered by `eot_gate_b.md` (independent SLSQP agrees to 1e-12) and
`eot_solver_compare.md` (both solvers compared, `n ∈ {24,70,120,300}`). The Newton-CG-vs-Sinkhorn
side-by-side on the **fixed-θ transport problem** lives in `eot_p1_reference.py` and has **no record and
no report**.

## [x] P2 — How small can F actually be?

From the full-grid reference solution, build several edge sets:

- `B` + edges carrying **99 %** of the mass;
- `B` + edges carrying **99.9 %** of the mass;
- `B` + edges selected by **per-row and per-column** mass thresholds.

Re-solve each restricted problem, compare against the full-grid reference.

Purpose: does a smaller `F` **reproduce the answer**, or merely carry most of its mass?

*Result:* `eot_restricted_edges.md`. Sparse masters are cheap and accurate, but 5–10 % of the grid keeps
only 7–29 % of the full-grid plan mass. At `n=300, rel_ε=0.25` an oracle needs ≈52 % of the grid for
99 % of the mass. **The user's fractions (1–20 %) are an order of magnitude too small.**

## [ ] P3 — Adaptive loop vs oracle

Run the actual adaptive loop on the same instances and ε. Compare against the P2 oracle:

| measure | oracle F | adaptive F |
|---|---|---|
| number of edges | ✔ | ✔ |
| grid coverage | ✔ | ✔ |
| θ error | ✔ | ✔ |
| objective error | ✔ | ✔ |
| marginal residual | ✔ | ✔ |
| computation time | ✔ | ✔ |

**Key sentence to keep:** if the oracle needs 5 % of the grid but EOT needs 50 %, the problem is
not necessarily the entropic penalty — it may be **how EOT discovers its active edge set**.

*Status:* **blocked.** The adaptive loop and the certificate are not validated; see
`eot_restricted_edges.md` §"Still blocked".
