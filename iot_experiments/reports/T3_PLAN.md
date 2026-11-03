# T3 — priorities (user, 2026-11-03)

Binding order. Do not reorder. Do not merge steps.

## P1 — Full-grid reference (F = 100 %)

Solve the planted problem on all edges, at each ε. **Newton-CG and Sinkhorn independently**,
recorded side by side: convergence, marginal residuals, objective value, Newton steps, CG
iterations, total time.

**Also solve the full θ master, not just the fixed-θ transport problem.**

## P2 — How small can F actually be?

From the full-grid reference solution, build several edge sets:

- `B` + edges carrying **99 %** of the mass;
- `B` + edges carrying **99.9 %** of the mass;
- `B` + edges selected by **per-row and per-column mass thresholds**.

Re-solve each restricted problem, compare against the full-grid reference.

Purpose: does a smaller `F` **reproduce the answer**, or merely carry most of its mass?

## P3 — Adaptive loop vs oracle

Run the actual adaptive loop on the same instances and ε. Compare against the P2 oracle:

| measure | oracle F | adaptive F |
|---|---|---|
| number of edges | ✔ | ✔ |
| grid coverage | ✔ | ✔ |
| θ error | ✔ | ✔ |
| objective error | ✔ | ✔ |
| marginal residual | ✔ | ✔ |
| computation time | ✔ | ✔ |

**Key sentence to keep:** if the oracle needs 5 % of the grid but T3 needs 50 %, the problem is
not necessarily the entropic penalty — it may be **how T3 discovers its active edge set**.
