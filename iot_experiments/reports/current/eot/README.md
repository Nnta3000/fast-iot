# EOT — what it is, what has been done, what is open

**Status: CURRENT (2026-11-05).** This is the index for the **entropic inverse-OT master** family.
Formerly called "T3"; the reports, drivers and study directories were renamed on 2026-11-05, so an older
note that says "T3" means this directory.

## 0. What "EOT" is

Instead of maximising the dual margin `delta` (the exact loop) or minimising a regret over the relaxed
LP (the `iot` arm of `metric_learning`), EOT minimises an **entropic, jointly convex** objective on an
edge set `F`:

```
Psi_F(theta, f, g) = <phi_bar, theta> - a.f - b.g + eps * sum_{e in F} exp(sigma_e / eps)
sigma_e = f_i + g_j - theta.phi_e
```

`Psi_F` is convex in `(theta, f, g)`, so a converged point is a global minimiser of the objective on `F`.
Two code paths exist and they are **not** interchangeable:

| path | module | is it EOT proper? |
| --- | --- | --- |
| **joint** — one Newton-CG on `(theta, g)`, `f` eliminated in closed form, plan falls out of the potentials | `src/iot/entropic/joint_master.py` | **yes** |
| **nested** — SLSQP on `theta` with a full entropic OT solve per `theta` | `src/iot/entropic/soft_master.py` | no — comparison only |

`src/iot/entropic/joint_reference.py` is an **independent** reduced-objective SLSQP written from the
formulas (scipy `logsumexp`), used as the trusted reference. `src/iot/entropic/__init__.py` carries the
same table for anyone reading the code.

## 1. What has been done

| # | report | question | verdict | record |
| --- | --- | --- | --- | --- |
| 1 | `eot_gate_b.md` | does the joint formulation have the right optimum? | **yes** — matches an independent SLSQP reference to 1e-12 from 4 starts; formulation is correct | `out/eot_joint_reference/joint_reference.json` |
| 2 | `eot_solver_compare.md` | which solver carries it — SLSQP reference or the custom Newton-CG? | **open** — SLSQP 24/24 at the optimum by flag 23/24; Newton-CG 16/24. Recommendation: keep SLSQP as the reference, do not yet make it production | `out/eot_solver_compare/solver_compare.json` |
| 3 | `eot_scaling_slsqp.md` | where does dense SLSQP stop being practical? | **measured to n=1500** — optimiser time 1.6 → 17.9 s (`rel_eps=0.25`), peak RSS 848 MB, per-evaluation cost ×14.5. Not memory-bound at these sizes | `out/eot_scaling_slsqp/scaling_slsqp.json` |
| 4 | `eot_restricted_edges.md` | does a restricted `F` stay cheap **and** still approximate the full grid? | **cheap yes, close no** — 5 % of the grid solves to 3.4e-6, but keeps only 7 % of the full-grid plan mass; at `n=300, rel_eps=0.25` 99 % of the mass needs ≈ 52 % of the grid | `out/eot_restricted_edges/restricted_f.json` |
| 5 | `eot_bias.md` (record only) | is `argmin_theta inf_{f,g} Psi` an estimator of `theta_star`? | **no, not even at small eps** — `L1(theta_hat, theta_star)` 0.203 → 0.206 → 0.170 on `n=40`, 0.285 → 0.241 → 0.155 on `n=120` at `rel_eps` 0.5 / 0.25 / 0.1; `theta_hat` sits on a simplex vertex | `out/eot_bias/bias.json` |
| 6 | `eot_inner_observations.md` | conditioning of the inner solve, plan sparsity, sweep counts, gradient propagation | **partial, measurement only** | `out/eot_inner_observations/observations.json` |

### Rows with **no record on disk** — do not cite as numbers

| driver | study name | state |
| --- | --- | --- |
| `eot_planted.py` | `eot_planted` | **no `out/eot_planted/` directory**; nothing was kept |
| `eot_p1_reference.py` | `eot_p1` | **no `out/eot_p1/` directory** |
| `eot_edge_fraction.py` | `eot_scaling` | record exists (`out/eot_scaling/scaling.json`), **no report**; it is the `|F|`-fraction-vs-`m` study and is the one that decides whether 20 % of the grid survives at scale |

`eot_edge_fraction.py` and `eot_planted.py` are the two drivers whose result would most change the
verdict in row 4; neither is written up.

## 2. Status against `PLAN.md` (the user's binding P1/P2/P3 order)

| step | what it asks | state |
| --- | --- | --- |
| **P1** | full-grid reference: transport problem and full-theta master, Newton-CG **and** Sinkhorn, side by side | **partly done, unwritten.** `eot_p1_reference.py` exists; no record, no report. The formulation half of P1 is covered by `eot_gate_b.md` |
| **P2** | how small can `F` be before it stops reproducing the answer? | **done and negative** — `eot_restricted_edges.md`. The honest range is 50–80 % of the grid, not 5–20 % |
| **P3** | adaptive loop vs the P2 oracle, on edges / coverage / `theta` error / objective / marginals / time | **blocked** — `eot_restricted_edges.md` §"Still blocked". The adaptive loop and the certificate are not validated |

## 3. How to read the four solver verdicts together

`eot_gate_b.md`, `eot_solver_compare.md` and `eot_scaling_slsqp.md` were written on consecutive days
and each revises the previous one. The order of authority is:

1. `eot_gate_b.md` — the **formulation** question. Answer: right.
2. `eot_solver_compare.md` — the **solver** question. Answer: neither solver is certificate-grade;
   SLSQP finds the optimum more reliably, Newton-CG is 2–4× faster and more accurate when it converges.
3. `eot_scaling_slsqp.md` — the **scale** question. Answer: SLSQP is practical to `n≈1500` with
   `n_eff≈900`; its KKT residuals (1e-6…1e-4) are not certificate-grade, so the certificate cannot
   rest on it.
4. `eot_restricted_edges.md` — read last; it **corrects** the "20 % of the grid is enough" premise and
   puts P2/P3 at risk.

## 4. Config of every row above

All rows: planted instances, `family=diagonal` (Monge, exactly optimal `pi_hat` under `theta_star`),
`point_dimension=4`, seed 0 unless stated; `epsilon = rel_eps * std(c(theta_uniform))`; simplex
`Sum(theta)=1, theta>=0`; gauge `g_0=0`. Sizes are per report (row 1: `n=24`; row 2: `n ∈ {24,70,120,300}`;
row 3: `n ∈ {300,450,675,1000,1500}`; row 4: `n ∈ {70,120,300}`, seeds {0,1}).
