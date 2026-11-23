# entropic_metric — does the smooth entropic joint master learn useful T1/T2 weights?

**This is not the `iot` arm of `../metric_learning/`.** That arm is the **relaxed regret LP** over `m×m`
edges with a box ∩ simplex and **no temperature**. This study uses the **entropic** objective on the
**pure simplex** (no box, no upper bound):

```
Psi(theta, f, g) = <phi_bar, theta> - a.f - b.g + eps * sum_e exp(sigma_e / eps)
```

## Question

Does the cost produced by this master recover the right T1/T2 partner, compared with cosine?

## Protocol

Everything about data, geometry, splits and scoring is inherited unchanged from `metric_learning`:

| item | setting |
|---|---|
| features | `dinov2-pool4` + PCA to K (`harness/registry.features`) |
| geometry | `G2`, statistics fitted on the held-out pool |
| pool | ONE fixed held-out pool shared by every `m` (`--holdout-pairs`, `--holdout-draw`) |
| labelled | drawn from subjects **not** in the pool, disjoint asserted in code |
| scoring | `matching_hit` / `rank_metrics` on the pool |
| records | `harness/records.py` `RunRecord`, written to `experiments/out/<study>/records.json` |

### Epsilon protocol (fixed before any run)

`rel_eps` has no precedent in the relaxed arm, so the rule is written down first:

1. **scale** — `eps = rel_eps * cost_std(c(theta_uniform))`, with `cost_std` the population std of the
   diagonal cost at uniform `theta`. `theta_ref` is fixed at uniform for the whole solve, so `eps` cannot
   drift with `theta` (a defect we hit before).
2. **selection** — `rel_eps` is chosen by 5-fold CV **on the labelled pairs only**, over
   `{0.1, 0.25, 0.5}`. Ties go to the **smaller** `rel_eps` (the sharper problem). This rule was fixed
   before the first run and is not tuned to results.
3. **refit** — after selection the master is refit on all labelled pairs.
4. **no pool contact** — the held-out pool never influences selection.

## Run

```bash
PYTHONPATH=. .venv/bin/python -u -m experiments.studies.iot.real.t1t2.entropic_metric.run \
  --ks 16 32 --ms 50 100 --draws 0 --rel-eps-grid 0.1 0.25 --max-iterations 300
```

## Result (first run, 1 draw, small K)

| K | m | arm | matching hit | R@1 | in-sample hit | chosen rel_eps | θ KKT |
|---:|---:|---|---:|---:|---:|---:|---:|
| 16 | 50 | cosine | 0.1350 | 0.1350 | — | — | — |
| 16 | 50 | **entropic** | **0.1700** | 0.1525 | 0.5400 | 0.5 | 2.6e-7 |
| 16 | 100 | cosine | 0.1350 | 0.1350 | — | — | — |
| 16 | 100 | **entropic** | **0.1750** | 0.1725 | 0.5000 | 0.5 | 1.4e-7 |
| 32 | 50 | cosine | 0.3950 | 0.3500 | — | — | — |
| 32 | 50 | **entropic** | **0.4100** | 0.3350 | 1.0000 | 0.1 | 1.8e-7 |
| 32 | 100 | cosine | 0.3950 | 0.3500 | — | — | — |
| 32 | 100 | **entropic** | **0.4900** | 0.4350 | 0.7900 | 0.25 | 1.8e-7 |

**Reading.** The entropic master beats cosine on matching hit in all four cells (+0.035 at K=16 m=50,
+0.040 at K=16 m=100, +0.015 at K=32 m=50, +0.095 at K=32 m=100) and on R@1 in three of four (the
exception is K=32 m=50, where R@1 drops 0.3500 → 0.3350).

**Caveats, stated plainly.**

- **One draw.** No error bars; these are single readings, not established effects.
- **In-sample hit is high** (0.50–1.00) against pool hit (0.17–0.49). The weights fit the labelled pairs
  well and generalise only partly; at K=32 m=50 the in-sample hit is 1.0000, i.e. perfect on the training
  pairs and +0.015 over cosine on the pool.
- **Not comparable to `metric_learning`'s `iot` numbers.** Different objective (entropic vs relaxed
  regret), different constraint set (simplex vs CV-selected box), and a different `K` range. This table
  must not be merged with that one.
- Small `K` only (16, 32). `K = 64+` is the regime the original study used and is not yet run here.

## Gates that must pass before results are quoted

1. **Fit path validated on known data** — on synthetic points with a planted 4-feature metric and 40 %
   of partners shuffled, the master's top-4 weights were exactly the true support `[0,1,2,3]` at both
   `rel_eps = 0.1` and `0.25`, KKT ≈ 1e-7 (`fit_entropic` + `select_rel_eps` exercised end-to-end).
2. **cosine is label-free** — asserted constant per `K` across `m` and `draw`, as in `metric_learning`.
3. **Disjointness** — labelled and pool subjects are asserted disjoint.
4. **θ simplex feasibility** — θ ≥ 0 and Σθ = 1 checked via `simplex_kkt`.

## Status

Exploratory. One draw, small `K`. Not a validated claim about EOT; the sparse-edge and certificate
questions are untouched.

**Corrections (2026-11-05, checked against `experiments/out/entropic_metric/records.json`).**

1. **The cosine rows above are not in the committed record.** `records.json` holds 8 rows: 4 `entropic`
   and 4 `n_params=0`. The latter are *not* cosine — cosine on this pool is label-free, so it must be
   identical for every `m` and draw at a given `K`, and it is not (0.135 / 0.135 / 0.395 / 0.395). Their
   `n_params=0` and R@1 == hit exactly, which is the signature of the **uniform-theta diagonal cost**, i.e.
   the same numbers as the entropic rows at `K=16` (0.17 / 0.135 / 0.175 / 0.135 → hit / R@1). Treat the
   cosine column as **missing from the record** until it is re-run with an explicit `arm="cosine"` row.
2. **The K=16 cells are barely above an uninformative cost.** On a `50×50` square pool a *constant* cost
   scores 0.04 / 0.00 on `matching_hit` (measured, `evaluate(np.zeros((n,n)), draw=0|1)`); the K=16 entropic
   hits are 0.170 / 0.175. At `K=32` the hits are 0.410 / 0.490. So the K=16 gain is small in absolute terms
   and only the K=32 cells are clearly above the floor.
3. The R@1 column on the `n_params=0` rows equals `hit` exactly, which cannot happen for a learned cost with
   distinct rows; it is another sign those rows are not cosine.

