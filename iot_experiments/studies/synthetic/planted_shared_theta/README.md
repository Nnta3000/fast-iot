# planted_shared_theta — the relaxed arm's own sanity check

**Status: sanity check, not evidence for the method.** It asks one narrow question: when the observed plans come from
the arm's own model, does a cost fitted on some windows transfer to a held-out one? Passing says the pipeline is not
broken. It says nothing about real data — the NYC taxi study addresses that, and there the same arm does not predict
flows.

## Protocol

1. `n` points in `[0,1)²` → distance matrix `d`; features `{s, s², log(1+s), same-zone}` with `s = d/max(d)`, scaled
   to unit Frobenius norm (the gauge the taxi study uses).
2. Plant `θ* = (0.10, 0.55, 0.35, w_same)/Σ` on the simplex; `c* = Σ_k θ*_k Φ_k`.
3. `windows` marginal pairs (`Dirichlet(1)+0.1`); observed plans `P_l = EMD(a_l, b_l; c*)` — **exactly optimal at
   `θ*` by construction**.
4. Leave one window out: fit `θ` on the rest (relaxed arm, summed regret, Frank–Wolfe), evaluate on the held-out one.
   `--recovery-only` instead fits on all windows, which is the right shape for "does `θ` converge with more windows".

Readings on the held-out window: `F_train`; `R_test` and `share = R/R_independent`; `plan_l1` against the planted
plan; `plan_cost_rel` = excess cost of the predicted plan under `c*`; `cost_rel_scaled` = the fitted cost up to the
positive scale that leaves every OT plan unchanged; `θ_l1`; and the **diagonal mass** of both plans.

`--arm control` repeats the whole pipeline with plans generated at `c* + 0.5·s³`, a direction the family does not
contain. A reading that cannot separate the control from the planted case is not diagnostic.

## Three knobs, each closing a hole found by measurement

| knob | why it exists |
| --- | --- |
| `--same-zone-weight` (default 0.0) | the first version put 0.30 on the same-zone column, so the planted plans never touched the diagonal — **diagonal mass 0.0000**. A degenerate cost supported only on that column makes any diagonal-free plan optimal, so the test could be passed by a cost that ignores everything else |
| `--diagonal-offset` (default 0.0) | with no same-zone weight the diagonal is *free* (`d = 0`), so the planted plans put **0.62** of their mass in place and the control becomes hard to separate (`cost_rel` 0.067 vs 0.082). The offset adds a constant to every distance, so staying put costs a typical trip |
| `--fresh-points` | every window gets its own point cloud (same `θ*`), so the held-out plan is for a **new geometry** — generalisation to new pairs, not just to new marginals of the same points |

## Results (n = 40, 6 windows, 3 seeds, 18 folds per arm, CPU, seconds)

Planted arm, `cost_rel_scaled` (lower is better) and `θ_l1` to the planted parameter:

| planted cost | `diagonal mass` (planted plans) | `plan_l1` planted / control | `cost_rel_scaled` planted / control | `θ_l1` planted / control |
| --- | ---: | ---: | ---: | ---: |
| `w_same = 0.30`, offset 0 (first version) | **0.0000** | 0.015 / 0.094 | 0.460 / 0.956 | 0.58 / 1.53 |
| `w_same = 0`, offset 0 | 0.618 | 0.114 / 0.277 | 0.067 / 0.082 | 0.55 / 0.57 |
| `w_same = 0`, offset 0.52 | 0.660 | 0.031 / 0.162 | **0.018 / 0.045** | 0.26 / 0.75 |

The middle row is the one to note: removing the same-zone weight as asked does **not** strengthen the test on its own —
it swaps a penalised diagonal for a free one, the plans pile up in place, and the control's cost error approaches the
planted one (0.082 vs 0.067). The offset is what closes both holes.

**Does `θ` converge with more windows?** No. Fitting on all windows (`--recovery-only`, offset 0.52), planted arm:

| windows | `cost_rel_scaled` (three seeds) | `θ_l1` (three seeds) |
| ---: | --- | --- |
| 6 | 0.0096 / 0.0184 / 0.0068 | 0.210 / 0.264 / 0.171 |
| 12 | 0.0145 / 0.0182 / 0.0119 | 0.192 / 0.265 / 0.231 |
| 25 | 0.0148 / 0.0149 / 0.0149 | 0.229 / 0.250 / 0.251 |
| 50 | 0.0148 / 0.0138 / 0.0252 | 0.229 / 0.247 / 0.289 |

The parameter does not approach `θ*`; the residual is flat from 6 to 50 windows. The plans pin the cost down to a cone
and more marginals of the *same* points add little. (This is the identifiability statement, not a defect — but it means
"more data will identify the cost" is not supported.)

**New points per window** (`--fresh-points`, offset 0.52, planted): `plan_l1` 0.041, `cost_rel_scaled` 0.017,
`θ_l1` 0.23 — statistically the same as the shared-geometry case (0.031 / 0.018 / 0.26). So the arm also transfers to
a held-out *geometry*.

## What the numbers do and do not say

- **Does:** the pipeline reaches `F_train ≈ 0` on data from its own model, the held-out plan is optimal under the
  transferred cost, and it is usually reproduced exactly; a control generated outside the family is separated by
  2.5–5× on `cost_rel_scaled` and `plan_l1`.
- **Does not:** the parameter is not recovered and does not converge with more windows; `share` and `R_test` are ~0 in
  both arms, because a cost that vanishes on the support (or only on the diagonal) makes any such plan optimal — so
  the arm's own scalar cannot detect a wrong model. Any claim must be made on plan/cost comparison against a control.
- **Does not:** three seeds with six correlated folds per seed is thinner than "18" suggests. And the whole test is a
  necessary condition.

## Reproduce

```bash
python -m iot_experiments.studies.synthetic.planted_shared_theta.run                       # first version
python -m iot_experiments.studies.synthetic.planted_shared_theta.run --diagonal-offset 0.52 --same-zone-weight 0
python -m iot_experiments.studies.synthetic.planted_shared_theta.run --fresh-points --diagonal-offset 0.52
python -m iot_experiments.studies.synthetic.planted_shared_theta.run --recovery-only --windows 50 --diagonal-offset 0.52
```

Tests: `tests/iot_planted_shared_theta_test.py` (the pipeline reproduces the held-out plan while the parameter is not
recovered; a same-zone-only cost makes any diagonal-free plan optimal; the offset removes the free diagonal).
