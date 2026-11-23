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

## Results

**The control had to be fixed before these numbers meant anything.** The first version generated the control's plans
from `c* + 0.5·s³`. That is outside the 4-column family, but for the 8-column family `s³` **is a column** — so in the
`K = 8` cells the "control" was representable and its plans were reproduced exactly. The control now adds a seeded
**random symmetric** cost, which no small family can represent. All numbers below use it.

### The sweep over `n` and the number of features (the headline)

6 windows, 3 seeds, 18 observations per cell, `--diagonal-offset 0.52`, no same-zone weight, 120 Frank–Wolfe
iterations. Medians; `ratio` is control / planted, i.e. how much worse a misspecified cost does.

| `n` | `K` | planted `cost_rel_scaled` | control | ratio | planted `plan_l1` | control | ratio |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 20 | 4 | 0.0502 | 0.1595 | 3.2 | 0.0663 | 1.7665 | 26.6 |
| 20 | 8 | 0.0433 | 0.3312 | 7.7 | 0.0852 | 1.8364 | 21.6 |
| 40 | 4 | 0.0167 | 0.1462 | 8.8 | 0.0290 | 1.9032 | 65.7 |
| 40 | 8 | 0.0444 | 0.3266 | 7.4 | 0.0322 | 1.9163 | 59.4 |
| 80 | 4 | 0.0207 | 0.1435 | 6.9 | 0.0301 | 1.9371 | 64.4 |
| 80 | 8 | 0.0358 | 0.3311 | 9.3 | 0.0500 | 1.9629 | 39.2 |
| 160 | 4 | 0.0184 | 0.1405 | 7.6 | 0.0322 | 1.9738 | 61.4 |
| 160 | 8 | 0.0347 | 0.3239 | 9.3 | 0.0425 | 1.9793 | 46.6 |

Three readings:

1. **The planted arm stays separated from a misspecified cost at every `n`**: 3–9× on the recovered-cost error and
   22–66× on plan distance. Nothing collapses as `n` grows; the failure the exact method shows at small `n` then
   large `n` is not reproduced here.
2. **Recovered-cost error saturates, it does not fall to zero**: 0.050 at `n = 20`, then 0.017–0.021 for
   `K = 4` — flat from `n = 40` to 160. More points do not identify the cost (the plans pin a cone).
3. **Four extra feature columns hurt**: `K = 8` is worse on the recovered cost at every `n` (0.035–0.044 vs
   0.017–0.021) and slightly worse on `plan_l1`, and the gap does not close with `n`. The extra profiles carry zero
   weight in the planted cost, so this is a pure cost of the larger parameterisation.

### The two holes, and the knobs that close them

Planted arm at `n = 40`, 6 windows, 3 seeds, new control:

| planted cost | diagonal mass (planted plans) | `plan_l1` planted / control | `cost_rel_scaled` planted / control | `share` planted / control |
| --- | ---: | ---: | ---: | ---: |
| `w_same = 0.30`, offset 0 (first version) | **0.0000** | 0.015 / 1.898 | 0.460 / 0.957 | 0.000 / 2.72 |
| `w_same = 0`, offset 0 | 0.618 | 0.114 / 1.896 | 0.067 / 0.549 | 0.000 / 2.00 |
| `w_same = 0`, offset 0.52 | 0.660 | 0.031 / 1.891 | **0.018 / 0.146** | 0.000 / 0.90 |

The first row is the confound: with 0.30 on the same-zone column the planted plans never touch the diagonal, and a
cost supported only on that column makes any diagonal-free plan optimal. The middle row is the second hole: removing
that weight makes the diagonal **free** (`d = 0`), the plans put 0.62 of their mass in place, and the control comes
closest to the planted arm (0.067 vs 0.549 on cost, but only 0.114 vs 1.896 on plan distance). `--diagonal-offset`
adds a constant to every distance so staying put costs a typical trip; the diagonal stays in play and the separation
returns.

**Correction to an earlier statement.** I had written that `share` is 0 in both arms and therefore cannot detect a
wrong model. That was true of the old `s³` control, whose plans were diagonal-free and hence optimal under the
same-zone-only cost the fit collapsed to. With the random-cost control, whose plans do use the diagonal, `share`
separates cleanly: 0.000 planted against 0.90–2.72 control. The metric is diagnostic once the control is genuinely
out of family, and its readings above 1 mean the observed plan is *more* wasteful than the unrelated plan.

### Does `θ` converge with more windows?

No. Fitting on all windows (`--recovery-only`), planted arm, three seeds:

| windows | `cost_rel_scaled` | `θ_l1` to `θ*` | control `cost_rel_scaled` |
| ---: | --- | --- | --- |
| 6 | 0.0096 / 0.0184 / 0.0068 | 0.210 / 0.264 / 0.171 | 0.138–0.151 |
| 12 | 0.0145 / 0.0182 / 0.0119 | 0.192 / 0.265 / 0.231 | 0.138–0.152 |
| 25 | 0.0148 / 0.0149 / 0.0149 | 0.229 / 0.250 / 0.251 | 0.138–0.151 |
| 50 | 0.0148 / 0.0138 / 0.0252 | 0.229 / 0.247 / 0.289 | 0.138–0.151 |

Flat from 6 to 50 windows: more marginals of the *same* points do not identify the cost further. What does grow with
windows is the control's *training* objective (0.04 → 0.36) while the planted one stays at 0 — a misspecification
becomes more visible with more data even though the parameter does not become identified.

### New points per window

`--fresh-points` (same `θ*`, a new point cloud in every window), offset 0.52: planted `plan_l1` 0.041,
`cost_rel_scaled` 0.017, control 1.903 / 0.150. Statistically the same as the shared-geometry case
(0.031 / 0.018), so the arm transfers to a held-out **geometry**, not only to new marginals.

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
python -m experiments.studies.iot.synthetic.planted_shared_theta.run                       # first version
python -m experiments.studies.iot.synthetic.planted_shared_theta.run --diagonal-offset 0.52 --same-zone-weight 0
python -m experiments.studies.iot.synthetic.planted_shared_theta.run --fresh-points --diagonal-offset 0.52
python -m experiments.studies.iot.synthetic.planted_shared_theta.run --recovery-only --windows 50 --diagonal-offset 0.52
```

Tests: `tests/iot_planted_shared_theta_test.py` (the pipeline reproduces the held-out plan while the parameter is not
recovered; a same-zone-only cost makes any diagonal-free plan optimal; the offset removes the free diagonal).
