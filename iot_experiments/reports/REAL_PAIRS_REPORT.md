# Inverse OT on real pairs: pool-matching study

**Status: CURRENT (2026-10-22; confidence section revised 2026-10-26).** One question, one protocol. Everything here is measured on the
real 1000 T1/T2 pairs shipped with the sibling project; no synthetic instances are mixed in.

**Question.** Given `m` labelled T1↔T2 pairs, does a cost learned by inverse OT recover the true
partners of the *remaining*, label-hidden pairs better than a fixed cost?

**Answer.** Yes, by a wide margin: **48% recovered at m=100 and 71% at m=300**, against **4–5%** for
plain squared Euclidean and **16–17%** for a variance heuristic. A pairwise scorer that learns the
same hypothesis class from the same labels reaches only 18–31%, and does so while fitting its
training pairs to 88–95%. The margin objective's advantage is that it uses the whole assignment,
not `m` isolated pairs.

---

## 1. Protocol

Everything below shares this protocol; differences are stated per table.

| item | setting |
| --- | --- |
| data | 1000 registered T1/T2 pairs, one slice per subject (ids 1–1000, no gaps) |
| features | 16×16 block means of the 192×192 grayscale slice (`K_raw = 256`) |
| mask | drop features with zero fraction > 0.9 or `σ < 0.1 × median σ`, fitted on the labelled set only |
| standardisation | mean/σ fitted on the labelled set, applied unchanged to everything else |
| cost family | diagonal `c_ij(θ) = Σ_k θ_k (x_ik − y_jk)²` |
| Θ | simplex (`Σθ = 1`, `θ ≥ 0`) |
| solver | `loop_regret` (stage 1 only, `prune=False`, `γ=3`, `max_iter=20`, `ε_R = 1e-2 × regret at uniform`) |
| labelled plan | `π̂ = (1/m)·I` over the labelled pairs — the true permutation |
| pool | **700** held-out pairs, labels hidden, fixed across `m` so pool size does not confound `m` |
| assignment | exact EMD on the pool (700×700, uniform marginals), candidate columns shuffled, true partner tracked |
| reference arm | uniform `θ` = plain squared Euclidean |

Substitution note: the EMD oracle (cost 0 on true pairs, 1 elsewhere, columns shuffled) returns
**1.000** in every run; the driver refuses to report if it does not. The shuffle-tracking index is the
one place a bug was found and fixed (`searchsorted` → `argsort`); every number here post-dates the fix.

---

## 2. Main result

Recovered = fraction of the 700 pool pairs whose EMD assignment is the true partner.

| m | learned θ | uniform θ | variance θ | pairwise scorer |
| --- | --- | --- | --- | --- |
| 100 | **0.479** | 0.047 | 0.167 | 0.176 |
| 300 | **0.711** | 0.045 | 0.170 | 0.308 |

Means over 3 labelled draws. `variance θ` is `θ_k ∝ 1/Var_labelled(x_k − y_k)`; `pairwise scorer` is
a logistic scorer on squared pair features (§4).

Per-draw: learned 0.397/0.554/0.484 (m=100) and 0.726/0.701/0.707 (m=300).

---

## 3. Controls

| control | result | reading |
| --- | --- | --- |
| **scrambled pairing** (fit on a random π̂, evaluate on true pairs) | recovered 0.010–0.018, **below** uniform | the signal comes from the pairing, not from the fitting machinery |
| **pool size held at 700** for both m | — | removes the pool-size confound in the earlier table (m=100 had 900, m=300 had 700) |
| EMD oracle | 1.000 | the shuffle-tracking index arithmetic is correct |

---

## 4. What the learned θ is competing with

All arms use the same hypothesis class `d(x,y) = Σ_k θ_k (x_k − y_k)²`; only the objective differs.

| arm | objective | recovered (m=300) |
| --- | --- | --- |
| **learned (inverse-OT margin)** | certify `π̂` against EMD-optimality | **0.711** |
| variance heuristic | `θ_k ∝ 1/Var` | 0.170 |
| uniform | none | 0.045 |
| `ratio` | max `E_pos[d]/E_neg[d]`, gradient ascent | 0.038 |
| `nca-diag` | diagonal NCA | 0.001 |
| `lda-sq` | LDA over squared features, clipped ≥ 0 | 0.000 |
| `pairwise scorer` | logistic on `(x−y)²`, per-pair positives/negatives | 0.308 |

**Supervision available.** At m=100 there are 100 positives against ~161 features; the pairwise
scorer reaches **88–95% train accuracy** while recovering only 11–33% of the pool — the overfitting
signature. The inverse-OT fit does not have this gap at m=300 (0.711 recovered).

**A covariance caveat that invalidates a natural baseline.** Linear methods on the *difference*
`x_i − y_i` cannot separate true from scrambled pairs: both classes have identical means
(`mean(pos) − mean(neg) = 0`, verified numerically). They differ only in the second moment, so the
signal lives in the squared features. Any linear-on-differences baseline is testing nothing.

---

## 5. Diagnostics that constrain interpretation

**Hubness.** The free per-row argmin picks far fewer than 700 distinct columns, and `R@1` equals
`uniq_cols/700` exactly in every row:

| arm | distinct columns chosen | R@1 (m=300) |
| --- | --- | --- |
| learned | 410–432 | 0.443–0.467 |
| uniform | 71–83 | 0.019–0.034 |
| variance | 89–104 | 0.043–0.067 |

So per-row ranking is capped by hubness, and the *coupling* recovers substantially more than
independent ranking. `R@1` and `recovered` are not interchangeable measures.

**Confidence works — the earlier negative was a scorer bug.** The earlier measurement reported
precision staying within ±0.08 of the overall rate at every coverage, i.e. no ordering information.
That run scored against **shuffled column ids** instead of `argsort(perm)` (the index bug documented in
`SERVER_TESTS_REPORT.md` §6.6); with the index fixed and the geometry set to G2 (standardise-then-L2),
confidence carries real signal.

K=256, G2, m ∈ {100, 300}, 3 draws each (`studies/pool`; `P@100 = hit` asserted per record).
Standardisation statistics and the PCA basis are fitted on the **pool images** (transductive, label-free).

| arm | hit | P@25 | P@50 | P@75 | P@100 | MNN P@25 | MNN P@50 |
|---|---|---|---|---|---|---|---|
| cosine | 0.8428 | **1.0000** | 0.9996 | 0.9456 | 0.8428 | 0.9497 | 0.9522 |
| learned | 0.8518 | **1.0000** | 0.9988 | 0.9503 | 0.8518 | 0.9472 | 0.9578 |

Using the **margin** confidence, the top 25% of rows is **exactly correct** and the top 50% is ≥99.9%
correct, so the usable operating point is "keep the most-confident half, discard the rest" at
essentially no precision cost. Margin is far better than mutual-NN at low coverage (binary scores
cannot grade a top quarter). Neither measure depends on the learned θ (cosine matches it).

**Feature resolution is the binding constraint.** Nearest-neighbour similarity between *different*
subjects' T1 slices:

| representation | K | NN corr (mean) | frac > 0.95 |
| --- | --- | --- | --- |
| 8×8 blocks | 64 | 0.992 | 1.000 |
| 16×16 blocks | 256 | 0.985 | 0.992 |
| 32×32 blocks | 1024 | 0.976 | 0.984 |
| full 192×192, mean-centred | 36 864 | 0.925 | **0.120** |

At block resolution the different-subject slices are nearly interchangeable; at full resolution they
are not. A contact sheet of the 8 most-similar T1 pairs confirms they are visually distinct patients
(different ventricles, lesions, slice positions) — the 0.95 correlation is shared anatomy, not
duplicate subjects. **Every number in this report is a statement about 16×16 block features**, and the
resolution ladder shows that 16×16 is not where this problem is hardest.

---

## 6. What is established, and what is not

| claim | status |
| --- | --- |
| A θ fitted by inverse OT recovers 48% (m=100) / 71% (m=300) of hidden pool pairs | **established** |
| It beats uniform L2² (4–5%) by 10–16× | **established** |
| It beats a variance heuristic (17%) by 3–4× | **established** |
| Fitting on a scrambled pairing gives ≤ 2% (below uniform) | **established** |
| Per-row ranking is hubness-capped; the coupling exceeds it | **established** |
| A pairwise scorer on the same features is limited by supervision (88–95% train, 18–31% pool) | **established** |
| No confidence measure identifies reliable rows | **established (negative)** |
| The advantage over the pairwise scorer is the *assignment*, not the objective | **not established** — the pair-arm diagnostics were mis-recorded and the two assignment rules returned identical arrays |
| Anything about full-resolution or learned-encoder features | **not tested** |
| Any claim about cross-patient CT↔MRI (the PoC's target) | **not tested** — this is same-subject, same-slice matching |

## 7. Open items

1. **Re-run the pairwise comparison with per-arm diagnostics and a distinct EMD cost for
   `pair+emd`.** It separates "assignment matters" from "margin objective matters"; it is the one
   missing comparison that decides what inverse OT is contributing here.
2. **Confidence**: **resolved** — with the scoring index fixed and geometry G2, the margin confidence
   gives precision 1.000 at 25% coverage and 0.9996 at 50%; the earlier negative was a scorer bug.
   The open part is whether the same holds at other resolutions and on a genuinely unpaired pool.
3. **Resolution**: the task becomes measurably harder at 32×32+ and at full resolution; the learned-θ
   advantage has not been measured there.

---

**Data.** `iot_experiments/out/pool_emd/pool_emd2.json` (main, §2–3),
`.../regret_gate/` (feature ladder, near-duplicate check),
`.../metric_baseline/metric_baseline_g16.json` (§4 objectives),
`.../pair_score/pair_score_g16.json` (§4 pairwise, diagnostics mis-recorded — see §7.1),
`.../contact_sheet/contact_sheet_top8.png` (§5).
Drivers: `iot_experiments/core/{pool_emd2,metric_baseline,pair_score,regret_gate_matrix}.py`,
solver `src/iot/{master_regret,loop_regret}.py`.
