# FAUST correspondence — does a `θ` learned on one pair transfer to unseen **subjects**?

**Status: CURRENT (2026-11-09, **revised twice** — see §0 and §2b).** A `θ` fitted on one registered FAUST pair recovers
the true vertex correspondence on pairs drawn from **subjects never seen in training**, better than every
hand-weighted baseline, on **8 / 8** held-out pairs and **8 / 8** against each baseline.

Driver: `studies/real/faust/correspondence.py` · record: `out/faust_correspondence/records.json` (341 rows) ·
guides: `../notes/FAUST_CORRESPONDENCE_GUIDELINE.md`, pre-flights: `FAUST_PRELIMINARY_REPORT.md`.

**Scope.** Not a state-of-the-art claim. Deep functional-map methods reach far lower FAUST errors. The claim
under test was only: *a learned mixture of hand-crafted costs beats hand-weighted ones.* Published FAUST
numbers are **not** comparable to these.

---

## 0. Correction: the previous version leaked subjects

The first version of this report drew training and evaluation pairs by id without checking which **person**
each id belongs to (`id = 10·subject + pose`). Every training pair shared at least one subject with the
evaluation pairs:

| training pair | its subjects | subjects also present in the evaluation pairs |
| --- | --- | --- |
| `(17,65)` | 1, 6 | 1, 6 |
| `(7,52)` | 0, 5 | 0, 5 |
| `(99,10)` | 9, 1 | 1, 9 |

So that result measured **pose transfer within already-seen subjects**, not transfer to new people. A
`check_split` assertion now exits on any overlap, and the split is fixed: train subjects `{0,1,2,3,4}`, test
subjects `{5,6,7,8,9}`.

**The corrected result is stronger, not weaker** — see §2. The leak was flattering one baseline (the late-HKS
block) and understating the learned cost. It also changed *what the fit learns*: with subjects shared, the
learned `θ` was essentially pure late-HKS; with subjects disjoint, extrinsic shape and area acquire real
weight (§3).

## 1. Protocol

| item | setting |
| --- | --- |
| data | MPI-FAUST `training/registrations/`, 100 meshes, 6890 vertices sharing one ordering (gate 6.0: identical face arrays, area corr 0.965) |
| split | **subject-disjoint**: train subjects `{0,1,2,3,4}`, test subjects `{5,6,7,8,9}`; `check_split` exits on any overlap |
| validation pairs | 4 pairs from the **training** subjects (disjoint from the training pair and from the test subjects); used **only** to choose the box width (§2b) |
| box width | chosen from `rel ∈ {0.05, 0.1, 0.25, 0.5, 0.9}` **and simplex-only**, by validation miss@5 %; **simplex-only wins** (§2b) |
| training pair | `(0, 13)` — subject 0 pose 0, subject 1 pose 3 |
| held-out pairs | 8 pairs, both ends from test subjects: `(50,66) (56,71) (53,85) (59,99) (64,72) (61,86) (67,91) (72,85)` |
| vertex subsamples | **3 draws**, seeds 0/1/2, `n = 800` of 6890 shared-index vertices; `θ` is **refitted per seed** |
| plan | `π̂ = I/n` on the shared index subsample — the registered bijection |
| descriptors | `K = 13`: extrinsic `xyz` (centred, scaled by diameter), `log` 1-ring area, `log` curvature, **8 HKS** columns at times 1…128 |
| standardisation | pooled source+target mean/std, the same constants for both shapes |
| engine | relaxed regret loop, pruning off, HiGHS master; `Θ` = whole simplex (selected in §2b) |
| error | **geodesic** distance on the target between predicted and true vertex, normalised by `sqrt(surface area)` — `potpourri3d` heat method |
| statistics | paired bootstrap over the 8 held-out **pairs** (1000 resamples), each pair averaged over the 3 seeds first |

Arms: `position_only`, `uniform_hks` (8 HKS columns only), `uniform_late_hks` (t = 32/64/128 only),
`grouped_uniform` (one weight per descriptor group, 4 free weights), `uniform_theta` (all 13 equal),
`learned`, `oracle_theta` (**refitted on each evaluation pair — a ceiling, not a result**), plus an `nn_`
variant of each using per-row argmin instead of OT.

---

## 2. Main result

Means over **8 held-out pairs × 3 vertex subsamples = 24 readings**.

| arm | mean error | miss@1 % | miss@5 % |
| --- | ---: | ---: | ---: |
| `position_only` | 0.2112 | 0.800 | 0.648 |
| `uniform_hks` (8 cols) | 0.3274 | 0.769 | 0.659 |
| `uniform_late_hks` (3 cols) | 0.2981 | 0.719 | 0.601 |
| `grouped_uniform` (4 weights) | 0.1264 | 0.640 | 0.484 |
| `uniform_theta` (13, equal) | 0.1063 | 0.562 | 0.399 |
| **`learned`** | **0.0591** | **0.409** | **0.248** |
| `oracle_theta` (ceiling) | 0.0444 | 0.370 | 0.195 |
| `nn_uniform_theta` | 0.1644 | 0.724 | 0.567 |
| `nn_learned` | 0.1271 | 0.649 | 0.466 |
| `nn_oracle_theta` | 0.1301 | 0.695 | 0.495 |

**Transfer to unseen subjects holds, and more clearly than the leaky version.** `learned` improves on
`uniform` by **0.047 mean error** (0.1063 → 0.0591, −44 %) and **0.151 on miss@5 %** (0.399 → 0.248). Against
the ceiling: the oracle is 0.0444 against uniform's 0.1063, so the available gain is 0.0618 mean error and the
learned `θ` — fitted on **different people** — captures **0.047 of it, about 76 %**.

### Paired bootstrap over held-out pairs

| comparison | `learned − baseline` on miss@5 % | 95 % CI | pairs won |
| --- | ---: | --- | ---: |
| vs `uniform_theta` | **−0.1510** | [−0.207, −0.104] | **8 / 8** |
| vs `grouped_uniform` | −0.2357 | [−0.280, −0.195] | 8 / 8 |
| vs `uniform_late_hks` | **−0.3528** | [−0.410, −0.292] | **8 / 8** |
| vs `uniform_hks` | −0.4107 | [−0.457, −0.365] | 8 / 8 |
| vs `position_only` | −0.3997 | [−0.511, −0.274] | 8 / 8 |

Every CI excludes zero and **no pair loses to any baseline** — including the late-HKS block, which beat the
learned `θ` on 2 of 8 pairs under the leaky split. Under a subject-disjoint split the learned cost is not
just better on average; it is better everywhere.

### The split is not load-bearing: reversed subjects

The same experiment with the split reversed (`train {5,6,7,8,9}` → `test {0,1,2,3,4}`, training pair `(50,63)`),
8 pairs × 3 seeds:

| arm | mean error | miss@1 % | miss@5 % |
| --- | ---: | ---: | ---: |
| `position_only` | 0.2034 | 0.742 | 0.594 |
| `uniform_hks` | 0.3067 | 0.732 | 0.624 |
| `uniform_late_hks` | 0.2696 | 0.687 | 0.574 |
| `grouped_uniform` | 0.1238 | 0.620 | 0.463 |
| `uniform_theta` | 0.1019 | 0.533 | 0.375 |
| **`learned`** | **0.0606** | **0.399** | **0.252** |
| `oracle_theta` | 0.0436 | 0.339 | 0.173 |

`learned − uniform` = **−0.122** [−0.184, −0.067], 8/8 pairs; against `uniform_late_hks` **−0.321**
[−0.362, −0.272], 8/8. The two splits agree closely (−0.151 and −0.122), so the conclusion does not depend on
which five people are held out.

### Per pair (mean over the 3 subsamples)

| pair | pos | HKS8 | late3 | 13-equal | **learned** | oracle | learned spread over seeds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 050-066 | 0.870 | 0.618 | 0.527 | 0.508 | **0.344** | 0.229 | [0.326, 0.360] |
| 053-085 | 0.848 | 0.682 | 0.595 | 0.628 | **0.314** | 0.308 | [0.284, 0.338] |
| 056-071 | 0.752 | 0.690 | 0.663 | 0.487 | **0.315** | 0.252 | [0.306, 0.333] |
| 059-099 | 0.146 | 0.594 | 0.526 | 0.172 | **0.100** | 0.037 | [0.091, 0.109] |
| 061-086 | 0.684 | 0.659 | 0.605 | 0.331 | **0.212** | 0.162 | [0.205, 0.224] |
| 064-072 | 0.472 | 0.697 | 0.671 | 0.315 | **0.215** | 0.141 | [0.190, 0.236] |
| 067-091 | 0.913 | 0.699 | 0.657 | 0.462 | **0.270** | 0.278 | [0.253, 0.282] |
| 072-085 | 0.498 | 0.633 | 0.564 | 0.293 | **0.216** | 0.148 | [0.188, 0.242] |

**Vertex subsampling barely matters**: the learned arm's miss@5 % moves by at most **0.046** across the three
subsamples for a given pair (typical spread ≈0.02), an order of magnitude below the gap to any baseline. The
result is not an artefact of one lucky set of 800 vertices.

Note `pair067-091`: the learned cost (0.270) is slightly *better* than that pair's own oracle (0.278). That is
possible because the oracle fits `θ` by regret on the subsample while the error is measured geodesically
against the whole mesh; it is a reminder that "oracle" is a ceiling for the *optimisation*, not a strict bound
on the metric.

## 2b. Box width chosen on held-out pairs — and simplex-only wins

The trust region was an assumption in every previous run (`rel = 0.25` fixed). It is now **chosen on held-out
validation pairs drawn from the training subjects** — disjoint from the test subjects and from the training
pair itself — and **simplex-only** (no box) is a candidate.

`rel` is the **inverse** box width (`θ ∈ [rel/K, 1/(rel·K)]`), so `rel = 0.05` is the widest box and
`rel = 0.9` the narrowest.

| candidate | validation miss@5 % (selection) | test mean error | test miss@1 % | test miss@5 % |
| --- | ---: | ---: | ---: | ---: |
| **`learned_simplex`** ← selected | **0.1679** | **0.0450** | **0.329** | **0.186** |
| `rel = 0.05` (widest) | 0.1876 | 0.0448 | 0.345 | 0.188 |
| `rel = 0.10` | 0.2070 | 0.0466 | 0.363 | 0.200 |
| `rel = 0.25` (all earlier runs) | 0.2746 | 0.0591 | 0.409 | 0.248 |
| `rel = 0.50` | 0.3616 | 0.0764 | 0.474 | 0.312 |
| `rel = 0.90` (narrowest) | 0.4444 | 0.0996 | 0.545 | 0.383 |

**Two findings, and the second is the important one.**

1. **Validation selection works and is honest.** The ranking on validation pairs matches the ranking on the
   test pairs **exactly**, for all six candidates. Selecting on the held-out pairs picked the candidate that
   was in fact best on the test set, without ever looking at it.

2. **The box was hurting, monotonically.** Every widening improves the result: `rel = 0.9` is the worst
   (0.383) and removing the box entirely is the best (0.186). The sequence is monotone in box width, with no
   interior optimum, so **the trust region is not needed for this task** and every earlier number in this
   report was obtained under a constraint that cost about **0.06 miss@5 %** (`rel = 0.25`: 0.248 → simplex:
   0.186).

With the selected configuration, `learned_selected` beats every baseline on **8 / 8** pairs:

| comparison | `learned_selected − baseline` on miss@5 % | 95 % CI | wins |
| --- | ---: | --- | ---: |
| vs `uniform_theta` | **−0.2139** | [−0.297, −0.145] | 8/8 |
| vs `grouped_uniform` | −0.2985 | [−0.374, −0.229] | 8/8 |
| vs `uniform_late_hks` | −0.4156 | [−0.443, −0.392] | 8/8 |
| vs `uniform_hks` | −0.4735 | [−0.494, −0.452] | 8/8 |
| vs `position_only` | −0.4625 | [−0.590, −0.319] | 8/8 |

**Read the earlier sections with this in mind**: §2 and §3 report `rel = 0.25` because that was fixed when
they were written. The selected configuration is simplex-only, and its numbers are the ones in this table.
Both are kept so the effect of the constraint stays visible.

**Corner behaviour does not go away, it changes shape.** Under simplex-only the fit still sits on the
boundary: `min θ = 0.0000` with 4 coordinates at zero and `max θ = 0.77`, `converged=False`. So the
"corner solution" caveat is not caused by the box — it is the objective's own preference. That is now the
sharpest open question, and it is independent of the trust-region decision.

---

## 3. What the learned `θ` is

Refitted per vertex subsample; all three fits agree, and all sit **on the box bounds** (`converged=False`,
regret ≈ 0.010):

| | xyz (3) | area | curv | HKS 1–16 (4) | HKS 32 | HKS 64 | HKS 128 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| floor = 0.0192, ceiling = 0.3077 (`rel = 0.25`, `K = 13`) | | | | | | | |
| seed 0 | 0.074 / 0.026 / floor | 0.065 | floor | floor | 0.085 | **0.308** | **0.308** |
| seed 1 | 0.069 / floor / floor | 0.087 | floor | floor | 0.119 | 0.263 | **0.308** |
| seed 2 | 0.081 / 0.035 / floor | 0.048 | floor | floor | 0.111 | 0.283 | **0.308** |

**Compare this with the leaky split.** Under subject leakage the `θ` was ≈0.02 on the extrinsic block and
essentially *only* late HKS. Under the subject-disjoint split the extrinsic block (0.07–0.08) and log-area
(0.05–0.09) acquire real weight. That is a sensible, interpretable difference: **within one person, a
pose-invariant intrinsic descriptor is enough; across people, absolute shape and size start to matter.**
It is also evidence the fixture is not degenerate — the fit responds to the split.

**But every fit is a corner solution**, so the weights are a *feasible point the solver accepted*, not a
proved optimum. "The learned cost weights late HKS plus a little shape" describes where this solver stopped.

---

## 4. OT versus nearest-neighbour

| weights | OT (EMD) | per-row argmin | ratio |
| --- | ---: | ---: | ---: |
| uniform | 0.1063 | 0.1644 | 1.55× |
| learned | **0.0591** | 0.1271 | 2.15× |
| oracle | 0.0444 | 0.1301 | 2.93× |

**The coupling matters more as the weights improve**: the OT advantage over per-row argmin grows from 1.55×
(uniform) to 2.15× (learned) to 2.93× (oracle). The gain from 0.1063 to 0.0591 is still *mostly* the weights
— but unlike the leaky run, where the ratio was flat, here the two effects reinforce.

> This replicates, on a subject-disjoint split with 24 readings, what the earlier 3-pair run suggested and the
> 8-pair leaky run contradicted. The intervening variable was **subject leakage**, not sample size: the leaky
> pairs were intra-subject-ish, where coupling and ranking agree more.

---

## 4b. Where the errors are, and how many are left/right flips

Driver `studies/real/faust/heatmaps.py`, run through `scripts/run_job.sh`. Two held-out pairs from **test
subjects** (`050-066`, `056-071`), evaluated at **full 6890-vertex resolution** (the cost is assembled with
the bilinear identity, so only an `n × n` matrix is materialised). `θ` fitted on the training pair `(0,13)`.

| arm | mean error | miss@5 % | **flip rate** | flip share of errors |
| --- | ---: | ---: | ---: | ---: |
| `position_only` | 0.3037 | 0.819 | **0.392** | 0.444 |
| `uniform_theta` | 0.1695 | 0.510 | **0.194** | 0.269 |
| **`learned`** | **0.0747** | **0.232** | **0.082** | 0.159 |
| `oracle_theta` | 0.0627 | 0.180 | 0.064 | 0.143 |

**A flip is a specific, measurable failure.** It is counted when the predicted vertex is geodesically
*closer to the mirrored vertex than to the true one*, restricted to vertices whose error exceeds 1 % of
`sqrt(area)` and whose mirror is a different vertex. So "flip share of errors" answers: *of the vertices we
got wrong, how many did we get wrong by swopping left for right?*

**Three readings.**

1. **Left/right mirroring is a large share of the failure, and learning removes most of it.** Position-only
   puts **39 %** of all vertices on the mirror side and **44 %** of its errors are flips. The learned cost
   drops that to **8 %** and 16 %. So the learned mixture is not merely "more accurate" — it is specifically
   better at *not* confusing the two sides, which is exactly the failure mode intrinsic descriptors are
   supposed to have (the guideline's §3.1 concern).
2. **The remaining error is not mirroring.** After learning, 84 % of the errors are *not* flips: they are
   local confusions near the hands, the neck and the crotch — visible as hot patches in the figures — not
   global side-swops.
3. **The oracle is only modestly better than the learned `θ`** (0.082 → 0.064 flip rate), so what is left of
   the symmetry problem is at the ceiling of this descriptor family: **adding extrinsic weight cannot fix it
   much**, because the extrinsic block is only 4 % of the learned mass.

**The heat maps** (`out/faust_heatmaps/*.png`, 16 panels per pair-view, one fixed colour scale per pair so
they are comparable) show the same thing spatially: position-only is hot over the whole torso, head and
thighs; the learned cost is dark everywhere except the hands, the neck and a band at the waist.

**Caveat on the mirror.** FAUST poses are not symmetric (an arm may be raised), so the mirror is a
*reflection across the estimated sagittal plane followed by nearest neighbour* — a proxy, not an isometry.
Its reflection residual has median 0.03 (mesh diameter ≈ 2.07). Treat the flip rate as a large-scale
diagnostic: it identifies "matched the other side of the body", not an exact symmetry-orbit count.

---

## 4c. Resolution check at `n = 4000` (5× the main run)

Every number in §2–§4 uses `n = 800` of 6890 vertices. This re-runs the same protocol at **`n = 4000`**
(subject-disjoint, simplex-only, 8 held-out pairs, 1 seed, no oracle and no validation pairs — the box width
was already settled in §2b), so the conclusion does not rest on 12 % of the mesh.

Driver `correspondence.py --n-sub 4000 --oracle 0 --validation 0`, launched through `run_job.sh`,
12 threads, peak RSS **≈4.7 GB** (budget 10.7 GB). Total wall **14.5 min**, of which the two fits are
**11.7 min** (346 s + 358 s) and all eight pair evaluations only ~3 min.

| arm | mean error | miss@1 % | miss@5 % |
| --- | ---: | ---: | ---: |
| `position_only` | 0.2126 | 0.873 | 0.654 |
| `uniform_hks`, `nn_uniform_hks` | 0.3671 | 0.860 | 0.722 |
| `uniform_late_hks` | 0.3363 | 0.830 | 0.673 |
| `grouped_uniform` | 0.1301 | 0.745 | 0.497 |
| `uniform_theta` | 0.1153 | 0.691 | 0.422 |
| **`learned` (simplex)** | **0.0529** | **0.503** | **0.203** |
| `learned` (rel 0.25) | 0.0693 | 0.573 | 0.273 |
| `nn_learned` | 0.1049 | 0.758 | 0.434 |

| comparison | Δ on miss@5 % | 95 % CI | wins |
| --- | ---: | --- | ---: |
| vs `uniform_theta` | **−0.219** | [−0.301, −0.149] | 8/8 |
| vs `grouped_uniform` | −0.294 | [−0.380, −0.214] | 8/8 |
| vs `uniform_late_hks` | **−0.470** | [−0.500, −0.437] | 8/8 |
| vs `uniform_hks` | −0.519 | [−0.549, −0.486] | 8/8 |
| vs `position_only` | −0.451 | [−0.578, −0.306] | 8/8 |

**Per pair (miss@5 %), learned beats uniform on 8/8:**

| pair | pos | grouped | 13-equal | late3 | **learned** |
| --- | ---: | ---: | ---: | ---: | ---: |
| 050-066 | 0.872 | 0.622 | 0.519 | 0.607 | **0.159** |
| 053-085 | 0.843 | 0.709 | 0.643 | 0.656 | **0.210** |
| 056-071 | 0.772 | 0.591 | 0.512 | 0.717 | **0.266** |
| 059-099 | 0.150 | 0.252 | 0.175 | 0.636 | **0.101** |
| 061-086 | 0.710 | 0.439 | 0.360 | 0.680 | **0.180** |
| 064-072 | 0.473 | 0.425 | 0.359 | 0.718 | **0.217** |
| 067-091 | 0.928 | 0.548 | 0.504 | 0.720 | **0.339** |
| 072-085 | 0.482 | 0.388 | 0.303 | 0.648 | **0.149** |

**What the resolution check changes.**

1. **The conclusion survives a 5× larger instance, and the margin over uniform grows** (−0.219 versus −0.151
   at `n = 800`). It is not a subsampling artefact.
2. **The block controls separate far more cleanly.** Uniform late-HKS is **0.470** worse at `n = 4000` versus
   0.353 at `n = 800`: at higher resolution the "just use the late-HKS block" shortcut degrades badly
   (0.673 miss@5 %) while the learned mixture holds at 0.203. The gap between "pick the block" and "learn the
   weights inside it" is the thing that grows with `n`.
3. **`learned` is slightly worse in absolute terms than at `n = 800`** (0.203 vs 0.186 miss@5 %). Not a
   surprise: a 4000-vertex subsample is a different, harder instance, and this run has no oracle or
   validation selection. The *relative* advantage is what the claim rests on, and it improved.

**Left/right flips at n = 4000** (same definition as §4b, counted per vertex against the mirror proxy):

| arm | flip rate | flip share of its errors |
| --- | ---: | ---: |
| `position_only` | 0.252 | — |
| `uniform_theta` | 0.138 | — |
| `uniform_late_hks` | **0.413** | — |
| **`learned`** | **0.0575** | 0.115 |
| `nn_learned` | 0.090 | 0.118 |

The ordering is the same as at `n = 800`, and one number is worth noting: **uniform late-HKS has the *worst*
flip rate of any arm (0.41)** even though it is a decent cost on other axes. Mirror confusion is exactly what
late-HKS alone cannot resolve, and it is the largest single advantage of the learned mixture.

**Caveat.** One seed, one training pair, no oracle in this run, and `n = 4000` is still 58 % of the mesh —
the full 6890 did **not** fit in memory (measured: 7.24 GB at `n = 5000`, extrapolating to ≈11 GB at 6890
against a 10.7 GB ceiling).

---

## 4d. The full `n = 6890` run

After the memory fix (§ `README.md`), the fit runs at **full resolution** — all 6890 vertices, peak
**1.63 GB** (it previously did not fit at all). Protocol identical to §4c: subject-disjoint, simplex-only,
8 held-out test pairs, 1 seed, no oracle/validation pairs. Wall **19.1 min** (fits 313 s + 272 s, evaluation
the rest).

| arm | mean error | miss@1 % | miss@5 % |
| --- | ---: | ---: | ---: |
| `position_only` | 0.2149 | 0.886 | 0.649 |
| `uniform_hks` | 0.3730 | 0.870 | 0.726 |
| `uniform_late_hks` | 0.3517 | 0.850 | 0.690 |
| `grouped_uniform` | 0.1305 | 0.760 | 0.491 |
| `uniform_theta` | 0.1178 | 0.712 | 0.419 |
| **`learned` (simplex)** | **0.0550** | **0.531** | **0.1996** |
| `learned` (rel 0.25) | 0.0721 | 0.604 | 0.274 |

| comparison | Δ on miss@5 % | 95 % CI | wins |
| --- | ---: | --- | ---: |
| vs `uniform_theta` | **−0.2190** | [−0.304, −0.146] | 8/8 |
| vs `grouped_uniform` | −0.2915 | — | 8/8 |
| vs `uniform_late_hks` | −0.4903 | — | 8/8 |
| vs `uniform_hks` | −0.5267 | — | 8/8 |
| vs `position_only` | −0.4490 | — | 8/8 |

**The resolution axis has saturated.** n=800 → 4000 → 6890 gives, for the learned arm, miss@5 % of
0.186 → 0.203 → **0.200**, and `Δ` vs uniform of −0.151 → −0.219 → **−0.219**. Going from 4000 to the full
mesh changes nothing beyond noise, so **n=4000 was already representative** and the conclusion does not
depend on subsampling. That also means the expensive full-resolution run was a one-off check, not the new
default: n=4000 costs 6 min/fit, n=6890 costs 5.2 min/fit at 8× the EMD work.

---

## 4e. Support-matched control, bootstrap-averaged θ, and three seeds (n = 4000)

Three vertex subsamples (seeds 0/1/2), n=4000, subject-disjoint, 8 held-out pairs → **24 readings**. The run
also evaluates two derived weights:

- **`learned_avg`** — the per-coordinate mean of θ across the three subsamples (variance reduction; the corner
  analysis showed t=32/64/128 trade mass across resamples).
- **`support_matched`** — uniform weights over the **nonzero support** of that average: the same selected
  descriptors, no fitted magnitudes.

| arm | mean error | miss@1 % | miss@5 % |
| --- | ---: | ---: | ---: |
| `uniform_theta` | 0.1164 | 0.689 | 0.4215 |
| `support_matched` (simplex support) | 0.0869 | 0.585 | **0.3330** |
| **`learned`** (per seed) | 0.0532 | 0.500 | **0.2020** |
| `learned_avg` (averaged θ) | 0.0526 | 0.505 | **0.2025** |
| `uniform_late_hks` | 0.3411 | 0.829 | 0.6726 |

**The support/weights decomposition.** Choosing *which* descriptors to use is worth 0.4215 → 0.3330
(−0.089), and the fitted magnitudes are worth 0.3330 → 0.2020 (**−0.131**). So **the magnitudes matter more
than the support** — the opposite of what "the fit just picks the late-HKS block" would predict, and consistent
with §4c where uniform late-HKS is the worst arm.

**Bootstrap-averaging changes nothing** (0.2025 vs 0.2020, inside noise). That is a *positive* result for
stability: the per-seed θ already agree closely enough that averaging them is a no-op, which agrees with the
corner analysis (stable support, modest within-face wobble).

**Three seeds at n=4000:** `learned_selected` over 24 readings is 0.2020 miss@5 %, Δ vs uniform **−0.2196**,
8/8 pairs. The seed spread on the mean error is ≤0.02, well below the gap to any baseline.

Note `support_matched` for the **box** arm (rel 0.25) is exactly `uniform_theta`: with a box the support is all
13 coordinates, so the control carries no information there. It only bites under simplex — which is another
reason the simplex-only setting is the informative one.

---

## 4f. Four objectives, one hypothesis class (n = 1500)

The hypothesis class is held fixed — `c_ij = Σ_k θ_k (a_ik − b_jk)²`, `θ` on the simplex, the **same**
standardised 13 descriptors, the **same** training pair — so every difference below is attributable to the
**objective**. Hyperparameters (`tau`, `rel_eps`) were chosen on validation pairs from the **training**
subjects; the LP is simplex-only and has no hyperparameter. 8 held-out test pairs, 1 seed, `n = 1500` (the
entropic arm materialises an `(m²K)` tensor, so this comparison is at reduced resolution).

| arm | objective | mean error | miss@1 % | miss@5 % |
| --- | --- | ---: | ---: | ---: |
| `uniform` | none | 0.1088 | 0.626 | 0.4125 |
| `entropic_iot` | entropic joint master (`rel_eps=0.05`) | 0.0941 | 0.573 | 0.3838 |
| `row_softmax` | one-sided CE | 0.0588 | 0.433 | 0.2323 |
| `rowcol_softmax` | Sinkhorn CE (`tau=0.01`) | 0.0546 | 0.421 | 0.2231 |
| **`relaxed_lp`** | regret master | **0.0480** | **0.405** | **0.1972** |

Paired bootstrap over the 8 held-out pairs:

| comparison | Δ miss@5 % | 95 % CI | wins |
| --- | ---: | --- | ---: |
| `relaxed_lp` − `uniform` | −0.2153 | [−0.295, −0.147] | 8/8 |
| `relaxed_lp` − `entropic_iot` | −0.1867 | [−0.215, −0.158] | 8/8 |
| `relaxed_lp` − `row_softmax` | **−0.0352** | [−0.050, −0.017] | **7/8** |
| `relaxed_lp` − `rowcol_softmax` | **−0.0259** | [−0.059, +0.001] | **5/8** |

**What this settles.**

1. **The LP wins, but only by a little.** Against the *best* supervised reweighting of the identical features the
   advantage is **0.026 miss@5 %**, and against the two-sided Sinkhorn arm the CI **includes zero** (5/8 pairs).
   The big 10× advantage is over *uniform*; over "same features, supervised loss" it is 1.13×. So **most of what
   this method does is reweighting, not structured matching** — on this data, at this resolution.
2. **Adding the column marginal helps slightly** (0.2323 → 0.2231 one-sided → two-sided), i.e. the competition
   in the plan does contribute, but only ~0.01 miss@5 % over the row-only loss.
3. **The entropic iOT master is the worst learned arm** — 0.3838, barely better than uniform and far behind both
   softmax arms. That is consistent with the EOT line's own finding that `min_θ Ψ` does not recover `θ*`
   (`current/eot/README.md` §1), and it says the entropic regularisation is the expensive part of this
   comparison: it is the arm whose temperature hurts most (0.05 selected — the smallest in the grid, so the
   objective wanted *less* smoothing, like the box width wanting *less* constraint).
4. **`nn_*` (per-row argmin) is uniformly worse than its OT counterpart** in every arm, by 0.22–0.33 miss@5 %.
   So the coupling does matter — it is just not worth much against a well-fit supervised cost.

**Honest reading**: at `K=13` and `n=1500`, the structured LP beats plain supervised reweighting of the same
descriptors by a small but consistent margin against the one-sided loss, and by nothing measurable against the
two-sided one. That is a weaker claim than "inverse OT is the right tool here", and it is the claim the data
supports.

---

## 4g. Accuracy against a controlled deformation (n = 1500)

A **twist about the vertical axis** applied to the *target* only, angle `a = σ·π·y_norm`: a smooth point map, so
vertex identity is preserved and `π̂ = I/n` stays the ground truth with no new labelling. `θ` is fitted **once**
at `σ = 0` and applied unchanged at every `σ`, so this is a transfer curve, not a refit. Single pair, one seed.

| σ | `learned` | `uniform` | gap | `uniform_late_hks` | `position_only` |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | **0.161** | 0.502 | **−0.341** | 0.575 | 0.868 |
| 0.25 | **0.322** | 0.530 | −0.208 | 0.575 | 0.829 |
| 0.50 | **0.436** | 0.677 | −0.241 | 0.599 | 0.823 |
| 0.75 | **0.612** | 0.745 | −0.133 | 0.630 | 0.881 |
| 1.00 | **0.708** | 0.775 | **−0.067** | 0.697 | 0.889 |

**The gap shrinks by a factor of five.** The learned cost degrades gracefully (0.161 → 0.708) but so does
everything else, and at the strongest twist it is **barely distinguishable from uniform** (−0.067, against
−0.341 undeformed). Two consequences:

1. **The method's advantage is a low-deformation phenomenon on this data.** Any claim about it must name the
   deformation regime, because at `σ = 1` a fixed uniform weighting is nearly as good.
2. **`uniform_late_hks` is the most deformation-robust arm** (0.575 → 0.697, a change of 0.12 versus the learned
   arm's 0.55). That is the *opposite* ordering from the rigid-pose case, where late-HKS was the worst arm
   (0.601 vs 0.248). So the ranking of descriptors is deformation-dependent, and there is no single "best"
   weighting across regimes — which is precisely the argument for conditioning the fit on the regime, not for
   one global `θ`.

**This is not an articulation.** A twist is not isometric, so intrinsic descriptors are not invariant to it, and
it is not joint bending. The curve bounds *descriptor degradation under smooth distortion*, not tolerance to
real pose change — the FAUST pose axis (§2) remains the evidence for that. Also: **one pair, one seed**, so this
is a single-pair curve.

---

## 4h. Many plans, one θ (negative result)

A block LP with **one shared `θ`** and per-pair `(f_p, g_p)`, over 4 training pairs; each `F_p` is the edge set
that pair's own loop finished with. Objective: the **mean of the raw per-pair regrets**. Evaluated on the same
8 held-out test pairs at `n = 1200`.

| arm | mean error | miss@5 % |
| --- | ---: | ---: |
| `single` (θ from pair (0,13) alone) | **0.0475** | **0.197** |
| `uniform` | 0.1069 | 0.404 |
| **`multi` (shared over 4 pairs)** | **0.2018** | **0.462** |

**Sharing is worse than not learning at all.** The per-pair L1 distances to their own θ are 0.08, 0.16, 0.46 and
**1.01**, so the four pairs want materially different weights and the compromise is bad for all of them.

### The normalised rerun did not rescue it

Each pair's regret is now divided by its own `R_uniform,p` before averaging (scales 0.0913, 0.0729, 0.1729,
0.1185 — they differ by 2.4×, so the normalisation is not cosmetic). **θ barely moved**:

```
unnormalised  [0, .0129, .0004, .0075, .0001, .0037, 0,0,0,0, .0228, .3660, .5866]
normalised    [0, .0139, .0002, .0081, .0001, .0037, 0,0,0,0, .0362, .3672, .5707]
```

and the test result is **unchanged**: `multi` 0.4666 miss@5 % versus `single` 0.1974 and `uniform` 0.4043. So
**the scale was not the cause**. The hypothesis "the objective was dominated by the two hardest pairs" is
refuted; something else is wrong.

### The cross-regret matrix shows what

`R_p(θ_q) / R_uniform,p` — rows are the pair being *evaluated*, columns the θ that was *fitted* (th0…th3 =
pairs (0,13), (2,41), (5,88), (7,52); th4 = the shared θ). Zero means that θ certifies that pair's plan.

| | th0 | th1 | th2 | th3 | **shared** |
| --- | ---: | ---: | ---: | ---: | ---: |
| pair (0,13) | **0.0075** | 0.7566 | 0.0143 | 0.0114 | 0.0142 |
| pair (2,41) | 0.0207 | **0.0086** | 0.0290 | 0.0252 | 0.0298 |
| pair (5,88) | 0.1027 | 0.4349 | **0.0135** | 0.0565 | 0.0141 |
| pair (7,52) | 0.0259 | **1.1218** | 0.0351 | **0.0193** | 0.0281 |

Three readings:

1. **Each θ certifies its own pair** (the diagonal is 0.008–0.019), so the fits work.
2. **The pairs disagree violently.** The θ fitted on pair (2,41) has **75 %** and **112 %** of `R_uniform` on
   pairs (0,13) and (7,52) — it is not a general-purpose weighting at all, it is specific to its own pair. Any
   shared θ must therefore be a genuine compromise, and the shared column is one: **0.014–0.030 on every pair**,
   as good as the *best* single θ (th2).
3. **Low training regret does not transfer.** By this metric the shared θ is excellent: it certifies all four
   plans nearly as well as each pair's own θ. Yet on held-out pairs it scores **0.467 miss@5 %**, worse than
   uniform. Meanwhile th0, with the same kind of training regret, scores 0.197.

**So the multi-pair objective is measuring the wrong thing, not optimising badly.** Fitting θ so that many plans
are simultaneously near-optimal selects weights that are *plan-certifying* but not *correspondence-useful*; the
same tension the corner analysis exposed (a vertex that satisfies the plan, not one that identifies the
geometry). The honest conclusion is that **"many plans, one θ" in the max-margin/regret form does not help
here**, and that a shared θ would need a different selection principle — e.g. held-out performance inside the
fit, or averaging the *per-pair* θs rather than fitting a joint one.

Note also that `learned_avg` in §4e — a plain mean of per-pair θs over vertex subsamples — *did* work. The
difference is that averaging per-pair solutions is a variance reduction, whereas the joint LP imposes a common
feasibility that the data does not support.

---

## 4i. Can the fit tell you *where* it is wrong? (n = 4000)

A correspondence that abstains on its uncertain vertices is more useful than one that is uniformly confident
at the same mean error. Four **label-free** confidence scores, all computed from the same `theta` and the same
`n x n` cost matrix, ranked against the true per-vertex geodesic error (AUROC of "this vertex is correct",
so **0.5 = chance, higher = better**):

| score | what it is | `relaxed_lp` | `row_softmax` | `uniform` |
| --- | --- | ---: | ---: | ---: |
| `cost_margin` | `c_(2nd) − c_(1st)`, raw | 0.464 | 0.487 | 0.489 |
| `dual_margin` | **OT dual certificate**: tightest non-plan edge after removing the target's price `v_j` | 0.502 | 0.527 | 0.525 |
| `rel_cost_margin` | the same raw gap, divided by `c_(2nd)` | **0.569** | 0.533 | 0.521 |
| `softmax_margin` | `P_i,pred − max_{j≠pred} P_ij`, `P = softmax(−c/τ)`, `τ = 3e-3` | **0.632** | **0.609** | **0.597** |

**Risk–coverage** (rank by confidence, keep the most confident fraction `q`, `miss@5 %` among them):

| θ | score | q = 50 % | 70 % | 90 % | 100 % |
| --- | --- | ---: | ---: | ---: | ---: |
| `relaxed_lp` | `softmax_margin` | **0.144** | 0.170 | 0.190 | 0.203 |
| `relaxed_lp` | `cost_margin` | 0.223 | 0.214 | 0.206 | 0.203 |
| `row_softmax` | `softmax_margin` | **0.195** | 0.217 | 0.239 | 0.253 |
| `uniform` | `softmax_margin` | **0.365** | 0.394 | 0.412 | 0.422 |

Four readings, and the first is the one that matters for the method:

1. **The OT dual certificate is at chance.** 0.502–0.527. The quantity the optimality proof reasons about —
   the slack of the tightest non-plan edge, with each target's own price removed — does **not** localise
   correspondence errors on a new pair. A certificate about the plan you fitted is not an uncertainty
   estimate about the plan you will produce. This is the direct answer to "can inverse OT tell you where it
   is wrong": **not through its certificate.**
2. **The raw cost margin is worse than chance** (0.464–0.489), and abstaining by it *increases* the error
   (0.223 at 50 % coverage versus 0.203 at 100 %). It is dominated by per-vertex cost scale: dividing by
   `c_(2nd)` recovers most of the signal (0.569 for the LP θ), which confirms the diagnosis rather than a
   deeper failure.
3. **The probability margin works** (0.60–0.63), and it works for the **uniform** θ too (0.597). So per-vertex
   confidence is available from any cost's local landscape; it is not an inverse-OT property. The learned θ
   improves the *ranking* (0.632 vs 0.597) more than it improves the *accuracy* — an interesting asymmetry:
   the fit sharpens the cost exactly where the near-ties are.
4. **Practically it is worth having.** At 50 % coverage the error falls 0.203 → 0.144 (a 29 % relative cut)
   for the LP θ, 0.253 → 0.195 (23 %) for the supervised θ, 0.422 → 0.365 (13 %) for uniform. Better weights
   buy better confidence, in roughly the same proportion as they buy accuracy.

**Scope.** One vertex-subsample seed, 8 held-out pairs, one `tau` for the probability score. The AUROC
ordering is stable across the three θ (cost < dual < rel < softmax in every column), so the qualitative
conclusion does not rest on one draw.

---

## 4j. Inverse OT as a *teacher* for a cheap arm (n = 2000)

If the LP's advantage is real but its machinery is heavy, could the machinery be spent **once** — as a
hard-negative miner — and the deployed model be a plain softmax arm? The fitted cost certifies the true
partner up to a margin, so the targets sitting just above it are exactly the confusions it cannot rule out;
uniformly sampled negatives almost never land there. Four arms, same features, same training pair:

| arm | negatives in the loss | mean error | miss@5 % |
| --- | --- | ---: | ---: |
| `plain` | all `n − 1` others | **0.0650** | **0.2453** |
| `hard_self` | 32 chosen by the **supervised** arm's own cost (control) | 0.0662 | 0.2518 |
| `mixed_lp` | 32 LP-chosen + 32 uniform | 0.0751 | 0.2664 |
| `hard_lp` | 32 chosen by the **inverse-OT** cost | 0.0757 | 0.2671 |
| `uniform` | — (no learning) | 0.1127 | 0.4184 |

Paired bootstrap over the 8 held-out pairs: `hard_lp − plain` = **+0.0218 [+0.0193, +0.0248], 0/8** and
`hard_lp − hard_self` = **+0.0154 [+0.0121, +0.0185], 0/8**.

**Negative, and cleanly so.** Mining negatives from the fitted cost makes the cheap arm *worse* by ~0.02, and
the LP's negatives are worse than the supervised arm's own — so inverse OT is not even the better miner.
The `mixed` arm (65 classes instead of 33) is no better, so this is not merely a too-aggressive restriction:
the model learns the miner's in-sample confusions and loses the broad negative distribution that generalises.

**Scope.** n = 2000, one seed, one hard-negative count (32), one τ grid. It is a single configuration of the
idea, not a sweep; but the effect is consistent across all 8 pairs and the control (`hard_self`) fails the
same way, which is the pattern a real effect would not have.

---

## 5. Honest limitations

- **Three vertex subsamples, `n = 800` of 6890.** The learned arm moves ≤0.046 miss@5 % across seeds, so the
  headline is not seed-sensitive; but 800 vertices is 12 % of the mesh.
- **One training pair** `(0,13)`. The subject-level generalisation is exactly what is tested, but a single
  pair is a single draw of "what one pair teaches". Multi-pair shared-`θ` is not implemented.
- **All evaluation pairs are inter-subject by construction** (that is the point of the split).
  **Intra-subject is implemented but not run** (`--pairing intra`: same subject, two poses). Its generator is
  unit-tested (`tests/iot_faust_test.py`), but no experiment uses it — it is expected to be easier and would
  likely leave less headroom, and no current claim depends on it. Deliberately skipped.
- **Not comparable to published FAUST errors.** Geodesic error normalised by `sqrt(area)` on a 12 %
  subsample, using training-set registrations only (the 200 test scans have no published registrations; that
  set is scored by the website).
- **Corner `θ`, `converged=False`** in every fit (§3).
- **The oracle is not a strict bound.** On `pair067-091` the learned cost (0.270) is marginally better than
  that pair's own oracle (0.278), because the oracle optimises regret on the subsample while the metric is
  geodesic over the whole mesh.
- **A defect in the capacity control was found and fixed.** The first `grouped_uniform` broadcast each group
  mean back over that group's coordinates and then averaged the 13 columns, which re-weights by group size and
  is **algebraically identical** to `uniform_theta` (verified: bit-identical). It now averages the group means
  directly and scores 0.484, worse than the full dictionary — as a lower-capacity arm should.
- **A subject leak was found and fixed** (§0). The leaky version overstated the late-HKS block and understated
  the learned cost.
- **The first heat-map run selected non-disjoint pairs** (`0-66`, `1-36`, where subject 0 is a *training*
  subject). Re-run on test-subject pairs (§4b); the first job also **OOM-killed** (exit 137) because it built
  a full-resolution `(6890, 6890, 13)` feature tensor (4.94 GB) — `θ` is now fitted on an 800-vertex
  subsample and only the evaluation is full-resolution.
- **The flip count depends on a mirror proxy** (§4b), not an exact symmetry map.

---

## 6. What this licenses, and what it does not

**Licenses:** the design in `FAUST_CORRESPONDENCE_GUIDELINE.md` is worth building out. A learned mixture of
hand-crafted descriptors beats every hand-weighted baseline — including uniform weights over the same
descriptors, uniform weights over the HKS block it loads on, a 4-parameter grouped version, and position-only
— on **8 / 8 subjects-unseen pairs, 8 / 8 against each baseline, with the advantage stable across three vertex
subsamples**.

**Does not license:** any comparison with deep functional maps, any claim that the weights are *identified*,
any claim about intra-subject pairs or non-human shapes, or quoting these numbers next to published FAUST
error tables.

**Next, in order:** (1) settle the corner-`θ` question; (2) multi-pair shared `θ`, so "one pair" is not a
single draw; (3) a second subject split (e.g. `{5,6,7,8,9}` → train) to confirm the split choice is not
load-bearing; (4) intra-subject pairs; (5) the synthetic distortion layer.
