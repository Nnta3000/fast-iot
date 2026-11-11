# FAUST preliminary tests — do the pre-flight gates pass?

**Status: CURRENT (2026-11-09).** Preliminary only. Three gates, three shape pairs, 800 of 6890 vertices.
No `θ` has been trained on one pair and tested on another yet, so **nothing here is an experiment result**;
it is the answer to "is this direction worth building?".

Driver: `studies/real/faust/{data_check,gates}.py` · records:
`out/faust_data_check/records.json` (7 rows), `out/faust_gates/records.json` (24 rows) · guide:
`../notes/FAUST_CORRESPONDENCE_GUIDELINE.md`.

---

## 1. Gate 6.0 — is `π̂ = I/n` really the correspondence?

Everything rests on this: if the 100 training registrations do not share one vertex ordering, the plan is
wrong and every downstream number is meaningless.

| check | result |
| --- | --- |
| vertex / face counts across 100 registrations | one distinct pair: **(6890, 13776)** |
| **face arrays identical** (topology, not just counts) | **yes**, all 100 |
| per-vertex 1-ring area, pairwise Pearson correlation | mean **0.9653**, min **0.9417** |
| per-edge length, pairwise Pearson correlation | mean **0.9564**, min **0.9347** |
| id → (subject, pose) recovered from the area-correlation matrix | **subject-major** (`id = 10·s + p`): within 0.9843 vs between 0.9652; pose-major gives 0.9739 vs 0.9663 |

**Verdict: PASS.** A scrambled vertex order would give ≈0 correlation. These are ≈0.96 on two independent
fingerprints, and the face arrays are identical, so vertex `i` **is** the same anatomical point on every
scan. `π̂ = I/n` is the ground truth.

Two data facts worth recording:

- **`training/ground_truth_vertices/tr_gt_XXX.txt` indexes SCANS, not registrations**: the first file has
  176 387 lines against a 6 890-vertex registration. It cannot be used directly to mask registration
  vertices; a scan→registration map would be needed first.
- The 200 test scans ship **without** registrations (withheld for the website), so any evaluation here must
  split the 100 training registrations by pose instead.

---

## 2. Gates 6.1–6.3 — is there headroom?

Descriptors: `K = 13` — extrinsic `xyz` (centred, scaled by the bounding-box diameter), `log 1-ring area`,
`log` cotangent-Laplacian curvature, and 8 **HKS** columns at times `1…128`. All standardised with the
pooled source+target mean and std, the same constants for both shapes.

Pairs `17–65`, `1–94`, `23–74` (subject-major, so all three are inter-subject); `n = 800` shared-index
subsample; exact EMD with uniform marginals; **error = `‖B[pred] − B[true]‖ / diameter(B)` — Euclidean, a
proxy for the published geodesic error, and optimistic. State this beside every number.**

| arm | mean error | miss@1 % | miss@5 % |
| --- | ---: | ---: | ---: |
| position-only OT | 0.1680 | 0.903 | **0.768** |
| uniform weights over descriptors | 0.0825 | 0.613 | **0.380** |
| **oracle `θ` (fitted on this pair)** | **0.0392** | 0.401 | **0.170** |
| nearest-neighbour, uniform | 0.1006 | 0.734 | 0.495 |
| nearest-neighbour, oracle `θ` | 0.0779 | 0.617 | 0.357 |

Per pair, miss@5 % for (position-only → uniform → oracle): `0.861 → 0.431 → 0.207`,
`0.554 → 0.190 → 0.037`, `0.889 → 0.517 → 0.264`.

| gate | rule | verdict |
| --- | --- | --- |
| **6.1** position-only fails visibly | majority of vertices off by > 5 % of the diameter | **PASS on all 3** (0.86 / 0.55 / 0.89) |
| **6.2** uniform weights leave room | miss@5 % > 5 % | **PASS on all 3** (0.19–0.52) |
| **6.3** oracle `θ` beats uniform | ≥ 5 points better miss@5 % | **PASS on all 3** (−22, −15, −25 points) |

**Verdict: PASS.** The direction has headroom: the natural cost is bad, equal weights are clearly
imperfect, and a `θ` *chosen with knowledge of the answer* cuts the miss rate by 15–25 points. That last
gate is the one that killed the previous two studies, and here it is comfortably clear.

---

## 3. The result that matters most: HKS is load-bearing

The first run used only the 5 cheap descriptors (xyz, area, curvature). Gate 6.3 **failed**:

| descriptor set | uniform miss@5 % | oracle miss@5 % | gate 6.3 |
| --- | ---: | ---: | --- |
| 5 cheap (xyz + area + curvature), pair 17–65 | 0.755 | 0.729 | **FAIL** (−3 points) |
| + 8 HKS columns (K = 13), pair 17–65 | 0.431 | 0.207 | PASS (−22 points) |

With cheap descriptors alone the oracle `θ` collapsed onto the box boundary — `θ = [0.05, 0.05, 0.05, 0.80,
0.05]` at `rel = 0.25` (floor 0.05, ceiling 0.80), i.e. "all the weight on log area, the minimum on
everything else" — and it still could not separate the true plan. **Local curvature and position carry
almost none of the correspondence signal; the spectral embedding does.** With HKS the oracle puts its mass
on the late-time columns (times 32/64/128), which is the global-shape regime.

That is a concrete, useful negative: any future `K` sweep can start from "the spectral block is the one
that matters" rather than guessing.

---

## 4. OT vs nearest-neighbour

Worth separating, because it decides whether the coupling contributes anything:

| weights | EMD (OT) | per-row argmin (no OT) |
| --- | ---: | ---: |
| uniform | 0.0825 | 0.1006 |
| oracle `θ` | **0.0392** | 0.0779 |

OT is better in both cases and **the advantage doubles once the weights are good** (2× rather than 1.2×).
So the global competition does contribute, and it is not merely a reweighting — the same conclusion the
pairing study could not reach.

---

## 5. Honest limitations

- **One subsample draw.** `n = 800` of 6890, one random index set, three pairs. No error bars.
- **No held-out evaluation.** The only `θ` so far is the oracle, which sees the pair it is scored on. It is a
  *ceiling*, and a ceiling is not a result.
- **Euclidean error.** The geodesic metric that the FAUST protocol uses is not implemented (no geodesic
  solver installed). Euclidean distance across a gap can be short while the surface path is long, so these
  numbers are optimistic and **not comparable to published FAUST results**.
- **Three pairs, all inter-subject.** No intra-subject (same person, different pose) case, which the
  guideline expects to be easier.
- **A criterion was changed mid-way, on the record.** Gate 6.1 started as "position error > 3× uniform"
  (misreads when both arms are bad), then gained "mean > 0.1" (which failed a genuinely-bad pair at mean
  0.0828 while 55 % of its vertices were still off by >5 %), and finally settled on the single absolute rule
  now used: **majority of vertices off by more than 5 % of the diameter**. All three variants and their
  reasoning are in `gates.py`'s comment and `PROVENANCE`.
- **`θ` at a box vertex.** The oracle `θ` sits on the box floor/ceiling in every run and the loop reports
  `converged=False` with a small regret (≈0.01). That is the same corner behaviour the mosaic study showed;
  it needs the same investigation (wider box, simplex-only, held-out `λ`) before any claim about what the
  weights *mean*.

---

## 6. What this licenses

It licenses **building the actual experiment**: learn `θ` on one pair, evaluate on held-out pairs, against
uniform weights and the oracle ceiling. It does **not** license a claim about correspondence quality, and
the numbers above must not be compared to published FAUST errors.

Before that experiment, two things should be settled cheaply: a **geodesic** error implementation (so the
metric is the standard one), and the **corner-`θ`** question (so "learned weights" mean something).

| need | state |
| --- | --- |
| geodesic distance on a mesh | **absent** — needs `potpourri3d` or a Dijkstra approximation |
| multi-pair shared `θ` | **absent** — `solve_inverse_ot_regret` takes one problem; block-restricted scanning is a code change |
| WKS columns, SHOT-like descriptors | absent (HKS + curvature + area is what exists) |
| FAUST test-set ground truth | **unavailable by design** (website); split the 100 training registrations by pose |
