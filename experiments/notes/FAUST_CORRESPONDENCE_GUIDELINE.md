# Guideline — inverse OT for shape correspondence on FAUST

**Status: DESIGN ONLY. Nothing here has been run. No number in this file has a record behind it.**
It supersedes the photomosaic line (`legacy/pictures/`), which was retired because its natural
formulation was degenerate and its workable one was arbitrary.

**Why this task, in one sentence.** The observed plan is free and exact (a registered mesh has a known
vertex bijection), the natural cost fails visibly across poses (so there is headroom), a standard
evaluation protocol already exists (geodesic error normalised by surface area), and the per-vertex
descriptor form is exactly the diagonal-weight family this solver already handles.

---

## 1. Data

### 1.1 Primary — FAUST

300 scans of 10 people in many poses. The **training** set is 100 scans (10 per subject) with
ground-truth alignments; the 200 test scans are scored through the evaluation website.

**The property everything rests on:** the 100 training registrations are 10 people in 10 poses, all
sharing **one mesh structure with 6890 vertices**, so vertex `i` on one shape corresponds to vertex `i`
on every other. **Verify this on the files actually downloaded** — the plan `π̂ = I/n` is only correct if
it holds, and it is the first pre-flight check (§6.0).

**Access.** Available for research via the FAUST website, which requires registration and accepting a
licence. **This is a blocker that only the user can clear**; nothing below can run until the meshes are
on disk. Do not assume a mirror is interchangeable.

### 1.2 Secondary

| dataset | what it gives | note |
| --- | --- | --- |
| **TOSCA** | 80 synthetic meshes, animals and people; meshes of the same class share topology | correspondence is immediate, different vertex counts per class |
| **SCAPE** | many poses of a single person | from memory; check topology before use |
| **SHREC'19** | 44 shapes, 430 pairs, 5K–50K vertices, ground truth through the 6890-vertex SMPL model | **skip in round one**: vertex counts differ, so the correspondence is not a bijection and the balanced formulation does not apply |

### 1.3 Controlled-difficulty layer (the synthetic axis)

Take one registered FAUST mesh and apply random **bends, twists and articulations** that keep vertex
identity. This gives unlimited pairs with a tunable distortion strength `σ_d`, which makes the
"accuracy vs deformation" curve cheap and monotone. It is the same role the three-axis design gives to
`(family, marginal, scale)` elsewhere: one axis you can turn without recruiting data.

**Also usable:** pose distance between real FAUST scans as a natural difficulty axis (§7, figure 5).

---

## 2. The instance

| item | choice | why |
| --- | --- | --- |
| plan | `π̂ = I/n`, uniform marginals | an exact permutation; the registration *is* the plan |
| shuffle | permute target vertex order before building the problem | otherwise index order is the answer |
| size | `n = 6890` → `4.7×10⁷` pairs; `K ≈ 100–300` | one scan ≈ `10¹⁰` flops, trivial on GPU; memory `O(Kn)` |
| cheap exact check | same index subset on both shapes (~2 000 vertices) | the bijection survives the subset, so exact EMD stays affordable for evaluation |
| effective sample size | **not** 6890 independent observations | neighbouring vertices are strongly correlated; keep `K` modest and validate on held-out pairs |
| training pairs | **one pair to start** | concatenating pairs adds cross-pair constraints that are not wanted; the clean version is block-restricted scanning with a shared `θ`, which is a code change (§8) |

**Do not** build a multi-pair instance by stacking pairs into one plan: that couples a source vertex to
another pair's target vertices and changes the problem.

---

## 3. The `φ_ij` family

Per-vertex descriptors `a_i` (source) and `b_j` (target), with

```
φ_ij,k = (a_ik − b_jk)²        c_ij(θ) = Σ_k θ_k (a_ik − b_jk)²
```

This is the diagonal-weight form, so the scan is the `O(mnK)` blockwise path (or the bilinear
embedding: `Σ_k θ_k (a_ik − b_jk)²` expands to `|√θ ⊙ a_i|² + |√θ ⊙ b_j|² − 2⟨√θ ⊙ a_i, √θ ⊙ b_j⟩`).

| group | contents | `K` | notes |
| --- | --- | --- | --- |
| **intrinsic spectral** | heat kernel signature at log-spaced times; wave kernel signature over a range of energies | 16 + 32 | pose-robust, but blind to left/right symmetry |
| **local geometry** | curvature at 3–4 scales; local-PCA ratios at several radii | ~12–20 | mildly pose-dependent |
| **global intrinsic** | average geodesic distance, geodesic eccentricity | 2–4 | cheap |
| **extrinsic** | `x, y, z` after centring | 3 | the naive baseline; should receive a **small** weight |
| optional (round two) | SHOT-like local histogram; multi-view foundation-model features lifted to vertices | 100s | needs plumbing |

The spectral features come from the standard Laplace–Beltrami eigenbasis — seconds per mesh with a
sparse eigensolver.

**Two variants, run both.** The per-coordinate version (`K ≈ 100–300`) has capacity; the **grouped**
version uses one weight per group on the within-group mean squared difference (`K ≈ 6–10`). The grouped
one has far lower capacity and an interpretable answer, and the pair of them is a free capacity control.

### 3.1 Symmetry

Intrinsic descriptors cannot tell the two sides of a body apart. A learned `θ` can only fix that through
the **extrinsic** columns. Report errors twice: the standard metric, and a **symmetric-flip-tolerant**
count (a match to the mirrored vertex counts as correct). If the extrinsic weight stays at the box
floor and the flip-tolerant error is much lower, that is the symmetry problem speaking, not a failure.

### 3.2 Identifiability and scale (carried over from the sanity report)

- **Row-only / column-only features.** In the earlier pairing setup, a descriptor that depended on one
  side alone was absorbed by the potentials and its weight unidentifiable. Here `φ_ijk = (a_ik − b_jk)²`
  is a paired difference, so the pathological case does not arise *by construction* — **but only if
  every descriptor has variance on both shapes.** Check that: a descriptor that is constant across the
  target has exactly the absorbed-direction behaviour.
- **Scale.** Standardise each coordinate **globally across shapes**, with the same constants at train
  and test time. The simplex budget makes this non-optional. Descriptors on wildly different scales
  otherwise silently rescale the whole problem.

---

## 4. Using the solver

1. Compute and standardise descriptors for all meshes (train and test, one global standardisation).
2. Build the instance from **one** pair, `π̂ = I/n`, target shuffled.
3. **Use the relaxed (regret) solver.** Exact feasibility will almost surely fail on real descriptors:
   the true correspondence is not Pareto-optimal for any non-negative weighting, so the exact master
   has nothing to certify.
4. Keep `B = S ∪ R`. Here `S` is `n` single-edge components, so the connector `R` carries about `n`
   edges and `|F| ≈ 5–10 n ≈ 35–70k` edges. That is comfortably inside the measured range
   (`SERVER_TESTS_REPORT.md`: `|F|/(m+n) ≈ 3–5`).
5. **Regularise with a trust region** around uniform weights,
   `Θ_λ = {θ ∈ Δ : ‖θ − θ_unif‖_∞ ≤ λ}`, with `λ` chosen on **held-out pose pairs**. This is the same
   box that `BoxSimplex` already implements in `harness/solver.py` (`rel` is its inverse-width knob), and
   the F2 study already cross-validates it — reuse that machinery rather than inventing a penalty.
6. **Evaluate by solving forward OT** between held-out pairs under the learned cost: exact EMD on the
   ~2 000-vertex index subsample, or a GPU solver on the full mesh. Take the matched vertex per source
   vertex.
7. Also record **unconstrained nearest-neighbour** matching, to separate the effect of OT competition
   from the weights themselves. (`REAL_PAIRS_REPORT.md` §5 is the precedent: ranking and coupling fail
   differently.)

### 4.1 Metric

The usual protocol is the **geodesic distance on the target** between predicted and true vertex,
normalised by the square root of the surface area. Report:

- mean error,
- the cumulative-error curve,
- share of vertices within **1 %** and **5 %** of the normaliser.

Exact-vertex accuracy is far too harsh: a neighbouring vertex on a dense mesh is a good answer.

**Caveat:** the normalisation convention is from memory and is a common source of non-comparable
numbers. State the convention used, exactly, next to every number.

---

## 5. Baselines and decision rule

| baseline | what it isolates |
| --- | --- |
| position-only OT | the naive cost; must fail visibly (§6.1) |
| uniform weights over all descriptors | the honest "no learning" reference |
| uniform weights per group | capacity control |
| Gromov–Wasserstein on a subsample | the standard geometric reference |
| **oracle `θ` fitted on the test pair** | the ceiling of the hypothesis class (§6.3) |
| shuffled-correspondence null | the floor |

**Statistics.** Paired bootstrap over **test pairs**, never over vertices: vertices within a mesh are
not independent, and a vertex-level bootstrap will manufacture significance.

**Success criterion, fixed in advance.** The learned `θ` beats uniform-weight OT on held-out
*inter-subject* pairs by more than the pair-to-pair spread.

**Scope.** This is **not** a state-of-the-art claim. Deep functional-map methods reach far lower errors
on FAUST. The claim is only that **a learned mixture of hand-crafted costs beats hand-weighted ones**.
Published numbers are not comparable across protocols — run the baselines.

---

## 6. Pre-flight checks — stop if any fails

These exist because the previous two studies both died of a ceiling effect that was visible in advance.

| # | check | stop if |
| --- | --- | --- |
| **6.0** | the 6890-vertex bijection actually holds in the downloaded files | it does not — then `π̂ = I/n` is wrong and nothing below is meaningful |
| **6.1** | position-only OT fails visibly across poses | it is already near-perfect → increase pose distance, the task has no headroom |
| **6.2** | uniform-weight descriptor OT is clearly imperfect | it is not → learning has nothing to add |
| **6.3** | the **oracle** `θ` (fitted on the test pair) beats uniform weights by a few points at the 1–5 % thresholds | it does not → the class has no room; this is the same ceiling as the cosine results, and it must be found *before* the experiment, not after |

Check 6.3 is the one that would have saved both previous studies. Run it first.

---

## 7. Figures

1. **Colour transport.** Colour each source vertex by its coordinates, transport the colours onto the
   target. A smooth rainbow means a good match; a scrambled one means a bad match, at a glance.
2. **Error heat map on the target surface** — position-only, uniform, learned, side by side.
3. **Cumulative-error curves** for all baselines.
4. **Bar chart of learned weights by group** (and per-coordinate for the high-capacity variant).
5. **Accuracy vs deformation strength** (synthetic layer) and **vs pose distance** (real pairs).

Each figure carries a `.caption.md` with the instance string, method and config (`AGENTS.md` §7.2). No
figure is committed until its record is.

---

## 8. What this repository does not have yet

Stated plainly so the estimate is honest:

| need | state |
| --- | --- |
| mesh I/O and geodesics | **nothing installed**: `trimesh`, `potpourri3d`, `robust_laplacian`, `igl`, `open3d` all missing. Geodesic error needs one of them (or a CPU Dijkstra approximation, which is a different metric — say which) |
| Laplace–Beltrami spectrum | not present; needs a sparse eigensolver on the cotangent Laplacian |
| multi-pair shared-`θ` | **not supported.** `solve_inverse_ot_regret` takes one problem. The clean version is block-restricted scanning with a shared `θ` column block — a real code change |
| FAUST data | not on disk; licence acceptance is the user's |
| a mesh feature family | `src/iot/core/features.py` has `DiagonalWeightFeatures`; that is the right form, but descriptors must be computed outside it and passed in |

**Suggested first milestone** is deliberately small: one pair, `K ≈ 20` (spectral + curvature +
global + extrinsic), the ~2 000-vertex subsample, and checks 6.0–6.3 only. If 6.3 fails on that, the
whole plan is dead cheaply.

---

## 9. Caveats

- **Everything in §3–§5 is design, from the user's brief and from memory.** The descriptor choices, the
  geodesic-error normalisation and the multi-pair design have **not** been checked against this codebase.
- **A single `θ` across distortion levels may be a compromise** that fails to beat the best fixed cost
  *at each level*. If the accuracy-vs-deformation curve is flat for the learned cost while each fixed
  cost wins its own region, that is the finding, and it argues for a `θ(σ_d)`.
- **Dynamic Gaussian splats with persistent identities** (deformation-field models) give the same
  experiment with a free association, but the "ground truth" is then the model's own association rather
  than physical truth. Treat as a follow-up, not as evidence.
- **A free plan is not a free lesson.** The plan being exact removes one failure mode (noisy plans,
  `AGENTS.md` §4.6) and leaves the real one: whether the descriptor family has enough capacity and
  whether the natural cost is really as bad as it looks. Checks 6.1–6.3 are the guard.
