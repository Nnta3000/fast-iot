# What iOT is for, beyond "paired data → cost → pair new data"

**Status: NOTES, not results.** Nothing here has been run. No number in this file has a record behind it,
so no number in this file may be cited. It lives outside `reports/` on purpose: every file there is either a
result with a record or a plan with a progress marker, and a speculative note would blur that.

**Origin.** Questions from the user, 2026-11-05: what else can learned inverse-OT costs do, and can any of
it make a picture worth looking at.

---

## 0. The framing that decides what is possible

**iOT does not generate anything.** It returns a cost `c_θ = Σ_k θ_k φ_k(ij)` fitted to examples. To produce
an image, a sound or a frame you plug `c_θ` into a **forward** OT process: an assignment, a displacement
morph, a colour transfer, or a noise→data coupling. Everything below is a choice of forward step.

The solver pays off in exactly three situations, and an application that is in none of them will not benefit:

1. **The observed plan is free and plentiful** — identity across frames, shuffle-and-restore, an ODE map.
2. **The cost mixes heterogeneous attributes** whose relative weights nobody wants to tune by hand —
   position, scale, colour, opacity.
3. **A global matching is needed at a size where minibatches lose the structure** — `n` in the thousands to
   hundreds of thousands.

A useful sanity habit before any of these: run the shuffled-pairs null (check A,
`studies/real/t1t2/sanity/run.py --check null`) at the same `(m, K)`. If the fit is no better than its
permuted control, the forward step is drawing from a cost that learned nothing.

---

## 1. Explanatory and diagnostic uses

### 1.1 Explain or audit a coupling
Take any observed assignment — a dispatcher's shift roster, a school-choice outcome, a black-box matcher's
output — and find the sparse `θ` that explains it. `θ` is the **revealed cost**, and the margin / `R*` say
how well determined it is. A fairness audit is then a question about where the weight sits: if a protected
attribute carries mass, the assignment is trading it for something. This is the use where the solver's
weakness on noisy paired data becomes a strength, because the plan is exact and plentiful **by construction**.

### 1.2 Test whether a process is OT-like at all
Exact feasibility means the observed plan is Pareto-optimal for the `K` feature costs, and `R*` is the best
simultaneous improvement available. With a shuffled-pairs null this is a test statistic: *is this coupling
distinguishable from an OT coupling under any non-negative combination of these features?* The obvious
target is biological lineage couplings, where the question "is lineage tree structure OT-like?" is asked
informally all the time.

### 1.3 Analyse generative models
A trained flow or diffusion ODE defines a deterministic map from noise to data — an exact plan with
unlimited free pairs. Fit `θ` and you learn which features it trades off; `R*` measures **how far it is from
being an OT map under a weighted cost**. This is the most comfortable use of the solver (huge, clean,
exact plans), and it needs no labels at all. *(Whether ODE maps are genuinely close to OT is argued in the
literature; that claim is not checked here.)*

### 1.4 Calibrate a ground metric from a few correspondences
Wasserstein losses and registration both need a ground metric, and both are usually handed `‖x−y‖²`.
A few hundred known correspondences turn that into a fitted `c_θ` with a stated scale — which is exactly
what the EOT line in `reports/current/eot/` is careful about.

### 1.5 Compress a plan
If a plan is exactly feasible, `K` numbers plus the point sets reproduce a sparse plan. Useful as a
storage/diagnostic trick; not an end in itself.

### 1.6 A representation-quality score (cheap, and close to home)
`R*` or `δ*` on a fixed labelled task is a score for **comparing encoders**, in the spirit of linear probing
for matching. It needs no new machinery — it is the fit already being run — and it is the most direct test of
the "dictionary of encoders" idea in §3.2.

### 1.7 A diagnostic on one's own generators
`SERVER_TESTS_REPORT.md` already records `rel_gap` per run; that is a measure of how far a generator's plan
is from OT-optimal. Using it deliberately as a **generator-quality metric** rather than a solver diagnostic
is free and unused.

---

## 2. Boundaries — where this framing fails

- **Aggregated observations only** (clone-level couplings, district-level flows): the problem becomes
  bilinear in `(θ, π)` and the LP framing is lost. Worth writing down as a boundary, not a defect to fix.
- **Noisy, non-optimal plans** (Sinkhorn output, sampled plans): the master is infeasible. Out of scope for
  v1 by `AGENTS.md` §4.6; do not threshold silently.
- **A free plan teaches only what it is.** The identity plan of one scene against itself says how attributes
  co-vary *under motion*; it says nothing about which parts of *different* objects should correspond. Any
  application that needs semantic correspondence needs real correspondence data.

---

## 3. Picture projects

Shared shape: an observed plan (cheapest where free), a cost dictionary (the diagonal-weighted squared
difference the solver already handles in `O(mnd)`), and a forward OT step. Plus a **baseline that is
visibly wrong in a way the examples reveal** — otherwise the picture proves nothing.

| project | plan comes from | features `φ_ij` | forward step | vs what |
| --- | --- | --- | --- | --- |
| **photomosaic / jigsaw** | free: shuffle tiles of a real image, the true layout *is* the plan | squared differences of sub-block colour, gradient orientation, DINO patch features | assign a bank of tiles (~10⁵) to the cells of a new image | colour-mean assignment; by eye and by reconstruction error |
| colour / style transfer | before/after edits give pixel plans by position | Lab differences, position, local gradient | OT between a new image and a target palette | classic OT colour transfer |
| flow matching with a learned coupling | encoder pairs `E(x) ↔ x`, or an ODE map | cross terms `z_k h_k(x)` | couple fresh noise to data, then train the flow | sample grids, FID, against the `‖z−x‖²` coupling |
| semantic morph | hard: needs real correspondences (landmarks, flow) | position, colour, DINO patch features | `x_t = (1−t)x_i + t y_j` along the plan | `W₂` morph |
| **splat morph / 4D tracking** | free only for the *identity* plan; see §3.3 | per-attribute squared differences on (position, log-scale, rotation, Lab, opacity) | interpolate along the plan, or assign Gaussians to cluster centres | hand-weighted attribute cost |

### 3.1 Photomosaic — the first project
Chosen because the plan is not only free but **verifiable**: you start from a real image, shuffle its tiles,
and the ground-truth layout is known at every pixel, so reconstruction error is an exact metric rather than
a visual judgement. It also exercises the 10⁵ regime where the "minibatches lose structure" argument has
teeth.

> **Retired 2026-11-09.** The photomosaic project was built and run; it is now in
> `../legacy/pictures/`. The natural formulation is degenerate (see that README) and the learned weights
> matched the unweighted ones, so this whole section is kept as reasoning, not as a plan. The active design
> is shape correspondence — `FAUST_CORRESPONDENCE_GUIDELINE.md`.

### 3.2 Gaussian splats
A Gaussian's descriptor (position, log-scales, rotation, Lab colour, opacity) is exactly the heterogeneous
attribute set that nobody wants to weight by hand, and the diagonal-weighted squared difference is the
family this solver already handles. A Bures–Wasserstein term between covariances is more principled but
costs a matrix square root per pair — start with the cheap proxy.

### 3.3 The trap in the free splat plan
The identity plan of a scene against itself is **degenerate for this solver**: support of size `n` with `n`
connected components, which `AGENTS.md` §4.7 flags — the connector set becomes a chain of `n` edges. A
single frame pair is therefore a stress test, not a data source. The well-posed version **averages across
frames** (many plans, one shared `θ`, §4.1) and needs real correspondence for anything semantic.

---

## 4. Levers, ordered for this repository

| # | lever | status | note |
| --- | --- | --- | --- |
| 1 | **held-out regret + plan recall** instead of hit-rate | recommended first | hit is a hard 0/1 per row on 400 pairs; regret is continuous and uses every pair. `regret_objective` already exists |
| 2 | shuffled-pairs null at the real `(m, K)` | **done** — `SANITY_CHECKS_REPORT.md` §2a | signal in 9/9 cells; margin decays with `m` |
| 3 | learning curve to the pool ceiling | **done to `m = 600`** — §2b | still rising at `K = 64` (+13.2 pts); `10³–10⁴` needs more subjects, not a different solver |
| 4 | same-class baselines (softmax / Sinkhorn / exact / relaxed) | written, not run | `sanity/run.py --check baselines` |
| 5 | trust region around cosine, by CV | **partly exists** | the `rel` box *is* `‖θ − θ_unif‖` bounded; F2 already CV's it. What is missing is treating `at_bound` as the diagnostic it is |
| 6 | feature hygiene (double-centre, standardise) | written, inconclusive | `sanity/ident.py`; no headroom on the instances tried |
| 7 | multi-encoder dictionary, sparse `θ` | not started | `K` in the hundreds–thousands with an `L1` penalty: multiple-kernel learning for OT |
| 8 | many plans, one `θ` | **the lever I would move up** | block-structured LP, shared `θ` column; makes the null a distribution instead of one permutation |

### 4.1 Why 8 before 7
The dictionary changes the hypothesis class; many-plans changes the **statistics**. It is also what makes
the shipped applications well-posed — every free-plan application (§3.3) is a many-plan problem or it is
degenerate.

### 4.2 Learning the features (P1–P4)
- **P1 frozen encoder + exact last layer** — the current setup; cheapest, uses the solver's scale.
- **P2 differentiable layer** — forward: solve the relaxed master; backward:
  `∂R*/∂φ_e = (π̂_e − π*_e)·θ*` by the envelope theorem, with `π*` from the master's multipliers. Then train
  `φ_ψ`, e.g. `h_ψ(x) ⊙ g_ψ(y)`. Non-convex in `ψ`, and `θ*` must be kept global rather than per-batch.
  SuperGlue-style. **Honest view: only worth it at `n ≫ 10⁴`**, where minibatches lose structure.
- **P3 alternate** — update `ψ` with `π̂ − π*`, then refit `θ` exactly. The cheap version of P2.
- **P4 `R*` as an encoder score** — free, and it is the test of the §3.2 claim.

---

## 5. Tasks that fit better than "paired data → cost → pair new data"

| task | why it fits |
| --- | --- |
| **tracking across frames** (cells, particles, objects) | ground-truth assignments from the tracker; many frames share one `θ`; `n` is large in light-sheet microscopy |
| **paired multi-omics** | truly paired cells (Multiome) give the plan; the learned cost is applied to unpaired data; held-out pairs give clean evaluation |
| **matching markets** | structural estimation of surplus (Becker; Choo–Siow; Galichon) — the "revealed transferable utility" problem *is* inverse OT |
| **landmark-based registration**, record linkage, cross-modal retrieval with known pairs | the plan is the correspondences, and it is available by construction |

---

## 6. Open question worth more than most of this file

`residual64` gains far faster than the OT arms in F2 (+25.5 points at `m = 300`). It is **not** an iOT arm:
it is a learned bilinear map `xᵀ(I + UVᵀ)y` with step 0 *equal to cosine*, trained by gradient descent. So
the comparison is "a flexible supervised metric under the same features and early stopping" against "a
simplex-constrained OT fit", and `metric_learning/README.md` records that residual64 selects the epoch-0
checkpoint (exactly cosine) in 7 of 27 fits. Understanding that asymmetry is likely worth more than any of
the applications above.
