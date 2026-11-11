# Changelog — `iot_experiments/reports/` and the tree it describes

Newest first. One line per change. This file records **organisational** changes; scientific
corrections live inside the report they correct.

## 2026-11-11 (latest) — support-matched control, averaged θ, three seeds

`correspondence.py` gains two derived weight vectors, evaluated on the same held-out pairs:
`learned_avg` (per-coordinate mean of θ across vertex subsamples) and `support_matched` (uniform over the
nonzero support of that average). Plus `--pairing intra|inter`.

At n=4000, 3 seeds, 8 pairs (24 readings): `uniform` 0.4215 → `support_matched` **0.3330** → `learned`
**0.2020** miss@5 %. So **the fitted magnitudes matter more than the support** (−0.131 vs −0.089) — the
opposite of "the fit only picks the late-HKS block", and consistent with uniform-late-HKS being the worst arm.
`learned_avg` (0.2025) is indistinguishable from per-seed, so averaging is a no-op — a stability result.

`support_matched` under the **box** arm is exactly `uniform_theta` (the box makes the support all 13), so the
control only carries information under simplex-only.

## 2026-11-11 (later still) — memory: the master's full-grid arrays are gone; n=6890 fits


**Full-resolution result**: the correspondence experiment now runs on all **6890 vertices**, 19.1 min, peak
1.63 GB, exit 0 (it previously did not fit). `learned` 0.1996 miss@5 % vs `uniform` 0.4186 and
`uniform_late_hks` 0.6899; 8/8 pairs against every baseline; Δ vs uniform **−0.219** [−0.304, −0.146].
The n=800 → 4000 → 6890 sequence (0.186 → 0.203 → 0.200) shows the resolution axis has **saturated**, so
n=4000 was already representative.


`memray` located the peak, and it was **not** the LP matrix (which is sparse: `assemble_lp` at `|F|=7805`
costs +5 MB). It was:

1. `repair_point` computing `features.costs(theta, all_rows, all_cols)` over the **whole grid** in one
   allocation — **4.9 GB at n=6890**, on *every* master solve. Only `max σ` is used, so it is now computed
   blockwise (`maximum_slack_grid`).
2. `master_regret` materialising `all_rows`/`all_cols` (16 B/edge) to feed that call.
3. `master_regret` materialising `all_keys` and `setdiff1d` against F for the `sigma_max_out_F` diagnostic
   (~1.1 GB at n=6890) — now `maximum_slack_outside_grid`, blockwise.

**Measured**: peak at n=2000 **2116.9 → 1565.4 MB**, at n=5000 **7239.7 → 1612.1 MB**, and **n=6890 now
peaks at 1628.1 MB** (previously ≈11 GB extrapolated and never completed). Memory is now flat in `n`.

**Verified value-identical** (`baseline.py` before/after, `diff`): exact `delta`/`is_optimal`/`gap`, relaxed
`theta`/`regret`/`lower`/`upper`/`max_sigma_all`, FAUST fit `theta`/`regret`. Only the memory lines moved.

**Tests**: `tests/iot_maximum_slack_test.py` gains 2 — the outside-grid max must equal the `setdiff` formula
exactly (across block sizes 1…1000), and an empty complement must return `-inf`.

**Tooling**: `memray` 1.20.0 and `pytest-memray` installed in the venv only. `pyproject.toml` is unchanged —
`AGENTS.md` §0 permits only the `"iot*"` include and the `psd` extra, so a profiling tool is not a project
dependency.

## 2026-11-11 (later) — FAUST at n=4000, and the boundary-θ question

**Resolution check**: the correspondence experiment re-run at **n = 4000** (5× the 800 used everywhere else),
subject-disjoint, simplex-only, 8 held-out pairs. `learned` 0.2028 miss@5 % vs `uniform` 0.4219 and
`uniform_late_hks` 0.6728; **8/8** pairs against every baseline; Δ vs uniform **−0.219** [−0.301, −0.149],
i.e. the margin *grows* with `n` (it was −0.151 at n=800). Uniform late-HKS also has the **worst flip rate**
(0.413 vs 0.058 for learned), so "just use the block" is the one shortcut that fails most clearly.

**Boundary-θ question settled** (`corner.py`): CuPDLPx and HiGHS reach the **same vertex** to ~0.006, so the
corner is the objective's (a linear objective over a polytope has a vertex optimum), not a solver defect.
Across 5 vertex subsamples the **zero pattern is identical** (4 descriptors always exactly 0) but
t=32/64/128 trade mass with std 0.019–0.028 — block-level weight claims are safe, per-coordinate ones are not.

**Two failures reported in-process**: an LP-based identifiability range was degenerate (objective `±θ_k`
leaves `(f,g)` cost-free) and HiGHS did not finish one cell in 18 min — killed (exit 143), replaced by
resampling; and a seed bug made the first stability run report a fake `std = 0.0000`, caught because it was
too good.

**Not run**: full `n = 6890`. Measured peak 7.24 GB at n=5000 and ≈11 GB extrapolated at 6890, above the
10.7 GB (70 %) ceiling; an n=5000 attempt was OOM-killed (exit 137) at 6.8 GB available.

## 2026-11-11 — `loop_regret`: no more full-grid index arrays

`_compute_baseline` used to `meshgrid` the indices of all `m·n` edges and keep them for the whole solve
(16 B/edge: 64 MB at `n=2000`, **760 MB at `n=6890`**), purely to feed a blockwise `max`. New
`core.scan.maximum_slack_grid` **generates** them per row block; `_baseline` no longer stores them.

**Only this change was applied**, because it is the only one that is provably value-identical: the same edges
are visited and `max` does not depend on order. The tempting second change — replacing the pass with
`max(0, scan.max_violation)` — was **rejected**: on `F` the master's inequalities hold only to the solver
residual (this solver reports `converged=False` routinely), so an under-stated `s` would make the certificate's
upper bound `U` too small. That is exactly the kind of short-cut that could invalidate a bound.

**Verified**, not assumed (`studies/real/faust/baseline.py`, run before and after, then `diff`):
`exact.delta`, `is_optimal`, `relative_gap`, relaxed `theta`/`regret`/`lower`/`upper`/`max_sigma_all`, and the
FAUST fit `theta`/`regret` are **bit-identical**; only the memory lines moved (peak `n=2000` 2142.8 → 2112.6 MB;
`n=3000` 3561.8 → 3423.9 MB, i.e. −137.9 MB, matching the two 144 MB arrays).

**Tests**: `tests/iot_maximum_slack_test.py` (4 tests) — equality across block sizes 1…1000, a case with the
maximum on the **last row** (the "scanned only the first block" failure), a `1 x 1` grid, and an end-to-end
planted exact-iOT certificate check.

## 2026-11-09 (final) — memory: the `(n, n, K)` tensor is gone

**New** `SquaredDifferenceFeatures` in `src/iot/core/features.py`: coordinate-wise weighted squared difference
between two **different** point sets, computing `(E, K)` only for the requested edges. It is the memory-safe
replacement for `DenseFeatureTensor` in this study.

**What it fixes**: the full-grid feature tensor is `n²·K·8` bytes — **4.94 GB** at `n = 6890, K = 13` — and
`correspondence.cost_tensor` **doubles** it transiently (`diff * diff` keeps `diff` alive). That allocation is
what OOM-killed the heat-map job (exit 137). `fit_theta` now takes a `FeatureMap`; `_evaluate_pair` builds one
`n × n` bilinear cost per arm instead of an `n²K` tensor.

**Verified memory-only**: the whole box-width sweep was re-run; every shared metric is identical to `1e-12`
and the bootstrap rows are bit-identical.

**Measured attribution of what still costs** (n=2000): library baseline +490 MB (`all_rows`/`all_cols` over the
full grid in `loop_regret._compute_baseline`), first round +897 MB (scan + LP), later rounds ~0. At `n = 6890`
the fit needs ~75 kB per `|F|` edge, so the subsample is load-bearing, not cosmetic.

**Tests**: `tests/iot_faust_test.py` gains two — the family must match the dense formula exactly, and it must
reject mismatched dimensions rather than broadcast.

## 2026-11-09 (latest) — FAUST: heat maps, flip counts, and study hygiene

**New** `heatmaps.py`: full 6890-vertex error heat maps (bilinear cost, only `n x n` materialised) and
**left/right flip counts** — a flip is a prediction geodesically closer to the *mirrored* vertex than to the
true one.

**Result**: `position_only` puts **39 %** of vertices on the mirror side (44 % of its errors are flips);
`uniform_theta` 19 %; **`learned` 8 %**; oracle 6 %. Mirroring is a large share of the failure and learning
removes most of it — and what remains is *not* mirroring (84 % of the learned errors are local, at hands/neck).

**Job killed, reported**: the first heat-map job exited **137 (OOM)** building a full-resolution
`(6890, 6890, 13)` tensor (4.94 GB). Fixed by fitting `θ` on an 800-vertex subsample and evaluating at full
resolution through the bilinear identity. Its first successful run also used non-disjoint pairs
(`0-66`, `1-36`, subject 0 is a training subject); re-run on `50-66`, `56-71`.

**Hygiene, added**: `analyse.py` (tables and `.caption.md` for every figure, from records only),
`tests/iot_faust_test.py` (8 tests), records copied into `out/records/faust_*/`, and **every run now launched
through `scripts/run_job.sh`** with `OMP/MKL/OPENBLAS_NUM_THREADS=4` — this machine has 16 cores and is
shared, and earlier FAUST runs were started with `nohup` and uncapped threads.

## 2026-11-09 (later still) — FAUST: box width chosen on held-out pairs; simplex-only wins

`correspondence.py` now sweeps `rel ∈ {0.05, 0.1, 0.25, 0.5, 0.9}` **and simplex-only**, selecting by miss@5 %
on 4 **validation** pairs drawn from the training subjects (disjoint from the training pair and the test
subjects).

**Validation selection is honest here**: its ranking matches the test ranking for all six candidates.
**And the box was hurting monotonically** — `rel=0.9` 0.383, `rel=0.05` 0.188, **simplex-only 0.186** — with no
interior optimum. So the trust region is unnecessary for this task, and every earlier FAUST number was
obtained under a constraint costing ≈0.06 miss@5 %.

Selected configuration beats every baseline 8/8: vs uniform **−0.214** [−0.297, −0.145]; vs uniform late-HKS
−0.416; vs uniform HKS −0.474; vs grouped −0.299; vs position-only −0.463.

**Corner behaviour is not caused by the box**: under simplex-only the fit still puts 4 coordinates at exactly
0 and one at 0.77, `converged=False`. That is the objective's own preference and is now the sharpest open
question.

## 2026-11-09 (later) — FAUST: subject-disjoint split and 3 vertex subsamples

**Defect found**: the first held-out run drew pairs by id without checking *which person* each id belongs to,
so **every training pair shared subjects with the evaluation pairs** (train `(17,65)` shares subjects 1 and 6
with the tests, and so on). That result measured pose transfer within seen subjects. A `check_split`
assertion now exits on any overlap; split fixed at train `{0..4}` / test `{5..9}`; `θ` refitted per
**3 vertex subsamples**.

**Corrected result is stronger**: `learned` beats `uniform_theta` by **−0.151** on miss@5 % (CI [−0.207,
−0.104], 8/8 pairs) and now beats the late-HKS block by **−0.353** on **8/8** (it lost 2/8 under the leak).
The reversed split (`train {5..9}` / `test {0..4}`) reproduces it at −0.122, so the split choice is not
load-bearing. Vertex subsampling moves the learned arm by ≤0.046.

**What the leak hid**: under the leaky split the fitted `θ` was essentially pure late-HKS; subject-disjoint
training gives real weight to extrinsic shape and log-area — within one person the pose-invariant intrinsic
descriptor suffices, across people absolute shape matters.

## 2026-11-09 — FAUST held-out transfer: the first positive result

`studies/real/faust/correspondence.py` fits `θ` on **one** registered pair and evaluates on **eight** held-out
pairs with the standard geodesic metric (`potpourri3d`, normalised by `sqrt(area)`). Three independent
training pairs, `n = 800`.

**Result**: miss@5 % falls from 0.528 (uniform weights) to 0.391 (learned), a paired-bootstrap effect of
**−0.13 to −0.14 with every 95 % CI excluding zero**, and **24/24 individual pair-wins**. The learned `θ` — a
`θ` that never saw the evaluation pairs — captures about **75 %** of the oracle's gain. Arms order
monotonically in capacity: position-only 0.789 → 4 grouped weights 0.598 → 13 equal 0.528 → 13 fitted 0.391 →
oracle 0.345.

**Block controls added on request** (`uniform_hks`, `uniform_late_hks`) because the fitted `θ` loads almost
entirely on the late HKS columns. Result: uniform late-HKS is **0.19 worse** on miss@5 % than the learned
mixture (CI [−0.300, −0.060], wins 6/8) and uniform all-HKS is 0.25 worse — so **learning is not merely block
selection**. The two pairs where the blunt block wins are the two hardest (oracle ≈0.7), and that is recorded
rather than averaged away. The bootstrap now compares against every baseline, not just `uniform_theta`.

**Two defects found on the way**: the `grouped_uniform` capacity control was algebraically identical to
`uniform_theta` (broadcasting a group mean back over its coordinates re-weights by group size); fixed, and it
now correctly scores *worse* than the full dictionary.

**Caveat kept in front**: the learned `θ` sits on the box bounds with `converged=False`, so it is a feasible
point the solver accepted, not an identified optimum. The weights are consistently "late-time HKS + a little
area".

## 2026-11-09 — FAUST preliminary tests: gates 6.0–6.3 pass

**New study** `studies/real/faust/` with `data_check.py` (gate 6.0) and `gates.py` (gates 6.1–6.3), plus
`iot_experiments/harness/faust.py` — a minimal PLY reader and descriptor primitives (`vertex_areas`,
`cotangent_laplacian`, `spectral_basis`, `heat_kernel_signature`), because **no mesh library is installed**.

**Results** (`reports/current/FAUST_PRELIMINARY_REPORT.md`): gate 6.0 passes on all 100 registrations
(identical face arrays; per-vertex area correlation 0.965, edge length 0.956), so `π̂ = I/n` is the true
correspondence. Gates 6.1–6.3 pass on three inter-subject pairs: position-only fails visibly (miss@5 %
0.55–0.89), uniform weights leave room (0.19–0.52), and an oracle `θ` fitted on the pair cuts the miss rate
by 15–25 points.

**The sharpest finding**: gate 6.3 **fails** with the five cheap descriptors alone (0.755 → 0.729) and passes
once 8 HKS columns are added (0.431 → 0.207). The spectral block is load-bearing; local curvature and
position are not. Also: OT beats per-row argmin, and the advantage doubles once the weights are good.

**Not done**: held-out `θ`, geodesic error (Euclidean proxy used and labelled as such), `K` sweep,
multi-pair shared `θ`.

## 2026-11-09 — photomosaic retired; shape correspondence (FAUST) adopted as the design

**Retired** the whole picture line: `studies/pictures/` → `legacy/pictures/`. Reason, in one line: the
natural formulation is degenerate (cost diagonal is exactly 0 for any `θ`, so restoring a shuffled image
is free) and the workable one needed a palette borrowed from another image, which is arbitrary; on that
variant the learned weights matched the unweighted ones to three decimals. `legacy/pictures/README.md`
records why, plus the six reusable pitfalls (view aliasing, oracle direction, per-tile normalisation,
jitter, wrong recall ground truth, a "hard" task solved by the dumbest baseline).

**New** `notes/FAUST_CORRESPONDENCE_GUIDELINE.md` — shape correspondence on FAUST as the replacement:
free exact plan (`π̂ = I/n` over 6890 registered vertices), a diagonal-weight descriptor family
(HKS/WKS/curvature/geodesic/extrinsic), the relaxed solver with a trust region, geodesic-error
evaluation, and four pre-flight gates of which **6.3 (the oracle `θ` must beat uniform weights)** is the
one that would have saved both previous studies. Design only; FAUST needs a licence accepted by the user,
and no mesh library is installed.

**Updated** `EXPERIMENTS.md` §6 to mark #47–#48 retired and point at the new plan; `notes/README.md`
likewise.

## 2026-11-09 — the photomosaic runs; the jigsaw is retired as degenerate

**Ran** `studies/pictures/photomosaic/` on the T1 PNGs already on disk (`g ∈ {16, 32}`, palette from another
image, `θ` fitted on the **pixel-optimal** assignment so `MAE / MAE_oracle` is meaningful). First numbers in
`studies/pictures/photomosaic/SPEC.md` §12.

**Retired** the jigsaw variant: with the image's own tiles the diagonal cost is exactly `0` for any `θ`, so
restoring the layout is free and every arm — colour-mean included — scored MAE 0. Its record is kept as
`out/photomosaic/records.jigsaw.json`, the stale `*_jigsaw_*.png` were deleted.

**Six defects found and fixed**, all recorded in `SPEC.md` §12.1. Two were visible in the first figures:
the per-tile contrast window drew a grid (boundary `|d|` 9× the interior), and `--jitter 4` produced a
**double** seam (high-derivative runs at 63 **and** 67) because the tile was wrapped by 4 px. Seams are now
single-pixel: `63, 127, 191, …`.

**Registered** in `EXPERIMENTS.md` §6 (#47–#48) and `studies/pictures/README.md` — both since retired, see the 2026-11-09 entry above.

## 2026-11-05 — sanity checks A–E run; `notes/` and `studies/pictures/` added

**New** `iot_experiments/notes/` — the non-result home: `APPLICATIONS.md` (what iOT is for beyond pairing,
the picture projects), `PLAN_EVALUATION_AND_APPLICATIONS.md` (metric change + the user's levers 1–8 and
P1–P4 with progress), `README.md` (the rule: nothing here is citable). No file under `reports/` carries a
speculative idea any more.

**New** `iot_experiments/studies/pictures/` — `README.md` with the dataset table (MRI PNGs on disk first,
STL-10 for a visible mosaic, licence caveat on stock corpora) and `photomosaic/SPEC.md`, a frozen spec
(question, baselines, pre-registered verdict) for the first picture project.

**Sanity study results** went to `current/SANITY_CHECKS_REPORT.md` §2a–§3 as they landed; check A and E
complete, check B partial at the time of writing.

**Registered** rows in `README.md` §1 (pointer to `notes/`) and `EXPERIMENTS.md` #42–#46.

## 2026-11-05 — the five sanity checks

**New study** `studies/real/t1t2/sanity/` (study name `sanity` in every `RunRecord`):
`run.py` (`--check null|curve|baselines`), `transfer.py`, `ident.py`, with `README.md` carrying the
questions, commands and the memory/time budget. Those are the checks the user asked for that had never
been run: the permutation null, the curve past `m = 300`, the three-losses comparison, pool-size
transfer, and identifiability under double-centring.

**New report** `current/SANITY_CHECKS_REPORT.md`: hypothesis → what existed (with paths) → what runs now,
plus a section stating that the runs are **not complete** and why (GPU contention, one launch, stopped).

**Registered** in `README.md` §2 and `EXPERIMENTS.md` #42–#46.

**Not changed:** no result in any existing report was altered; where this report restates a committed
number it was re-read from the record first (two drafting errors were caught and fixed that way).

## 2026-11-05 — organise the reports; rename "T3" → "EOT"

**Rename.** "T3" was an opaque label with no meaning outside a conversation. It is now **EOT**
(entropic inverse-OT master). Applied to:

- reports → `reports/current/eot/` (see below)
- drivers → `studies/real/t1t2/metric_learning/eot_*.py` (9 files, content unchanged apart from the label)
- `section` / `study` strings → `eot_*`; output directories under `iot_experiments/out/` moved to match
  (`t3_bias` → `eot_bias`, … `t3_restricted_f` → `eot_restricted_edges`)
- docstrings in `src/iot/entropic/*.py`, `src/iot/core/scan.py`, `tests/iot_entropic_*`, `tests/reference/dense_outer.py`
- git tag `t3-freeze-gate-b` keeps its historical name (tags are history, not code)
- `out/` logs and JSON `study` fields were rewritten so a record still names its study correctly

**Reports moved and renamed.**

| was | now |
| --- | --- |
| `T3_GATE_B_REPORT.md` | `current/eot/eot_gate_b.md` |
| `T3_SOLVER_COMPARISON_REPORT.md` | `current/eot/eot_solver_compare.md` |
| `T3_SCALING_REPORT.md` | `current/eot/eot_scaling_slsqp.md` |
| `T3_RESTRICTED_F_REPORT.md` | `current/eot/eot_restricted_edges.md` |
| `T3_PLAN.md` | `current/eot/PLAN.md` (progress markers added; user wording unchanged) |
| `HIERARCHY_STAGE_REPORT.md` | `current/hierarchy_warm_start.md` |

**New.** `current/eot/README.md` (family index, P1/P2/P3 status, order of authority),
`current/eot/eot_inner_observations.md`, `GLOSSARY.md`, this file, and a full rewrite of `README.md`.

**Status added.** Every EOT report and `current/hierarchy_warm_start.md` now carries `Status:`,
`Driver:`, `Record:` and a `Supersedes` / `superseded by` note. `eot_scaling_slsqp.md` carries a
retraction banner for two numbers corrected in `eot_restricted_edges.md`.

**Corrections surfaced while indexing.**

- `eot_bias.py`'s docstring said the θ̂–θ* gap "does not shrink with eps"; the record
  (`out/eot_bias/bias.json`) shows it **does** shrink from `rel_eps` 0.5 to 0.1 (0.285 → 0.155 at
  `n=120`). Docstring corrected; the result is now in `current/eot/README.md` §1 row 5.
- `studies/real/t1t2/entropic_metric/README.md` gained a §Corrections: its cosine rows are not in the
  committed record (the `n_params=0` rows are not cosine), and the K=16 cells sit barely above a
  constant cost.
- `EXPERIMENTS.md` learned that its key sentence is wrong in one direction: P2 is **done and negative**,
  not "still blocked" — that sentence now says so in `current/eot/PLAN.md`.

**Not touched.** `src/hello_ot/**` and the other read-only paths of `AGENTS.md` §0. The git-dirty
`entropic_metric/run.py` and the untracked `entropic_vs_relaxed/` are left exactly as found.
