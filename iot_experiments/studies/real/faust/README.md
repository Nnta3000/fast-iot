# faust — shape correspondence with a learned inverse-OT cost

**Status: PRELIMINARY TESTS DONE (2026-11-09).** Gates 6.0–6.3 of
`reports/current/FAUST_PRELIMINARY_REPORT.md` pass. The full experiment is **not** built yet: no training
loop over poses, no held-out evaluation, no baselines beyond the gate set.

**Question.** On registered 3D human shapes, can a cost **learned** by inverse OT over per-vertex descriptors
recover the true vertex correspondence better than **hand-weighted** descriptor costs?

**Why this task** (rationale in `../../notes/FAUST_CORRESPONDENCE_GUIDELINE.md`): the observed plan is free
and exact — registered meshes share one vertex ordering, so `π̂ = I/n` *is* the ground truth — the natural
cost fails visibly across poses, and a standard evaluation protocol already exists.

| file | what it does | run |
| --- | --- | --- |
| `data_check.py` | **gate 6.0**: do all 100 registrations share one topology and one vertex order? | `python -m iot_experiments.studies.real.faust.data_check` |
| `gates.py` | **gates 6.1–6.3**: does position-only fail, does uniform-weight leave room, does an oracle `θ` beat uniform? | `python -m iot_experiments.studies.real.faust.gates --pairs 17 65 1 94 23 74` |
| `correspondence.py` | **the experiment**: subject-disjoint split, box-width chosen on validation pairs, per-row-argmin control, per-vertex flips | see *Reproduce* below |
| `heatmaps.py` | **error heat maps** at full 6890-vertex resolution, plus left/right flip counts | `--pairs 50 66 56 71` |
| `analyse.py` | **tables and figure captions from the saved records only** (§7.3 / §7.2) | `--study correspondence` or `--study heatmaps` |
| `arms.py` | **four objectives on one hypothesis class**: one-sided CE, Sinkhorn CE, entropic joint master, regret LP, plus uniform | `python -m iot_experiments.studies.real.faust.arms --n-sub 1500` |
| `multi_pair.py` | **many plans, one θ**: block LP with a shared θ, per-pair normalisation, cross-regret matrix | `--n-sub 1200 --train-pairs 0 13 2 41 5 88 7 52` |
| `deformation.py` | **deformation sweep**: twist about the vertical axis, vertex identity preserved, θ fitted once at σ=0 | `python -m iot_experiments.studies.real.faust.deformation --n-sub 1500` |
| `abstention.py` | **where is it wrong?** label-free confidence scores (raw / relative cost margin, OT dual certificate, softmax margin) as risk–coverage | `python -m iot_experiments.studies.real.faust.abstention --n-sub 4000` |
| `curriculum.py` | **iOT as a teacher**: hard negatives mined from the fitted cost, given to a cheap cross-entropy arm | `python -m iot_experiments.studies.real.faust.curriculum --n-sub 2000` |
| `face.py` | **diagnostic D1**: does the admissible set matter? random walk inside the near-optimal set, every point scored on held-out pairs | `python -m iot_experiments.studies.real.faust.face --n-sub 4000` |
| `regional.py` | **diagnostic D2**: do body regions want different weights? one fit per k-means region, plus a preliminary field evaluation | `python -m iot_experiments.studies.real.faust.regional --n-sub 2000` |
| `pairing.py` | **does the training pair matter?** 8 subject-disjoint pairs, each θ decoded on the same held-out pairs; pooling curve; cross-pair regret as a selector | `python -m iot_experiments.studies.real.faust.pairing --n-sub 2000` |
| `modes.py` | **cost of the fit**: solver tolerances vs outer-round cap vs scan backend, with the per-round master/scan split | `python -m iot_experiments.studies.real.faust.modes --n-sub 2000` |
| `select.py` | **direction A**: rank the near-optimal candidates on training-subject validation pairs and take only the winner to the held-out pairs | `python -m iot_experiments.studies.real.faust.select --n-sub 4000` |
| `finalists.py` | **the two finalists at full resolution** with the τ grid extended downward, 3 seeds, paired bootstrap (driver only — the Sinkhorn arm needs more VRAM than this machine has free) | `python -m iot_experiments.studies.real.faust.finalists --n-sub 6890` |

Loader and descriptor primitives live in `iot_experiments/harness/faust.py` (`read_ply`, `vertex_areas`,
`cotangent_laplacian`, `spectral_basis`, `heat_kernel_signature`, `descriptors`, `standardise`,
`shared_index`), because no mesh library is installed — `trimesh`, `potpourri3d`, `robust_laplacian`, `igl`
and `open3d` are all absent.

Everything that more than one driver needs **is defined once**, in the harness, not copied per driver:
`harness/faust.py` (mesh, descriptors, index subsample), `harness/faust_match.py` (cost matrix in numpy and
torch, the exact-OT decoder, the geodesic-error metric, `fit_softmax`, paired bootstrap) and
`harness/master_probe.py` (`RecordingMaster`, used by `corner.py` and `multi_pair.py`). A driver is not a
library: prior revisions had `gates.py`, `heatmaps.py`, `arms.py` and `finalists.py` each re-implementing
`_cost` / `_match` / `_errors`, and two drivers imported private helpers out of `finalists.py`.

## Data

`data/MPI-FAUST.zip` (1.35 GB) → extracted to `data/faust_extract/MPI-FAUST/`. Only
`training/registrations/` (100 PLY, 6890 vertices, 13776 faces) and `training/ground_truth_vertices/` are
needed so far; the 200 test scans are withheld for the website, so **all evaluation here uses training
registrations split by pose**.

`data/` is git-ignored; the loader refuses a missing or unexpected file rather than silently substituting.

## What is established (2026-11-09)

- **Gate 6.0 PASS on all 100 registrations**: identical vertex/face counts, **identical face arrays**, and
  per-vertex area correlation mean 0.965 (min 0.942) with edge-length correlation mean 0.956 (min 0.935).
  A scrambled vertex order would give ≈0. So `π̂ = I/n` is the true correspondence.
- **Gates 6.1–6.3 PASS on all three pairs** (`n = 800` subsample): position-only fails visibly
  (miss@5 % 0.86 / 0.55 / 0.89), uniform-weight leaves room (0.19–0.52), and the oracle `θ` — fitted on the
  same pair — more than halves the miss rate on every pair (0.43 → 0.21, 0.19 → 0.04, 0.52 → 0.26).
- **HKS is load-bearing.** With only 5 cheap descriptors (xyz, area, curvature) gate 6.3 **failed**
  (0.755 → 0.729). Adding 8 heat-kernel-signature columns made it pass (0.431 → 0.207). The cheap
  descriptors alone have no room.
- **OT beats nearest-neighbour** once the weights are decent: with the oracle `θ`, EMD scores 0.039 mean
  vs 0.078 for per-row argmin. With uniform weights the gap is smaller (0.083 vs 0.101).

## Memory: what actually costs, and what was fixed

Measured on this machine, not estimated.

| item | size | status |
| --- | ---: | --- |
| `(n, n, K)` feature tensor | **4.94 GB** at `n=6890, K=13`, and **×2 transiently** (`diff * diff` keeps `diff` alive) | **removed** — the fit now uses `SquaredDifferenceFeatures`, which computes `(E, K)` only for the edges the solver requests |
| geodesic matrix `(n, n)` | 380 MB | unavoidable; only at evaluation, one pair at a time |
| EMD cost + POT internals + plan | ~1.1 GB at `n=6890` | unavoidable with exact EMD; this is why the heat maps solve one arm at a time |
| library full-grid index arrays (`all_rows`/`all_cols`, `all_keys`+`setdiff1d`) | 16–24 B/edge — 760 MB–1.1 GB at `n=6890`, **per master solve** | **removed** — `maximum_slack_grid` / `maximum_slack_outside_grid` generate indices per row block |
| `repair_point`'s full-grid cost | `(m·n, K)` in one shot = **4.9 GB at n=6890**, allocated on **every** master solve | **removed** — only `max σ` is needed, so it is computed blockwise |
| first round (scan + LP assembly) | +897 MB at `n=2000` | scales with `|F|`; ~75 kB per edge |

**What is fixed**: `fit_theta` no longer builds `(n, n, K)`. Verified: the whole box-width sweep re-ran and every
shared metric is **identical to 1e-12** (bootstrap rows bit-identical), so the change is memory-only.

**Also fixed (library)**: `loop_regret` no longer materialises the indices of all `m·n` edges. Measured at
`n = 3000`: peak **3561.8 → 3423.9 MB** (−137.9 MB, matching the two 144 MB int64 arrays), with the fitted
`regret` **identical to the last digit** (`0.234810725563`) and the exact-iOT certificate unchanged
(`delta = 0.009580575581`, `is_optimal = True`). `tests/iot_maximum_slack_test.py` pins the equality, including a
case where the maximum sits on the **last row** — the failure mode that would silently under-state the
certificate's upper bound `U`.

**Peak after the fix, measured with `memray`** (which located it: `features.py:180/181` and
`repair_point` on the full grid, not the LP matrix):

| n | peak before | peak after |
| ---: | ---: | ---: |
| 2000 | 2116.9 MB | **1565.4 MB** |
| 5000 | 7239.7 MB | **1612.1 MB** |
| **6890** | ≈11 GB (extrapolated, never completed) | **1628.1 MB** |

Memory is now essentially **flat in `n`** (1612 → 1628 MB from 5000 to 6890): the remaining terms are
fixed-size blocks. `delta`, `is_optimal`, relaxed `theta`/`regret`/`max_sigma_all` and the FAUST fit are
**bit-identical** before and after (`baseline.py`, diffed).

**What is not**: memory is **independent of `|F|`** (measured: tripling `|F|` at fixed `n` left the peak flat), but
it still grows with `n` — the remaining terms are the `n²` geodesic/cost matrices in the evaluation and one
~1 GB transient inside the relaxed loop that I could not attribute with `VmRSS` instrumentation. The subsample
(`--n-sub`, default 800) is therefore not a convenience — it is what makes the fit fit. The heat map itself still
evaluates at full resolution, because that path only ever holds `n × n`.

## What is not done

- One draw of the subsample, «800 of 6890 vertices», three pairs. Not a held-out evaluation.
- No `θ` learned on one pair and tested on another — the only `θ` so far is the oracle ceiling.
- No Gromov–Wasserstein baseline, no pose-distance stratification, no accuracy-vs-deformation curve.
- **Error is Euclidean, not geodesic**, normalised by the bounding-box diameter. No geodesic solver is
  installed; this is a proxy, and it is optimistic.
- No multi-pair shared-`θ` solver (block-restricted scanning is a code change that has not been made).

## Reproduce

**Every run goes through `iot_experiments/scripts/run_job.sh`** (AGENTS.md §0.2 / `scripts/README.md`): it
records the exit code, refuses to repeat a failed command, checks `MemAvailable` against a declared budget
plus a 2 GB reserve, and runs at `nice 19 / ionice idle`.

**Threads are capped.** `OMP_NUM_THREADS`, `MKL_NUM_THREADS` and `OPENBLAS_NUM_THREADS` are set to **4** in
every command below. This machine has 16 cores and is shared: an uncapped `6890 x 6890` matrix product inside
the heat-map renderer will take all of them.

```bash
cd /home/agent/Documents/fast-iot
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4

# gate 6.0 -- topology and vertex order (~3 s, < 1 GB)
scripts/run_job.sh --name faust-data-check --budget-gb 2 -- \
  .venv/bin/python -u -m iot_experiments.studies.real.faust.data_check

# gates 6.1-6.3 -- headroom (~20 s, < 2 GB)
scripts/run_job.sh --name faust-gates --budget-gb 3 -- \
  .venv/bin/python -u -m iot_experiments.studies.real.faust.gates --n-sub 800 --pairs 17 65 1 94 23 74

# the experiment -- subject-disjoint transfer, 3 vertex subsamples (~6 min, < 4 GB)
scripts/run_job.sh --name faust-corr --budget-gb 5 -- \
  .venv/bin/python -u -m iot_experiments.studies.real.faust.correspondence \
  --train-subjects 0 1 2 3 4 --test-subjects 5 6 7 8 9 --train-pair 0 13 --seeds 0 1 2 --n-sub 800

# box width chosen on held-out validation pairs (~7 min, < 4 GB)
scripts/run_job.sh --name faust-box --budget-gb 5 -- \
  .venv/bin/python -u -m iot_experiments.studies.real.faust.correspondence \
  --study faust_correspondence_box --rels 0.05 0.1 0.25 0.5 0.9 --simplex 1 --seeds 0 1 2 --n-sub 800

# heat maps at full 6890-vertex resolution + left/right flip counts (~10 min, < 6 GB)
scripts/run_job.sh --name faust-heat --budget-gb 7 -- \
  .venv/bin/python -u -m iot_experiments.studies.real.faust.heatmaps --pairs 0 66 1 36
```

`scripts/run_job.sh --skip-preflight` is only for a repeat after a confirmed environment check.

Records: `out/faust_*/records.json` with `run.log` and `preflight.json`, and **copied into
`out/records/faust_*/`** so a report number reproduces from a clone (the `out/` tree is otherwise git-ignored).

**Figures.** `out/faust_heatmaps/pair*-<arm>-<view>.png` plus a sibling `.caption.md` for every one, written by
`analyse.py`. They are diagnostic images, not published figures.

**Tests.** `tests/iot_faust_test.py` — PLY round-trip, truncated-file rejection, area/Laplacian invariants,
geodesic invariants (a surface path is never shorter than the straight line), the mirror map on an exactly
symmetric mesh, and the subsample-index convention of `mirror_indices` (the failure mode that would silently
corrupt a flip count).

## Next

1. A `θ` **fitted on pair (a, b)** and evaluated on held-out pairs — the actual experiment.
2. A geodesic error (needs `potpourri3d` or a Dijkstra approximation, and the convention stated).
3. `K` sweep: cheap descriptors vs +HKS vs +WKS, since the HKS result shows capacity is the binding term.
4. The synthetic distortion layer (bends/twists with fixed vertex identity) for an accuracy-vs-deformation
   curve.
