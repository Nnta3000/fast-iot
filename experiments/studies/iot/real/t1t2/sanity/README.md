# sanity — do the relaxed / EOT fits pass the basic sanity checks?

**Question.** Before anything is read into a hit rate: does the fit carry signal (capacity), is the
learning curve saturated or still rising (sample-limited vs feature-limited), does the OT structure beat a
rows-only reweighting (the right baseline), does a cost trained at one pool size transfer to another, and
is `theta` even identifiable (row-only / column-only features, scale)?

**Origin.** These are the checks the user sent on 2026-11-05. Which had been done, partially done, or never
done is the subject of `reports/current/SANITY_CHECKS_REPORT.md`; this directory holds the runs for the
ones that had not been run.

**Run status (2026-11-05).** A and E **ran**; B was running at the end of the session; C and D are written
and not launched. Per-check detail and the numbers that exist are in the report, never guessed here.

**What the runs changed in the code.** `run.py` uses `harness.solver.fit_full_lp` (F2's own diagonal
engine) for the relaxed fit. It first used `quadratic_f2._fit_loop`, which is `QuadraticFeatures`
(`|theta| = d(d+1)/2 = 2080` at `d = 64`) — a `ValueError` against a 64-wide point array, caught on the
first launch and fixed.

**Instance of a silent-failure class, from check E.** A cost with `theta` on columns that are a function of
one side only (`u_i` or `v_j`) is absorbed by the potentials: it looks plausible and does not certify the
observed plan (`R_ratio = 1.58` on the "textbook" construction). `ident.py` records this rather than
smoothing it.

**Data.** The real 1,000 T1/T2 pairs, `dinov2-pool4` PCA features, the same 400-pair pool and G2 geometry
as F2 (`studies/iot/real/t1t2/metric_learning`). Sub-studies A–C share F2's protocol verbatim so their rows are
directly comparable to F2. E is synthetic on purpose: there the row-only and column-only features are a
construction, not an assumption.

| file | check | `--check` / entry | needs |
| --- | --- | --- | --- |
| `run.py` | **A capacity null**, **B learning curve**, **C three losses** | `--check null\|curve\|baselines` | GPU (encoder + CuPDLPx) |
| `transfer.py` | **D pool-size transfer** | `-m experiments.studies.iot.real.t1t2.sanity.transfer` | GPU (encoder + CuPDLPx) |
| `ident.py` | **E identifiability and scale** | `-m experiments.studies.iot.real.t1t2.sanity.ident` | CPU only (HiGHS reference) |

Each writes `out/sanity*/<check>/run.log`, `preflight.json` and `records.json` (or `out/sanity*/records.json`
for the two single-entry scripts); the verdict rules are fixed in each file's `PROVENANCE` before the run.

## Commands

```bash
cd /home/agent/Documents/fast-iot

# A -- the permutation null: true vs permuted pairing at the same (K, m)
.venv/bin/python -u -m experiments.studies.iot.real.t1t2.sanity.run \
  --check null --ks 64 128 256 --ms 50 100 300 --draws 0 1 2

# B -- the learning curve, extended past F2's m = 300 to the whole labelled pool
.venv/bin/python -u -m experiments.studies.iot.real.t1t2.sanity.run \
  --check curve --ks 64 128 256 --ms 50 100 300 600 --draws 0

# C -- the same objective under three losses (row-softmax / Sinkhorn / exact iOT)
.venv/bin/python -u -m experiments.studies.iot.real.t1t2.sanity.run \
  --check baselines --ks 64 128 256 --ms 50 100 300 --draws 0 1 2

# D -- pool-size transfer (one EMD solve per pool size per cost)
.venv/bin/python -u -m experiments.studies.iot.real.t1t2.sanity.transfer --ks 64 128 256 --ms 100 300

# E -- identifiability under centring, and scale (CPU only)
.venv/bin/python -u -m experiments.studies.iot.real.t1t2.sanity.ident
```

## Memory and time budget

**A, B, C** each hold the DINOv2 encoder (~1.5 GB GPU) plus one feature array per `K`. The relaxed loop
evaluates `sigma` over all `m x n` edges: at `m = 300, n = 400` that is 120k edges, at `m = 600` 240k. Peak
**≈ 3 GB GPU and ≈ 3 GB host RAM**. Estimated wall: A ≈ 20–30 min (6 fits × 3 `K`), B ≈ 15 min (4 `m` × 3
`K`), C ≈ 30 min (two torch arms with a `tau × epochs` CV per cell, plus `iot`).

**D** needs no `m x n` grid, only `sum_C C^2` POT solves per cost: ~1.3e6 at `C = 1000`, ≈ 0.2 s each on
4 threads → **10–20 min per `(K, m)`**, so run it with `--ms 100 300` for one `K` first.

**E** is `m = n = 120`, `K = 8`, HiGHS, **seconds**, < 1 GB.

**Do not launch A–D while another GPU job is running.** Measured on 2026-11-05: with two other `python`
jobs resident (one holding 4.2 GB of the GPU), the DINOv2 encoder pass did not reach its first log line in
4 minutes, versus 7.7 s for the same 1,500-image pass in the SynthRAD run log. `MemAvailable` was 3 GB
against an estimated 3 GB peak — below the required 2 GB reserve.

## What each check decides

- **A** answers "is the fit signal or noise?" — per cell, `TRUE - PERM >= 0.03` on every draw.
- **B** answers "were we data-starved?" — SATURATED if `hit(600) - hit(300) < 0.03`, RISING otherwise.
- **C** answers "is it the OT structure or only the reweighting?" — `iot` vs `rowsoftmax`, the
  pre-registered comparison.
- **D** answers "does training on `m` pairs transfer to a different pool size?" — `CS - hit` per pool size,
  never pooled across sizes.
- **E** answers "can a learned `theta` mean anything?" — null mass before/after double-centring, and the
  `L1` move under unequal feature scales before/after standardisation.
