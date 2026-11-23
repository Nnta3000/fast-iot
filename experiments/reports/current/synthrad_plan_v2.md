# SynthRAD brain — evaluation protocol v2 (agreed 2026-10-28)

**Status: CURRENT.** Supersedes the proxy-metric gate. Written down so the protocol survives context loss.

## Why the change

The earlier gate scored a chosen CT against the true CT with SSIM/PSNR/NMI/IoU and declared pass/fail. It had
**no validated discriminating power** — a body-outline matcher with no image content scored 1.000 on it, so
it could not separate methods. It is **dropped**.

**The yardstick is now the ground-truth pairing itself**: SynthRAD supplies registered CT/MR pairs on an
identical voxel grid, so every slice has an exact identity `(patient, z_index)` and we can ask whether a
method recovered it. No method is used as the reference for another.

## Primary metric

Patient-disjoint **pair recovery**, with the true partners **in the pool**:

| metric | meaning |
|---|---|
| `hit` | exact-OT matching hit: fraction of matched pairs whose id equals the true id |
| `R@1` | is the true partner the top-scoring candidate for that row? |
| `P@50` | precision among the 50% of rows the method is most confident about (margin) |

Chance is `1/n` for a random permutation, where `n` is the pool size **per side**.

## Patients

- **Fit: 60 patients** (20 per centre A/B/C).
- **Evaluate: 30 other patients** (10 per centre). Disjoint from the fit set.
- **The 150 Phase-1 reserved patients stay untouched.**
- Pools: ≤ 10 slices per patient, ≥ 8 slices apart, **≥ 600 slices per side**.
- **≥ 3 draws** = different patient subsets; **the same evaluation patients are used across arms** within a
  draw, so arms are compared on identical pools.

## Arms (feature-based only)

| arm | note |
|---|---|
| `cosine` | G2/G3, K ∈ {64,128,256}, CT windows soft and bone |
| `infonce` | diagonal InfoNCE, box ∩ simplex |
| `resid16`, `resid64` | low-rank residual bilinear `xᵀ(I + UVᵀ)y` |
| `iot` | diagonal relaxed inverse OT; `rel` chosen on a **validation fold of fit patients**; solver-limited fits flagged |
| `nmi` | pixel NMI + Hungarian — **one labelled reference row only** |

**On the NMI reference row (correction).** SynthRAD pairs are *registered on a shared grid*, so pixel NMI is
close to an oracle here: it is a **registration** metric being asked a registration question. Zhu et al. use
NMI to pair **different patients** from an unpaired set, which is a harder and different task. The NMI row is
therefore kept as a labelled reference — it shows how much of this benchmark is answerable by registration
alone — and it must **not** be read as corroborating Zhu et al.

**Dropped as competitors:** `mask_iou` and `z_only`. Reason, stated in the report: for registered pairs both
are **tautological** — the mask and the slice index are properties of the same patient on a shared grid, so
using them as pairing evidence measures the grid, not compatibility. They are reported as descriptive
context only if needed, never as competing methods.

Hyperparameters are chosen on a **validation fold of FIT patients**, never on the evaluation patients.

## Reporting

1. **Within-centre and across-centre recovery** separately — a centre shortcut must be visible, not hidden.
2. **Uniform-random floor explained**: the 0.044 in the earlier table is `~1/23` because it was measured with
   ~23 patients per side, not `1/n` of slices. State the correct chance for every table.
3. **Recovery vs pool size**: n ∈ {1k, 2k, 5k, 10k, 20k, larger if feasible} for `cosine+OT` (our solver;
   Hungarian/POT where feasible) vs `cosine` nearest-neighbour. Report wall time, peak memory per method per n,
   and the **primal-dual gap** for the exact solver.
4. **The K=128 → K=256 cosine drop** (0.333 → 0.084) investigated: cosine at K ∈ {32, 64, 128, 256, 512},
   with and without whitening, on the larger fit set.
5. Final table: **arm × K × setting (within/across centre)**, mean ± std over draws, oracle assertions.
6. No E4/E5 work.

## Oracle assertion

Feed the true pairing into the scorer: `hit = R@1 = 1.0` exactly, or the ids are misaligned and the run is
invalid. Asserted in every cell.

## Acceptance bar (changed after seeing results)

The original bar was `U − L ≤ 1e-2 × R_uniform`. It is now **`U − L ≤ 5% × R_uniform`**, applied to every
run. This is a **threshold change made after observing** that m=300 landed at 1.4% and m=2000 at 1.1% of
`R_uniform`, i.e. both were near the old bar and would have been rejected on a few percent. It must be
stated as such wherever the bar is used.

## Known limitations to restate in the report

- The Phase-1 probe result (15 fit / 15 eval patients) is small; this protocol uses 60/30.
- Two inverse-OT cells previously gave `R'/R_uniform = nan` (uniform θ already certifies the fit plan); such
  fits must be flagged, not silently included.
