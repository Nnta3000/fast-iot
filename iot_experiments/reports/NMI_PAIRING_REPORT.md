**Status: CURRENT (2026-10-26, revised).** One question, one protocol. Every number below states its config.

# NMI + Hungarian pairing on the full pool

**Question.** Can mutual information (NMI) serve as a pairing cost, and does it beat whitened-DINOv2
cosine + OT and the learned cost, on a protocol where the true partners are hidden?

**Protocol (identical for all arms).** The candidate pool is **all 1000 subjects**. The T1 side and the T2
side are each shuffled **independently**, so an arm sees two unordered sets and nothing else. Each arm emits
a 1000×1000 cost decoded two ways — Hungarian and exact EMD — and scored by `pair_acc`, the fraction of
decoded pairs that join the same subject. **Chance = 1/1000 = 0.001.** No arm receives pairing or label
information; whitening and binning statistics use the images only.

## Result

| arm | cost | decoder | `pair_acc` (3 draws) | std |
|---|---|---|---|---|
| `nmi` | `H(A_i) + H(B_j) − I(A_i; B_j)` over full-resolution pixels | Hungarian | 1.0000 | 0 |
| `nmi` | same | EMD | 1.0000 | 0 |
| `cosine_ot` | whitened DINOv2 cosine distance (K=256) | Hungarian | 0.8250 | 0 |
| `cosine_ot` | same | EMD | 0.8250 | 0 |
| `learned` | learned diagonal weight in **G2** (K=256, m=300), **in-sample** | Hungarian | 0.8480 | 0.0127 |
| `learned` | same | EMD | 0.8480 | 0.0127 |

Config: feature `dinov2-pool4`, K=256, m=300, draws 0–2, solver CuPDLPx, `studies/nmi`
(`iot_experiments/out/nmi/records.json`). Hungarian and EMD agree on every arm, so the result is
about the **cost**, not the decoder. Per-draw learned values: 0.857, 0.830, 0.857.

### The learned row does NOT beat cosine once the fitted pairs are removed

The `learned` row above is **in-sample**: the 300 fitted pairs are also in the 1000-subject pool, so the
0.848 vs 0.825 gap is largely recovery of those 300 pairs. Re-scored on **held-out subjects only** (the 700
not used in fitting, excluded from both sides of the pool, each side shuffled independently):

| draw | learned | cosine | diff |
|---|---|---|---|
| 0 | 0.8529 | 0.8600 | −0.0071 |
| 1 | 0.8500 | 0.8529 | −0.0029 |
| 2 | 0.8714 | 0.8529 | +0.0186 |
| **mean** | **0.8581 ± 0.0095** | **0.8552 ± 0.0034** | **+0.0029** |

**No rank claim is made.** The out-of-sample difference is +0.003, learned loses on 2 of 3 draws, and the
in-sample advantage (+0.023) does not survive. Any future "learned beats cosine" statement must be computed
on a subject-disjoint pool; the in-sample number is retained only for comparison.

## This benchmark is saturated and is NOT used to rank methods

**The `nmi` row is not a win.** Same-subject T1/T2 slices are **registered**, so their pixels are nearly
identical; pixel MI measures *image-to-image similarity* and recovers such pairs trivially. The benchmark
therefore cannot compare methods, and this report makes no ranking claim on it. It is retained only as a
sanity check. SynthRAD (registered pairs too) will use it the same way.

The saturation is measurable, not asserted:

* On **cross-subject** pairs the MI cost separates by only **+0.02 SD** (same-subject 3.464 vs cross
  3.461), i.e. the signal that produces 1.000 is *pair identity*, not *content compatibility*.
* Every brain MRI correlates with every other (same-subject Pearson +0.698 vs cross-pair +0.690, a gap of
  0.11 SD), so cross-subject discrimination in pixel space is near zero for any pixel metric.

### The earlier `learned = 0.5423` row was a bug, now fixed

An earlier revision fitted θ in **G2** (standardised + L2) but applied the cost in the **raw unwhitened**
space. The two spaces have different scales, so θ was meaningless there: the same fit scored **0.549** in
the raw space and **0.863** in the space it was fitted in. The row labelled "G2" was therefore neither G0 nor
G2 — it was a θ/space mismatch.

Fixed by `registry.fit_geometry` + `apply_geometry` (fit statistics on the unlabelled set, then transform
any set **without refitting**) and by an **assertion** in the driver: the fit space and the scoring space must
agree, and both must have unit row norms. The assertion passed on every draw and the records carry
`space_asserted: true`.

## NMI robustness: the earlier "collapse" was sample starvation

The first revision reported that NMI collapsed under noise (0.003 at 32×32 / 32 bins). That conclusion was
**wrong**: 1,024 pixels were being histogrammed into 1,024 cells, i.e. one sample per cell.
`studies/nmi/robustness.py`, n=1000, σ added to T2 pixels:

| resolution | bins | σ=0 | σ=0.1 | σ=0.3 | samples/cell |
|---|---|---|---|---|---|
| 192×192 | 32 | 1.0000 | 1.0000 | **0.9640** | 36 |
| 192×192 | 64 | 1.0000 | 1.0000 | **1.0000** | 9 |
| 32×32 | 32 | 1.0000 | — | **0.0030** | **1.0** |

So NMI is robust at full resolution; the 32×32 cell was starved. **Sample count per histogram cell** is the
governing quantity, and it is now the first thing to report for any pixel-MI construction.

## The three earliest NMI constructions were non-paper variants

They were my own constructions, not variants from the literature, and all three failed for reasons now
understood:

| construction | why it failed |
|---|---|
| n rows as samples of one component pair | `cost[i,j]` built from a histogram over **all** rows, so `i` and `j` only appear as bin codes; at 32 bins only **32 rows** are distinguishable, so 1000 subjects cannot be identified |
| components of one candidate pair as its samples | MI of a **true** pair and a **false** pair are **identical** (2.0794 both at 8 components): too few samples to estimate a joint histogram |
| the same, made shuffle-invariant | invariance fixed, but planted recovery stayed at 0.01 — the estimator was still too weak |
| **pixels of one candidate pair** (the classical MI registration objective) | works: planted recovery 1.000, chance on independent data, shuffle-invariant |

Rule extracted: **NMI needs many samples per candidate pair**; below a few hundred it cannot pair, whatever
the normalisation.

## Metric discrimination check (now a harness gate)

`harness/discriminates.py` makes "check a metric before using it" a code invariant. A pairing metric is
usable only if its effect size beats the **same-centre** control, not merely the random floor — the
same-centre control is what exposes "this pairing is better because it is the same scanner/centre".

Applied to the T1/T2 pair-scoring metrics (64×64, n=60, 1 draw), both are **rejected**:

| metric | mean_true | mean_same | mean_random | eff_same | eff_random | usable |
|---|---|---|---|---|---|---|
| `ssim` | 0.5208 | 0.5193 | 0.4912 | **+0.01** | +0.24 | NO |
| `psnr` | 11.7930 | 11.7989 | 11.5234 | **−0.00** | +0.15 | NO |

Both look mildly useful against the random floor and vanish against the same-centre control — exactly the
centre shortcut. At full 192×192 resolution the SSIM gap is 0.00 SD (0.4973 true vs 0.4975 cross) and
SSIM-argmax recovers the true partner at chance.

**Consequence for the planned two-halves study:** with T1 from half A and T2 from half B the true partner is
never a candidate, so `pair_acc` is ~0 for every arm by construction, and SSIM/PSNR cannot distinguish arms.
The study was **not run** on T1/T2; that is recorded here as the finding, and SSIM/PSNR are excluded as
evaluation metrics until a dataset where they pass the discrimination check is in place.

## Bugs found in the harness while doing this

1. **`pair_accuracy` compared `perm[i] == i`.** With both sides shuffled the identity map is meaningless; it
   must be compared against the true-partner **column**. The tell was *exactly* 0.0000 (not a small number)
   for a cost that recovered 76.5% of rows by argmax.
2. **`load_gray` violated its own beartype annotation** (`Int[...]` on a `uint8` array), latent until the NMI
   study became the first harness caller.
3. **`typing.Sequence` / `typing.Callable`** emitted PEP 585 warnings on every import.

## Reproduce

```bash
PYTHONPATH=/home/agent/Documents/dpm-inverse-ot python -u -m iot_experiments.studies.real.t1t2.nmi.run \
    --ks 256 --ms 300 --draws 0 1 2 --solver cupdlpx
PYTHONPATH=/home/agent/Documents/dpm-inverse-ot python -u -m iot_experiments.studies.real.t1t2.nmi.robustness
```

Tests: `tests/iot_nmi_study_test.py` (planted recovery after shuffling, true-partner scoring,
Hungarian ≡ EMD, MI direction, unit-variance whitening, sample-per-cell diagnostic) and
`tests/iot_discriminates_test.py` (accept with signal, reject noise, same-centre control is stricter than
random, input validation).
