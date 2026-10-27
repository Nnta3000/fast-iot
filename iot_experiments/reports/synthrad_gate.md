# SynthRAD brain: pairing gate (Phase 2)

**Status: CURRENT (2026-10-27). Result: GATE 1 FAILS.** Data: `data/Task1/brain`, probe patients only.

This file states plainly what was run, because the shorthand arm names are not self-explanatory.

## What was paired

150 MR slices against 150 CT slices. The MR slices come from patients in one half of the probe set and
the CT slices from patients in the **other half**, so **no patient appears on both sides** and every
candidate pair joins two different people. Matching is an assignment: each MR gets exactly one CT and
each CT exactly one MR, so every method returns a permutation.

**Pool size caveat:** the probe set is 30 patients (10 per centre), and the rule "at most 10 slices per
patient, at least 8 slices apart" therefore yields **300 slices, i.e. 150 per side**. The specified target
of 1000 per side needs roughly 100 patients and was **not** reached. Every number below is at 150/side.

## The arms, in plain words

| arm | what it does |
|---|---|
| `random` | pairs at random. The floor: a method that does not beat this is worthless. |
| `z_only` | pairs by position in the head only: the CT whose `z_normalised` is closest to the MR's. No image content. Solved as an assignment. |
| `mask_iou` | pairs by body-outline overlap only: the CT whose mask overlaps the MR's mask most. No image content. |
| `nmi` | pairs by pixel mutual information: "which CT's brightness pattern co-varies most with this MR's?" |
| `cosine_nn` | nearest neighbour in feature space (not an assignment — several rows may pick the same column). |
| `cosine_ot` | the method under test: nearest neighbour in feature space, but forced into a one-to-one assignment by exact optimal transport. |

Two encoders, both producing one shared feature space for MRI and CT (PCA fitted on all pool slices of
both modalities, label-free):

- **`dinov2`** — frozen DINOv2 ViT-S/14, patch tokens pooled to a 4×4 grid (6144 dims), then PCA to K.
  This is the encoder specified for the experiment.
- **`pixpca`** — PCA on the raw 192×192 pixels. This is what an earlier version used; kept for comparison.

Geometry (`registry`): **G2** = standardise with pooled statistics of both modalities; **G3** = standardise
per modality. CT windows: **soft** = −200…300 HU; **bone** = −200…1500 HU.

## The measures, in plain words

| measure | question it asks | used by any arm? |
|---|---|---|
| `z` | how close in height were the paired slices? | — (pure geometry) |
| `mask_iou` | how much do the two body outlines overlap? | — (pure masks) |
| `nmi` | do the brightness patterns co-vary more than chance? | yes, `nmi` optimises it |
| `ssim` | how similar is the chosen CT to **this patient's real CT**? | no |
| `psnr` | same question, in decibels | no |

`ct_mask_iou` was removed: it computed the same mask comparison as `mask_iou` on the same mask pair and
produced numerically identical values, so it was one piece of evidence counted twice.

**The gate is judged on `ssim` and `psnr` only.** They are the only measures that compare against the true
partner, and no arm optimises them. `z` is reported descriptively: `z_only` maximises `z` by construction,
so using `z` as a pass/fail criterion would grade a height-sorter on its own answer key. That was a defect
in the earlier version of this gate.

## Measuring the measures first

Before scoring any method, each measure was checked on four kinds of pair: same patient same slice;
same centre same height (within 5%); same centre different height (more than 20% apart); different centre.
A measure survives only if it separates same-height from different-height **within the same centre** by
at least 0.5 standard deviations, so that spotting the scanner cannot be what makes it look good.

All five measures passed (effect sizes: `z` +2.33 SD, `psnr` +1.76, `ssim` +0.90, `mask_iou` +0.82,
`nmi` +0.56). It is a weak filter — `mask_iou` and `z` are near-constant across mid-brain slices — but it
did not have to reject anything here.

## Result

`cosine_ot` fails in **all 16 configurations** (2 encoders × 2 K × 2 geometries × 2 windows), on both
honest measures. Representative numbers, soft window, mean over 3 draws:

| cell | random | z_only | mask_iou | nmi | cosine_ot |
|---|---|---|---|---|---|
| dinov2 G2 K128 | 0.4436 | 0.6340 | **0.6980** | 0.6419 | 0.5179 |
| pixpca G2 K128 | 0.4436 | 0.6340 | **0.6980** | 0.6419 | 0.6147 |

(SSIM. The pattern for PSNR is identical.)

Best SSIM for `cosine_ot`: **0.6271** (pixpca, G2, K=128) and **0.5414** (dinov2, G3, K=256), against
**0.6980** for the mask baseline and **0.6340** for height alone.

## What this says, and what it does not

**Supported:** at this scale, cross-patient MR→CT pairing is better served by body-outline overlap and
slice position than by either encoder tested. Both beat every feature arm on both honest measures.

**Supported:** DINOv2 is *worse* than pixel-PCA here (best SSIM 0.541 vs 0.627). A 4×4 pooled grid
averages away the spatial detail that distinguishes one anatomical level from another; raw pixel PCA keeps
it. This is the opposite of the T1/T2 result, where DINOv2 was decisive — so the encoder choice is
task-dependent and cannot be carried over by default.

**Not supported by this run:** any claim about whether features and metadata are *complementary*. A
combined arm was not run (explicitly out of scope). It remains unknown whether adding height to the
feature cost would beat either alone.

**Not tested:** a pool of 1000 per side. A larger pool makes the task harder in a way that could help or
hurt the feature arms; that is unknown, not assumed.

## Files

- `records.json` — every arm × cell × draw, with timings.
- `metric_gate_soft.json`, `metric_gate_bone.json` — the measure-check effect sizes.
- `run.log` — timestamped run log written by the job itself.
- Code: `iot_experiments/studies/real/synthrad/gate1.py`; encoders in
  `iot_experiments/harness/dinov2.py`; measures in `iot_experiments/harness/e1_metrics.py`.

## E4 was not started

The pre-registered rule says do not start E4 if the gate fails. It failed.
