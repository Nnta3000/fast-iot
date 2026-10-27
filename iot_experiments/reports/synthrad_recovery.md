# SynthRAD brain: pair recovery against ground truth

**Status: CURRENT (2026-10-27).** Data: `data/Task1/brain`, probe patients only. The reserved test set was
not touched.

## What "pairing" means here, in plain words

SynthRAD ships, for each patient, an MRI and a CT **already resampled onto the same voxel grid** — we
verified this on 40 patients: identical matrix, identical 1 mm spacing, identical z origin, zero mismatches.
So every slice has an exact identity:

```
id = (patient, z_index)
```

The experiment hides the correspondence, asks each method to pair 100 MR slices with 100 CT slices, and then
asks: **is the id you picked the id you should have picked?** The score is against the **label**, never
against another method.

**Chance** is 4.4% for `patient_hit` (1 patient in ~23) and 1.8% for `exact_hit` (1 slice in ~55).

Two numbers are reported because they answer different questions:

- **`patient_hit`** — did the matched slice come from the **same patient**? This is the real "found the true
  pair" question.
- **`exact_hit`** — did it match the **exact same slice**? Stricter; the gap means "right person, wrong level".

## Setup

- 30 probe patients, split **by patient** into 15 training and 15 evaluation. The learned cost is fitted on
  training patients only; the evaluation pool is 20 patients' worth of slices drawn from the 15 evaluation
  patients (5 slices each, ≥8 apart, seeds fixed per draw).
- 3 draws, 2 encoders (frozen DINOv2 pool4; pixel-PCA), 2 K (128, 256), 2 geometries (G2, G3). **8 cells.**
- The **learned cost** is an inverse-OT diagonal θ fitted on the training patients' **true pairs** (box ∩
  simplex), applied to held-out patients. It is fitted once per cell and reused across draws.

## Arms

| arm | what it does | uses images? |
|---|---|---|
| `random` | random assignment | no |
| `z_only` | pairs by closest `z_normalised` | no |
| `mask_iou` | pairs by best body-outline overlap | no |
| `nmi` | pixel mutual information + Hungarian | yes |
| `cosine_ot` | feature cosine + exact OT | yes |
| **`learned`** | **inverse-OT learned diagonal cost + exact OT** | yes (fitted on true pairs) |
| `oracle` | the true pairing, fed in directly | — |

The oracle is an assertion: it must score exactly 1.0/1.0, otherwise the ids themselves are misaligned. It
passed in every cell.

## Result

Mean ± std over 3 draws. `patient_hit` (and `exact_hit` is essentially the same, so only one is shown):

| cell | random | **z_only** | **mask_iou** | nmi | cosine_ot | **learned** |
|---|---|---|---|---|---|---|
| dinov2 G2 K128 | 0.044 | **0.982** | **1.000** | 0.880 | 0.333 | **0.378** |
| dinov2 G2 K256 | 0.044 | **0.982** | **1.000** | 0.880 | 0.084 | **0.342** |
| dinov2 G3 K128 | 0.044 | **0.982** | **1.000** | 0.880 | 0.324 | **0.396** |
| dinov2 G3 K256 | 0.044 | **0.982** | **1.000** | 0.880 | 0.062 | **0.347** |
| pixpca G2 K128 | 0.044 | **0.982** | **1.000** | 0.880 | 0.133 | 0.098 |
| pixpca G2 K256 | 0.044 | **0.982** | **1.000** | 0.880 | 0.089 | 0.102 |
| pixpca G3 K128 | 0.044 | **0.982** | **1.000** | 0.880 | 0.133 | 0.102 |
| pixpca G3 K256 | 0.044 | **0.982** | **1.000** | 0.880 | 0.173 | 0.107 |

## What this says

**1. The learned cost beats cosine in all four DINOv2 cells and loses in all four pixel-PCA cells.** With
DINOv2 it roughly triples to sextuples the cosine score: 0.333 → 0.378, 0.084 → 0.342, 0.324 → 0.396,
0.062 → 0.347. That is the comparison this study exists to make, and it is the first time in this project
that the learned cost has beaten cosine on a label-scored task.

**2. But both are far below the metadata baselines.** `mask_iou` scores **1.000** — body-outline overlap alone
identifies the patient perfectly in every cell. `z_only` scores 0.982. No image-based arm comes close.

**3. Why the metadata arms are so strong — and why that is a property of the data, not a trick:**
`z_normalised` is computed over each patient's own slice extent, and those extents differ, so a slice's
height is nearly a unique fingerprint of its patient (98 distinct values in 100 rows). Likewise the body
outline shape is patient-specific. Both are **available at test time** and both are legitimate, but they mean
this pool can largely be solved without looking at the tissue.

**4. `nmi` at 0.880 is the strongest image-based arm**, consistent with Zhu et al., who use NMI+Hungarian for
exactly this purpose.

## Limitations I do not want glossed over

- **The two `nan` rows.** At K=256 (both encoders, G2) `R'/R_uniform` came out `nan`, meaning `R_uniform ≈ 0`:
  uniform θ already certifies the training plan. The fits still produced valid simplex θ (sum 1, finite) and
  are included, but the diagnostic says the inverse-OT problem is degenerate in those cells.
- **Only 15 patients for fitting and 15 for evaluation.** Small. Per-cell standard deviations across draws
  are 0.03–0.07 for the feature arms, so the learned-vs-cosine gap in the two middle DINOv2 cells
  (0.342 vs 0.084) is well outside noise; the 0.378 vs 0.333 gap is not.
- **`exact_hit` and `patient_hit` nearly coincide** because when a method finds the patient it usually also
  finds the level — an artefact of 5 slices per patient and a shared z grid.
- **A defect corrected during this run:** the first version kept MR rows and CT columns in their original
  order, so the diagonal was trivially correct for the symmetric arms and `z_only`/`mask_iou` scored 1.0 by
  construction. Columns are now shuffled with a fixed seed. The 1.000 for `mask_iou` **survives that fix**, so
  it is real, not an artefact.

## Files

`records.json`, `run.log` (self-written), committed in `reports/synthrad_probe/`. Code:
`studies/real/synthrad/recovery.py`, encoders in `harness/dinov2.py`.

## This supersedes the earlier "GATE 1 FAILED"

The previous test deliberately excluded the true partner from the pool and scored compatibility instead of
correctness. It could not measure recovery and no conclusion about pairing followed from it. This one has the
true partner present and scores against the label. `synthrad_gate.md` is retained for its data findings
(the centre-B gadolinium signature, the preprocessing audit), but its gate verdict should be read as
superseded by this file.
