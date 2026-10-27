# metric_learning — does relaxed iOT beat ordinary metric learning?

**Question.** On identical features and identical protocol, does the relaxed inverse-OT cost
(diagonal θ under a box) beat ordinary supervised metric learning — or does it only tie cosine?

**Why it exists.** Plain InfoNCE was attempted earlier, diverged, and was dropped, so this
comparison had **never actually been run**. This study makes it a runnable, gated control.

## Protocol (identical for every arm)

| item | setting |
|---|---|
| features | DINOv2 `vit_small_patch14_dinov2`, patch tokens pooled 4×4, PCA to K |
| K | 64, 128, 256 |
| geometry | `G2` — statistics fitted on the **unlabelled** images (`registry.fit_geometry`) |
| labelled | m ∈ {50, 100, 300} pairs, 3 draws |
| held-out pool | the unlabelled images, with the m labelled subjects removed from **both** sides |
| selection | hyperparameters / early stopping may use a validation fold of the **labelled** pairs only — never the pool |
| metrics | matching hit, R@1, margin-precision at 50% coverage (`studies/pool` convention) |
| space check | the fit space is asserted equal to the scoring space (unit row norms on both) |

## Arms

| arm | family | objective |
|---|---|---|
| `cosine` | — | reference: uniform θ (= cosine, asserted by the registry) |
| `iot_box` | diagonal θ, box `[0.25/K, 4/K]` | full-LP max-margin (reference) |
| `iot_wide` | diagonal θ, box `[0.05/K, 20/K]` | **skipped**: `fit_full_lp` only implements the iOT box; recorded as `skipped_unsupported_box`, never faked |
| `cca` | closed-form linear maps | ridge CCA, no gradients |
| `infonce_box` | diagonal θ, box `[0.25/K, 4/K]` | **same family as iOT**, objective swapped for symmetric InfoNCE |
| `infonce_wide` | diagonal θ, box `[0.05/K, 20/K]` | wide-box control |
| `infonce_free` | diagonal θ ≥ 0 | no box |
| `residual16` | `xᵀ(I + UVᵀ)y`, r = 16 | richer cross-modal family, `V = 0` so step 0 is exactly cosine |
| `residual64` | same, r = 64 | |

**All gradients come from torch autograd.** Hand-derived Jacobians were wrong three times in this
project (once by a factor of 46); autograd removes that failure mode.

## Pre-registered verdict

An arm beats cosine at a cell iff matching hit is higher on **all 3 draws** by **≥ 3 points** *and*
R@1 agrees in direction. Otherwise tie/loss.

Box differences are **not** interpreted as objective differences unless the wide box shows the
**same ordering**.

## Gates (all in `tests/iot_metric_learning_test.py`)

1. **Planted case is calibrated.** K=64, m=300, 8 signal features, σ=0.05 — after normalising `x`
   *first* and adding the signal as a residual. Oracle θ scores **0.983**, cosine **0.197**. If cosine
   were already high the plant could not distinguish a good arm from a bad one.
2. **Step 0 = cosine** for B/C/D, with the box projection the **identity on uniform θ** (asserted for all
   three boxes). CCA is closed form and is exempt.
3. **InfoNCE receives the similarity `s = xᵀdiag(θ)y`, not `-s`.** Passing `-s` inverts the objective;
   measured, every arm then put weight on dimensions that did *not* agree (hit 0.003). A test pins the sign.
4. **Planted recovery** for all three InfoNCE boxes and both residual ranks.

Measured on the plant (reference): cosine 0.197 → `infonce_box` 0.883 → `infonce_wide` 0.940 →
`residual64` 1.000, oracle 0.983.

## Bugs found while building this (all caught by running)

- InfoNCE gradient hand-derived **wrong by ~46×** (Jacobian of the mean-|z| normalisation).
- Normalisation divided by `mean(z)`, which flips sign when the mean is negative.
- CCA whitening missing `1/sqrt(m)`, giving "canonical correlations" of 3–10 (impossible; must be ≤ 1).
- **InfoNCE given `-similarity` as logits**, inverting the objective.
- The first planted test used σ=1.0, where even the *oracle* scores 0.007 — it could not detect anything.

## Run

```bash
PYTHONPATH=/home/agent/Documents/dpm-inverse-ot python -u \
  -m iot_experiments.studies.metric_learning.run --ks 64 128 256 --ms 50 100 300 \
  --draws 0 1 2 --solver cupdlpx --epochs 100
```

Output lands in `iot_experiments/gpu_bench/out/metric_learning/records.json`.
