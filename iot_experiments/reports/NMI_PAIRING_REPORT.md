**Status: CURRENT (2026-10-26).** One question, one protocol. Every number below states its config.

# NMI + Hungarian pairing on the full pool

**Question.** Can mutual information (NMI) serve as a pairing cost, and does it beat whitened-DINOv2
cosine + OT and the learned cost, on a protocol where the true partners are hidden?

**Protocol (identical for all three arms).** The candidate pool is **all 1000 subjects**. The T1 side and
the T2 side are each shuffled **independently**, so an arm sees two unordered sets and nothing else. Each
arm emits a 1000×1000 cost and is decoded two ways — Hungarian assignment and exact EMD — and scored by
`pair_acc`, the fraction of decoded pairs that join the same subject. **Chance = 1/1000 = 0.001.**
No arm receives any pairing or label information; whitening and binning statistics use the images only.

## Result

| arm | cost | decoder | `pair_acc` (3 draws) | std |
|---|---|---|---|---|
| **`nmi`** | `H(A_i) + H(B_j) − I(A_i; B_j)` over pixel histograms | Hungarian | **1.0000** | 0 |
| **`nmi`** | same | EMD | **1.0000** | 0 |
| `cosine_ot` | whitened DINOv2 cosine distance | Hungarian | 0.8250 | 0 |
| `cosine_ot` | same | EMD | 0.8250 | 0 |
| `learned` | learned diagonal-weight L2², K=256, m=300 | Hungarian | 0.5423 | 0.0164 |
| `learned` | same | EMD | 0.5423 | 0.0164 |

Config: feature `dinov2-pool4`, K=256 (learned arm; the NMI arm uses 32×32 pixels), m=300, draws 0–2,
solver CuPDLPx, `studies/nmi` (`iot_experiments/gpu_bench/out/nmi/records.json`).
Hungarian and EMD agree on every arm, so the result is about the **cost**, not the decoder.

## What the NMI number actually is — read this before citing 1.000

The perfect score is real on this data but it is **not** the semantic pairing result that the cosine and
learned arms report, and it is fragile:

| variant | `pair_acc` |
|---|---|
| 32×32 pixels, 32 bins (reported) | 1.0000 |
| 16×16 pixels, 16 bins | 0.9100 |
| 32×32 pixels + additive noise (σ=0.3), 32 bins | **0.0030** |

The data are **same-subject slice pairs**: the T1 and T2 images of one subject are nearly identical
pictures. MI over raw pixels measures **image-to-image similarity**, so it recovers near-identical pairs
almost perfectly and fails completely as soon as the two images stop being near-identical (the σ=0.3 row).
The cosine and learned arms operate on learned features, which degrade gradually instead of collapsing.
So: **NMI wins at the "are these two pictures of the same thing" task and loses the semantic task.** On a
genuinely unpaired pool (different acquisitions, a modality gap) the pixel-NMI arm should be expected to
behave like the noise row, not like the reported row.

## Three NMI constructions that failed, and why

This is the substance of the study. Only the fourth construction is defensible.

| construction | samples per pair | failure |
|---|---|---|
| n rows as samples of one component pair | n = 1000 | `cost[i,j]` built from a joint histogram over **all** rows, so `i` and `j` appear only as bin codes; at 32 bins only **32 rows are distinguishable**, so it cannot identify 1000 subjects. Scored exactly 0.0000 (below chance). |
| components of one candidate pair as samples | K = 16 or 256 | MI of a **true** pair and of a **false** pair are **identical** (measured 2.0794 for both at 8 components): a handful of samples cannot estimate a joint histogram, and MI reduces to a function of marginal bin occupancy. Scored at chance. |
| same, but "fixed" to be shuffle-invariant | K = 16 | Became shuffle-invariant and chance-level on independent data (both correct), but planted recovery fell to **0.01** — the invariance was right, the estimator was still too weak. |
| **pixels of one candidate pair** | **~1k–16k** | **works**: planted recovery 1.000, chance on independent data, shuffle-invariant. This is the classical MI image-registration objective. |

The rule that came out of it: **NMI needs many samples per candidate pair.** If a construction has fewer
than a few hundred samples per pair it cannot pair, no matter how it is normalised.

## Bugs found and fixed in the harness while doing this

1. **`pair_accuracy` compared `perm[i] == i`.** With both sides shuffled the identity map is meaningless;
   it must be compared against the true-partner **column**. The first version printed exactly chance for a
   cost that provably recovered pairs (argmax hit 76.5% of rows) — the tell was *exactly* 0.0000, not a
   small number.
2. **`load_gray` violated its own beartype annotation** (`Int[np.ndarray, ...]` on a `uint8` array). It was
   latent because nothing called it through the harness; the NMI study was the first caller.
3. **`typing.Sequence` / `typing.Callable`** in the harness emit PEP 585 deprecation warnings on every
   import; switched to `collections.abc`.

## Reproduce

```bash
PYTHONPATH=/home/agent/Documents/dpm-inverse-ot \
  .venv/bin/python -u -m iot_experiments.studies.nmi.run \
    --ks 256 --ms 300 --draws 0 1 2 --solver cupdlpx
```

Runtime: about 2 minutes (the n=1000 pixel-NMI cost is ~22 s on CPU). Tests:
`tests/iot_nmi_study_test.py` (5 claims: planted recovery after shuffling, true-partner scoring,
Hungarian ≡ EMD, MI direction, unit-variance whitening).
