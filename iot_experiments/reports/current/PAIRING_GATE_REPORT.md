# Pairing gate — does transport beat NICP's argmin?

**Status: RESULT OBTAINED (2026-11-13).** Runs on OUR descriptor cost, not NSR's. It is the control for the
NSR gate, not a replacement for it.

Driver: `studies/real/faust/pairing_gate.py` · records: `out/faust_pairing_gate/`,
`out/faust_pairing_gate_learned/` · decoders: `harness/nicp_pairing.py`

## Question

NICP's correspondence step is, per target point `k`, an independent decision:

```
i*_k = argmin_i ||F_theta(y_k)_i||^2
```

Only the **row sums** are constrained. Nothing forces the result to be a map. Replace it with a balanced
transport plan, which adds the column constraint:

```
P* = argmin_{P>=0} <P, C>   s.t.  P 1 = a,  P^T 1 = b
```

Does that change the correspondence, and does it help?

## Protocol

| item | setting |
|---|---|
| data | FAUST training registrations, `pi_hat = I/n` is exact ground truth |
| pairs | `000-066`, `001-036`, `050-066`, `023-074`, `002-041` (all inter-subject) |
| subsample | `n = 800` shared vertices of 6890 |
| descriptors | `K = 13` (xyz, log area, log curvature, 8 HKS) |
| θ | **fixed across both arms** — uniform, or fitted once on the source pair |
| error | geodesic on the target / `sqrt(surface area)` |
| decoding | row `argmax` of the exact LP plan (`ot.emd`, network simplex) |
| isolated | only the pairing rule differs between arms |

## Result — uniform θ (5 pairs)

| pair | argmin mean | transport mean | Δ | argmin miss@5 % | transport miss@5 % | argmin exact | transport exact |
|---|---:|---:|---:|---:|---:|---:|---:|
| 000-066 | 0.3407 | **0.1963** | −42 % | 0.751 | 0.526 | 0.149 | **0.312** |
| 001-036 | 0.1611 | **0.1081** | −33 % | 0.569 | 0.330 | 0.266 | **0.477** |
| 050-066 | 0.3202 | **0.1764** | −45 % | 0.738 | 0.487 | 0.191 | **0.384** |
| 023-074 | 0.2332 | **0.1973** | −15 % | 0.733 | 0.601 | 0.159 | **0.278** |
| 002-041 | 0.0937 | **0.0699** | −25 % | 0.535 | 0.370 | 0.239 | **0.381** |

**Transport wins on all five pairs, by 15–45 %.**

## Result — learned θ (2 pairs)

θ fitted on the source pair (`rel=None`, simplex), then **held fixed** across arms:

| pair | argmin mean | transport mean | Δ | argmin miss@5 % | transport miss@5 % | argmin exact | transport exact |
|---|---:|---:|---:|---:|---:|---:|---:|
| 000-066 | 0.1693 | **0.0521** | **−69 %** | 0.354 | **0.125** | 0.395 | **0.719** |
| 050-066 | 0.1225 | **0.0357** | **−71 %** | 0.379 | **0.116** | 0.357 | **0.736** |

Exact hits roughly **double**. Missing at 5 % falls by roughly **two thirds**.

## Why it works — the diagnostics

| pair (uniform) | argmin coverage | transport coverage | argmin imbalance | transport imbalance |
|---|---:|---:|---:|---:|
| 000-066 | 0.347 | **1.000** | 20.0 | **1.00** |
| 001-036 | 0.474 | **1.000** | 17.0 | **1.00** |
| 050-066 | 0.370 | **1.000** | 28.0 | **1.00** |
| 023-074 | 0.379 | **1.000** | 30.0 | **1.00** |
| 002-041 | 0.463 | **1.000** | 16.0 | **1.00** |

`imbalance = max_load / (K/M)`. Under argmin one template vertex absorbs **16–30×** its fair share, and
**only 35–47 % of template vertices are used at all**. The argmin is not a correspondence; it is a
many-to-one collapse. The transport plan is balanced by construction.

## What this establishes, and what it does not

**Establishes:** given a pairwise cost, replacing per-row argmin with a balanced transport plan is a large,
consistent improvement on this task (15–71 % depending on θ), and the mechanism is visible in the load
diagnostics.

**Does not establish:**

1. **Anything about NSR.** This is our diagonal descriptor cost, not NICP's offset cost
   `C[k,i] = ||F(y_k)_i||^2`. NICP's cost may already be well-behaved (its own iterative refinement pushes
   offsets to zero), in which case the argmin is less pathological and the gain will be smaller. **The NSR
   experiment is still required.**
2. **Anything at full resolution.** `n = 800` of 6890.
3. **Anything about partial/unreliable regions.** Balanced transport assumes all mass must be matched;
   FAUST has missing regions. A dustbin formulation is deliberately not implemented yet.

## Consequence for the plan

If the same effect holds on NSR's cost, the insertion point is the **final pairing**, and the contribution is
"global assignment improves a SOTA registration prior". If it does **not** hold, the honest conclusion is that
NICP's prior is already good enough that global consistency adds nothing — also a real result, and one worth
knowing before investing in inverse-OT cost learning.
