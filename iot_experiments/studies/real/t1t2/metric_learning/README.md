# metric_learning — does relaxed iOT beat ordinary metric learning?

**Solver:** all learned-cost fits use `harness/solver.fit_full_lp` (dense LP over all `m*m` edges,
CuPDLPx). At `m <= 300` the dense LP and the constraint-generation loop solve the same problem (the K=32
equivalence check passed), so these tables are valid for that range.

**Question.** On identical features and an identical protocol, does the relaxed inverse-OT cost (diagonal θ
under a box) beat ordinary supervised metric learning — or only tie cosine?

**Terminology.** The relaxed LP is a **regret minimiser**, not max-margin. It minimises
`R(θ) = <θ, g(π̂)> − R_uniform`; every iOT fit reports `R′/R_uniform` and the fraction of θ on a box bound.

**Why it exists.** Plain InfoNCE had been attempted, diverged, and dropped, so this comparison had never
actually been run.

## Protocol

| item | setting |
|---|---|
| features | DINOv2 `vit_small_patch14_dinov2`, patch tokens pooled 4×4, PCA to K |
| K | 64, 128, 256 |
| geometry | `G2` — statistics fitted on the **held-out pool** (unlabelled) |
| labelled | m ∈ {50, 100, 300}, 3 draws, drawn from subjects **not** in the pool |
| held-out pool | **one fixed 400-pair pool shared by every m** (`--holdout-pairs 400 --holdout-draw 0`) |
| metrics | matching hit, R@1, margin-precision @ 50% coverage |
| space check | fit space asserted equal to the scoring space (unit row norms on both) |
| run | 2421 s wall; log and records in `out/records/metric_learning/` |

**Cosine is label-free and therefore constant.** It sees only the fixed pool, so for a given K it must give
exactly the same score at every m and draw. Asserted in the study and verified: K=64 → 0.5675,
K=128 → 0.6800, K=256 → 0.8800, spread **0.0** in every cell. Any variation would mean the pool, the
geometry statistics or the shuffle had been contaminated by m or draw.

**Hyperparameters are chosen inside the labelled pairs only** — never on the pool:

- iOT `rel` ∈ {0.25, 0.5, 0.75, 0.9, 0.05} by 5-fold on the labelled pairs.

  **`rel` is INVERSE to box width.** The box is `[rel/K, 1/(rel·K)]`, so a *smaller* `rel` gives a
  *wider* box:

  | rel | window at K=64 | hi/lo |
  | --- | --- | --- |
  | 0.05 | [0.00078, 0.31250] | 400x (widest) |
  | 0.25 | [0.00391, 0.06250] | 16x |
  | 0.5 | [0.00781, 0.03125] | 4x |
  | 0.9 | [0.01406, 0.01736] | 1.2x (narrowest) |

  So `0.05` is the **widest** setting, and a CV that picks `0.05` is asking for *more* freedom, not less.
  (An earlier revision of this file called it "0.05(wide)", which was right; a later comment elsewhere
  claimed small `rel` was narrow and is corrected here.) The conclusion it carried was wrong in
  direction too: a CV that returns `0.05` is saying the arm wants *more* box, not that it wants
  to shrink toward uniform.
- InfoNCE / Sinkhorn / residual `lr` ∈ {0.01, 0.05}, `τ` ∈ {1.0, 0.1}, and **epochs from
  {0, 1, 2, 5, 10, 20, 40, 60, 100}** by 5-fold, with **epoch 0 a candidate** so early stopping can never
  return a checkpoint worse than step 0 (= cosine);
- CCA `ridge` ∈ {1e-4, 1e-3, 1e-2, 1e-1} by 5-fold; the chosen value is reported per cell.

**Early stopping never sees the gradient rows.** For every iterative arm the validation fold is
**disjoint** from the fold used in the gradient step; `select_and_fit_diagonal` and `select_residual` raise if
`intersect1d(train, val)` is non-empty, and `tests/iot_metric_learning_test.py` pins the partition
(disjoint, covers all pairs, monotone indices). The selected checkpoint is then refit on **all** labelled
pairs.

**All gradients come from torch autograd.** Hand-derived Jacobians were wrong three times in this project
(once by ~46×), so any InfoNCE reading from that era is untrustworthy.

## Arms

| arm | family | note |
|---|---|---|
| `cosine` | uniform θ | label-free reference |
| `iot` | diagonal θ, selected box | regret minimiser, `rel` chosen per draw |
| `cca` | closed-form linear maps | ridge by 5-fold |
| `infonce` | diagonal θ, box `rel=0.25` | symmetric InfoNCE |
| `sinkhorn` | diagonal θ, box `rel=0.25` | **Sinkhorn-InfoNCE**: logits `−C/τ`, 8 differentiable Sinkhorn iterations, loss = cross-entropy of the plan vs identity |
| `residual16` / `residual64` | `xᵀ(I + UVᵀ)y` | `V = 0` ⇒ step 0 is exactly cosine |

## Selected-checkpoint = epoch 0 (i.e. exactly cosine)

Reported as a result, not hidden as a tie. Over 27 fits per arm:

| arm | epoch-0 fits | fraction |
|---|---|---|
| `iot` | 0 | 0.00 |
| `cca` | 0 | 0.00 |
| `infonce` | 2 | 0.07 |
| `sinkhorn` | 5 | 0.19 |
| `residual16` | 6 | 0.22 |
| `residual64` | 7 | 0.26 |

The residual arms, which start at cosine, need the most "don't move" decisions — the validation fold rejects
training in a quarter of their fits.

## Gates (`tests/iot_metric_learning_test.py`, 12 tests)

1. **Planted case calibrated** (K=64, m=300, 8 signal features, σ=0.05, `x` normalised *before* the signal):
   oracle 0.983, cosine 0.197.
2. **Step 0 = cosine** for InfoNCE and residual, with the box projection the **identity on uniform θ** for
   every box (`rel` 0.25 / 0.5 / 0.9 / 0.05 / free).
3. **InfoNCE receives the similarity**, not `−similarity` (passing `−s` inverts the objective; measured hit
   0.003).
4. **Sinkhorn plan is doubly stochastic** under uniform marginals, and prefers low diagonal cost.
5. **Validation fold disjoint** from gradient rows.
6. CCA planted recovery; `at_bound_fraction` detects a binding box.

## Run

```bash
PYTHONPATH=/home/agent/Documents/dpm-inverse-ot python -u \
  -m iot_experiments.studies.real.t1t2.metric_learning.run --ks 64 128 256 --ms 50 100 300 \
  --draws 0 1 2 --solver cupdlpx
```

The run writes its own `run.log` (timestamped, per-step elapsed) into its output directory; a copy is kept
in `out/records/metric_learning/`.
