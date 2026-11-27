# Fixed ground-cost screen — results

**Question.** Without learning anything, does replacing the raw squared-Euclidean cost in `w2_mb` with a fixed
feature cost improve few-step OT-CFM on the toy targets?

**Protocol.** Exact OT on a fresh 256-point noise/data pool, exactly as `w2_mb`. Only the ground cost changes:
`l2` (raw), `rbf` (64 fixed random-Fourier features, Gaussian bandwidth from the median pairwise distance), and
`pca` (fixed PCA-whitened coordinates). Same MLP, CFM loss, Adam 1e-3, batch 256, 10 000 steps. Targets:
eight-Gaussians and two-moons; seeds 0/1/2. Sliced W2 (512 projections, 4096 generated vs fresh 4096) at Euler
NFE 1/2/5/10/50. The v2 `ot_pool_B256` (10k) is the external baseline; `global_ot` is the strong dataset-level
reference. This is a fixed-cost screen, **not** a learned cost and **not** inverse OT.

Records: `experiments/out/flow_matching_cost_learning/fixed_costs/records.json`; log: `run.log`.

## Results (10k, Euler, mean ± sd over 3 seeds)

### eight-Gaussians

| arm | NFE 1 | NFE 2 | NFE 5 | NFE 10 |
| --- | ---: | ---: | ---: | ---: |
| independent (v2) | 1.2920 ± 0.0195 | 0.4499 ± 0.0465 | 0.1540 ± 0.0195 | 0.1303 ± 0.0237 |
| `ot_pool_B256` (v2) | 0.1557 ± 0.0157 | 0.1262 ± 0.0192 | 0.1134 ± 0.0216 | 0.1113 ± 0.0232 |
| `global_ot` (v2) | **0.1209 ± 0.0125** | **0.1054 ± 0.0161** | **0.0996 ± 0.0194** | **0.0978 ± 0.0209** |
| `cost_l2` | 0.1712 ± 0.0255 | 0.1322 ± 0.0310 | 0.1155 ± 0.0364 | 0.1137 ± 0.0393 |
| `cost_rbf` | 0.1659 ± 0.0181 | 0.1302 ± 0.0294 | 0.1143 ± 0.0349 | 0.1132 ± 0.0375 |
| `cost_pca` | 0.1698 ± 0.0251 | 0.1318 ± 0.0307 | 0.1154 ± 0.0363 | 0.1133 ± 0.0393 |

### two-moons

| arm | NFE 1 | NFE 2 | NFE 5 | NFE 10 |
| --- | ---: | ---: | ---: | ---: |
| independent (v2) | 0.5304 ± 0.0256 | 0.3545 ± 0.0235 | 0.1296 ± 0.0163 | 0.0742 ± 0.0131 |
| `ot_pool_B256` (v2) | 0.0802 ± 0.0111 | 0.0451 ± 0.0096 | 0.0392 ± 0.0084 | 0.0380 ± 0.0075 |
| `global_ot` (v2) | **0.0440 ± 0.0065** | **0.0285 ± 0.0039** | **0.0251 ± 0.0024** | **0.0248 ± 0.0022** |
| `cost_l2` | 0.0772 ± 0.0033 | 0.0453 ± 0.0020 | 0.0393 ± 0.0036 | 0.0382 ± 0.0042 |
| `cost_rbf` | 0.0958 ± 0.0088 | 0.1226 ± 0.0057 | 0.0649 ± 0.0147 | 0.0502 ± 0.0139 |
| `cost_pca` | 0.0891 ± 0.0139 | 0.0442 ± 0.0041 | 0.0379 ± 0.0042 | 0.0369 ± 0.0044 |

## What the screen found

1. **No fixed feature cost beats the raw L2 control consistently.** On eight-Gaussians, RBF is 0.005 lower than
   `cost_l2` at NFE 1 and 0.002 lower at NFE 2; with 3 seeds and sd 0.02–0.03, this is not a signal. PCA is
   indistinguishable from L2.
2. **RBF is clearly worse on two-moons** (NFE 2: 0.1226 vs 0.0453), not better. PCA is marginally better than L2
   at NFE 5/10 (0.0379 vs 0.0393, 0.0369 vs 0.0382) but within the seed sd and not at NFE 1–2.
3. **`global_ot` remains the best arm at every NFE on both targets.** No fixed-cost arm approaches it.
4. The local `cost_l2` control is comparable to v2 `ot_pool_B256`; on eight-Gaussians it is slightly worse at
   NFE 1 (0.1712 vs 0.1557), which is a model-init/on-the-fly-pairing difference, not a cost effect.

## Decision

This fixed-cost gate shows **no few-step signal**. Following the pre-registered logic, a learned cost on these
features is not justified; I did not run anchor matching or a differentiable-Sinkhorn/learned-cost search. A learned
cost would have to beat both the L2 control and `global_ot`, and the fixed-feature screen gives no reason to expect
that on these toys.

## Caveats

- Two targets only; 3 seeds; 10k steps. Not CIFAR.
- Fixed costs, not learned costs; no inverse OT, no HELLO, no differentiable Sinkhorn, no global 50k plan.
- The RBF bandwidth is a single fixed choice from the median pairwise distance; a different kernel could behave
  differently, but this is the standard first choice and was not tuned.
- The `cost_l2` arm is not identical to v2 `ot_pool_B256` (different model seeding and on-the-fly pair construction);
  treat it as the internal control, and v2 as the external reference.
- Any null here is a null for this screen; it is not evidence that no cost can ever help.
