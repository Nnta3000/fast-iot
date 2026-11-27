# flow_matching_cost_learning — does a fixed alternative ground cost improve toy OT-CFM?

This is a small first screen, not an inverse-OT or learned-cost method. The coupling is exact OT on a fresh
256-point pool, identical to `w2_mb`. Only the ground cost changes:

- `l2`: raw squared Euclidean (the OT-CFM baseline);
- `rbf`: a fixed random-Fourier feature cost (64 features, Gaussian bandwidth from the median pairwise distance);
- `pca`: a fixed PCA-whitened cost.

Model, CFM loss, optimiser, 10k steps, batch 256 and the 8-Gaussian/two-moon targets match the v2 study. The v2
`ot_pool_B256` records are the external baseline. Few-step sliced W2 at NFE 1/2/5/10/50 is the readout.

A learned cost is only worth attempting if a fixed feature cost moves few-step W2 beyond the seed spread; a null is
a valid result. This screen does not test inverse OT, HELLO, a global plan, CIFAR, or a differentiable Sinkhorn.

```bash
python -m experiments.studies.flow_matching_cost_learning.run \
  --targets eight_gaussians two_moons --costs l2 rbf pca --seeds 0 1 2 --steps 10000
```
