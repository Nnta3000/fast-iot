# Global 50k x 50k OT plan on CIFAR-10 — note only

This note is a calculation, not a run. No code in this repository attempts a global CIFAR plan.

## Size

- 50 000 source images × 50 000 target images = **2.5 × 10^9** cost entries.
- Float32: **10 GB** for the cost matrix alone; float64: **20 GB**.
- A transport plan, dual potentials, and the solver workspace add at least another cost-matrix-sized object
  (10 GB in float32), and the training/image tensors sit alongside it. A single 80 GB GPU can in principle hold
  the matrix, but not comfortably with a training job; the 15 GB host and the 8 GB local card cannot.
- The cost is not arbitrary: for squared Euclidean image costs one can use the Gram identity, but a 50k × 50k
  matmul result is still 10 GB; storing it is the problem, not computing it.

## Time

- POT's exact EMD on 50k × 50k is far outside the per-batch regime measured here (128²–1024²); no timing is
  claimed. A CPU auction/network-simplex at this size would be hours to days and memory-bound. A GPU solver would
  have to avoid materialising the full matrix and use a low-rank, multiscale, or streaming representation.
- The one-time plan cost can be amortised over training (the toy `global_ot` paid 28–50 s for one 8 192² plan and
  reused it), but the CIFAR matrix is ~3 × 10^5 times larger in entries; amortisation does not remove the memory
  problem.

## What a solver would have to do

1. Never store the 50k × 50k matrix: use the feature factorisation, blockwise or low-rank costs, or an implicit
   matrix-vector product inside Sinkhorn/auction iterations.
2. Handle the marginals and the plan in blocks; keep the fixed noise bank in float16 or regenerate it from a seed.
3. Report a conditioning/iteration-limit certificate; an unconverged 50k plan can look plausible but be wrong.
4. Run the **fixed-noise control** required by the toy study: `global_ot` vs `fixed_random` on the same fixed noise
   set. In the v2 toys, `fixed_random` was indistinguishable from `independent`, while `global_ot` was the best OT
   arm at low NFE on d=2/d=8 and reversed at d=32. A CIFAR global plan would need the same control before any gain
   is attributed to the coupling rather than the fixed finite noise set.

## Consequence for this study

The existing Stage 1 `w2_mb` pool is 128. Phase D only tests 512/1024; it does not test a dataset-level plan. The
global-plan question is a solver/engineering question, not something the current POT path can answer.
