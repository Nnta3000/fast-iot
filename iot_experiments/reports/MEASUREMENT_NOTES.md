# Benchmark measurement hygiene — measure the algorithm, not the scaffolding

## What was wrong

`exact_benchmark_runner` built the instance (POT ground truth + `dual_optimality_certificate`)
and ran the loop **in the same process**, so the single reported "peak RSS" was dominated by the
ground-truth/checking tools, not the solver. At n=3600 that gave ~2.9 GB, almost all of it POT
network-simplex + dense `n×n` arrays.

Also, `import iot` (and `iot_experiments.benchmark`) pulled **POT** and **torch** as import
side effects, adding a fixed ~0.8 GB RSS *before* any algorithm code ran — so even a "clean"
worker process looked like it used ~860 MB.

## What changed

1. **Two-phase runner.**
   - `prepare`: builds the instance (POT + dual certificate) and pickles the minimal
     `InverseOTProblem` to disk; records the scaffolding's peak RSS.
   - `worker --problem`: loads only that problem and runs `solve_inverse_ot(validate=False)`,
     measuring its own RSS. No POT, no certificate, no evaluator.
2. **Import hygiene.** POT is now imported lazily in `src/iot/validation.py`, and the shared
   benchmark `__init__.py` uses PEP-562 lazy `__getattr__`, so importing the checkpoint schema no
   longer drags in POT/torch. `exact_benchmark_runner` imports `instances`/`runner` lazily too.
3. **Reported fields separated:** `instance_build_peak_rss_mb` (scaffolding) vs
   `rss_before_loop_mb` / `solver_peak_rss_mb` / `solver_extra_rss_mb` (algorithm).

## Measured separation (clean, K=8, `warm_purge`, seed 0)

| n | wall s | prepare (POT + cert) peak | worker before loop | **algorithm peak** | **algorithm extra** |
|---|---|---|---|---|---|
| 200 | 0.2 | 875 MB | 95 MB | 116 MB | **22 MB** |
| 1600 | 12.6 | 1274 MB | 95 MB | 249 MB | **154 MB** |
| 3200 | 59.4 | 2452 MB | 96 MB | 390 MB | **294 MB** |

- The **algorithm** is `O(K(m+n))` space: `solver_extra` grows linearly in `n` (22 → 154 → 294 MB)
  and is ~95 MB of import+problem plus the scan tile and the `O(|F|K)` LP matrix.
- The **scaffolding** is 5–10× larger and lives entirely in `prepare`; it never enters the
  algorithm's measurement.
- Extrapolating linearly, n=3600 is ≈ 0.5 GB algorithm peak (vs the ~2.9 GB previously reported).
- **Timing is unchanged and was already loop-only** (n=1600: 12.6 s; n=3200: 59.4 s with the
  purged warm solver).

Raw: `../legacy/scratch/results/warmstart_compare/warm_twophase/*.meta.json`.

---

# GPU arms (torch/CUDA)

Hardware: `NVIDIA GeForce RTX 4060 Laptop`, 8 GB, torch 2.14+cu130. New scratch kernels:
`gpu_gradient.py` (`torch_sinkhorn`, low-rank diagonal cost, GPU Adam runner).

**Validation (required before any GPU claim).** `torch_sinkhorn` vs the NumPy
`sinkhorn_plan_with_potentials`, n=200 seeds 0–2, identical tolerances: plan `max|Δ| ≤ 5.5e-16`
(float64) and identical iteration counts. So the GPU kernel is numerically the same algorithm.

## Gradient (entropic) method — CPU vs GPU (K=8, seed 0, num_outer=100)

| n | gradient CPU | gradient GPU | speedup | GPU G / R |
|---|---|---|---|---|
| 200 | 81.2 s | **6.3 s** | 13× | 0 / 0 |
| 800 | ~20 min (est.) | **21.5 s** | ~60× | 1.7e-6 / 1.2e-6 |
| 1600 | hours (est.) | **89.2 s** | — | 1.0e-6 / 4.0e-8 |

Single cold Sinkhorn solve, `ε=1e-3`, cap 200: CPU 0.45 / 7.3 / 31.0 s vs GPU 2.23 / 0.12 / 0.43 s
for n=200/800/1600 → **0.2×, 62×, 72×**. (GPU loses at n=200 to launch overhead; wins from n≈400.)

## Exact method — oracle scan micro-benchmark (dense low-rank cost + violation reduce)

| n | scan CPU | scan GPU | speedup |
|---|---|---|---|
| 1600 | 22.6 ms | 1.6 ms | 14× |
| 3200 | 68.7 ms | 7.6 ms | 9× |
| 3600 | 83.5 ms | 9.4 ms | 9× |

This is the *compute* part only (no tiling/top-γ); the real blockwise scan is ~1–1.5 s at n=3600,
so the achievable end-to-end win is bounded by Amdahl — **the LP (HiGHS) stays CPU** and is
~1.4 s of the ~3.4 s/iteration at n=3600.

## Fair, device-labelled head-to-head

| n | exact CPU (`warm_purge`) | gradient CPU | gradient GPU |
|---|---|---|---|
| 200 | **0.2 s** | 81.2 s | 6.3 s |
| 800 | ~5.7 s | ~20 min | 21.5 s |
| 1600 | **13.3 s** | hours | 89.2 s |

**Reading.** GPU turns the gradient method from "hours" into "seconds–minutes" at moderate n, but
it still does not beat the exact method at matched size (n=1600: 89 s GPU vs 13 s CPU; the exact
scan could shave it to ~8–9 s, LP-bound). The asymmetry is intrinsic: the exact method is
~40 % GPU-addressable (scan only, LP is CPU), the gradient is ~100 % GPU-addressable.

**Practical takeaway for experiments.** For algorithm comparisons, use the GPU gradient runner
(n=800 in 21 s, n=1600 in 89 s instead of hours), and keep the exact method on CPU where it is
already ≤14 s up to n=1600.

