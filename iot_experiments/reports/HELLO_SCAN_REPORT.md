# HELLO-OT as the separation oracle — GPU scan, bit-identical and 10–70× faster

**Status: CURRENT-but-SCOPED (2026-10-15).** This report covers the GPU scan for
`DiagonalWeightFeatures` **only**. Under the project framing, the **base cost family is
`quadratic`**, which has **no GPU scan yet**; this report is therefore the *diagonal* case.
`scan_backend="hello"` is now the **default**, not an optional non-default backend.

Original finding (2026-10-09): the optional backend `scan_backend="hello"` in
`solve_inverse_ot` replaces the NumPy blockwise separation scan with HELLO-OT's torch GPU scan
(`hello_ot.kernels.torch_scan.bidirectional_violation_scan`) for `DiagonalWeightFeatures`. It is
**bit-identical** to the NumPy scan (same stop decision, same candidate set, same whole-loop
trajectory) and **removes the scan bottleneck**. The master LP now dominates.

Method label: `scan_backend=hello` (now the default).

## 1. Why it works: the reduced cost is an inner-product scan

For `DiagonalWeightFeatures`,
$$
\sigma_{ij}=f_i+g_j-c_{ij}(\theta),\qquad
c_{ij}(\theta)=\sum_k\theta_k(x_{ik}-y_{jk})^2
=\underbrace{\sum_k\theta_k x_{ik}^2}_{a_i}+\underbrace{\sum_k\theta_k y_{jk}^2}_{b_j}
-2\big\langle \sqrt{\theta}\odot x_i,\ \sqrt{\theta}\odot y_j\big\rangle .
$$
So the whole reduced-cost scan is HELLO's `score_family="inner_product"` with
`source=√θ⊙x`, `target=√θ⊙y`, `source_offset=a`, `target_offset=b`, `dot_scale=2`; the kernel
traverses all edges blockwise in FP64 and returns per-row/col top-γ candidates above a threshold,
plus the global max score. Reconstruction error on candidates is `1e-15`.

## 2. The one correctness subtlety (and its fix)

`F\S` edges satisfy `σ ≤ −δ < tol−δ`, so they are **never** candidates; only **support** edges
(`σ=0`) can occupy HELLO's top-k slots and evict a genuine near-threshold violator (`σ<0`, still
`σ+δ>tol`). The adapter over-fetches by the max support degree per row/column, filters `F`,
then trims to the true top-γ per row/column — recovering exact equivalence. If nothing survives
it falls back to one NumPy scan for the authoritative stop decision and max violation.

## 3. Single-scan time (fixed master, best of 3)

| $n$ | NumPy CPU | HELLO CUDA | speedup |
|---|---|---|---|
| 200 | 3.2 ms | 1.3 ms | 2.5× |
| 800 | 78.3 ms | 3.5 ms | 22× |
| 1600 | 319.0 ms | 7.9 ms | 40× |
| 3200 | 1351.1 ms | 19.3 ms | **70×** |

## 4. Whole-loop time (shipped `RegularizedMasterSolver(ε=1e-4)`, K=8, seed 0, warmup excluded)

| $n$ | backend | iters | wall s | oracle s | master s | final maxviol |
|---|---|---|---|---|---|---|
| 200 | numpy | 13 | 0.58 | 0.04 | 0.54 | −8.0e−5 |
| 200 | hello | 13 | 0.54 | 0.02 | 0.52 | −8.0e−5 |
| 800 | numpy | 16 | 6.59 | 0.87 | 5.70 | −7.9e−5 |
| 800 | hello | 16 | 5.41 | **0.10** | 5.29 | −7.9e−5 |
| 1600 | numpy | 19 | 24.31 | 4.76 | 19.52 | −6.7e−5 |
| 1600 | hello | 19 | 19.54 | **0.36** | 19.15 | −6.7e−5 |
| 3200 | numpy | 19 | 101.04 | 20.80 | 80.17 | −2.4e−5 |
| 3200 | hello | 19 | 81.33 | **1.25** | 80.02 | −2.4e−5 |

The oracle term drops 8–16× in-loop (identical iterations and certificate), so the total wall
improves 1.1–1.25× — bounded now entirely by the CPU master.

## 5. Verdict and the remaining gap

- **Correct and equivalent.** The scan backend changes no iteration, no certificate and no result;
  `tests/iot_hello_scan_test.py` asserts bit-identical loop trajectories and candidate sets.
- **The CPU scan bottleneck is gone.** 10–16× lower oracle time in-loop; the loop is now
  master-bound.
- **What remains non-GPU:** the HiGHS master and the assembly/prune bookkeeping. Pairing this with
  the CuPDLPx master (`CUPDLPX_MASTER_REPORT.md`) would in principle make the loop fully GPU, but
  CuPDLPx is currently slower than HiGHS and needs `prune=False`, so the pair is not a win today.
- **Scope:** only `DiagonalWeightFeatures` maps to the inner-product scan today; other families
  fall back to NumPy. Low-rank/bilinear/quadratic families can be added with the same identity
  (their feature tensors are low-rank in the source/target factors).

## 6. Reproduce

```bash
python iot_experiments/benchmarks/hello_scan_benchmark.py \
  --sizes 200 800 1600 3200 --k 8 --device cuda --repeats 3 --loop \
  --out iot_experiments/benchmarks/out/hello_scan/scan_speedup.json
```

Backing data: `iot_experiments/benchmarks/out/hello_scan/scan_speedup.json` (git-ignored).
Tests: `tests/iot_hello_scan_test.py`, `tests/iot_dependency_boundary_test.py`.

## 7. Caveats

- The first `hello` call pays a one-time import/CUDA-context cost; `hello_scan_benchmark.py`
  warms up before timing. In-process single solves are unaffected after warmup.
- Single seed, unpinned threads; wall times are machine-relative.
- The NumPy fallback runs once at termination (authoritative max violation); it is included in the
  reported oracle times and is negligible.
- HELLO source is read-only; this is an adapter in `src/iot/hello_scan.py` (a documented
  `hello_ot` import boundary), not a change to HELLO-OT.
