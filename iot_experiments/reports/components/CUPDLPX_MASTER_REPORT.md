# CuPDLPx as an inverse-OT master backend — built and measured

> **Status: PARTIAL.** The **adapter** described here is current (it is the master on the target
> path). The **measurements** were taken against **HiGHS** and with the old `tie_break="none"`
> framing; treat the numbers as historical. For current GPU-path results see
> `current/SERVER_TESTS_REPORT.md`.

**Status: CURRENT, empirical (2026-10-09, local CUDA-13.4 build).** The adapter, the native
extension, and the in-loop comparison all execute. CuPDLPx returns valid optima
(`is_optimal_plan=True`, `delta` matching HiGHS to ~1e-4) but is **slower than HiGHS at the
current sizes**, with the relative gap shrinking as `K` grows (36× at `K=4` → 1.3× at `K=256`).

Method label: `master=cupdlpx` (optional, non-default; HiGHS stays the default).

## 1. Build (and the CUDA-13 compatibility shim)

- Host: RTX 4060 Laptop, **CUDA 13.4 / nvcc 13.4**, **torch 2.14.0+cu130**, Python 3.12.13.
- The documented native build needs **torch 2.7.1 + cu118 + nvcc 11.8**; this machine has **no
  conda/micromamba**, so that path is unavailable here.
- Source build under CUDA 13.4 failed only on `solver.cu`: CUDA 13's CCCL no longer pulls
  `thrust/tuple.h` transitively, so `thrust::get` / `thrust::make_tuple` are undeclared. HELLO-OT
  source is read-only, so the fix is an `nvcc` flag:
  `-include <CUDA_HOME>/targets/x86_64-linux/include/cccl/thrust/tuple.h`.
- Reproducible helper (top-level `pycupdlpx`, no HELLO-OT changes):
  ```bash
  python iot_experiments/legacy/methods/exact/build_pycupdlpx_cuda13.py
  # --dry-run prints the generated setup.py
  ```
  In the official cu118 environment, `HELLO_OT_BUILD_NATIVE=1 pip install -e .` builds the same
  extension packaged as `hello_ot._native.pycupdlpx` and needs no flag.

## 2. Adapter fixes found by running it

1. **Zero-copy lifetime.** The binding exposes `out["x"]` as a numpy view over `x_cache_` with
   `base=None`; `np.asarray` did not copy and returned non-deterministic garbage (`inf`, `5e47`,
   ~0). Fixed with `np.array(primal, copy=True)` plus a `np.isfinite` guard.
2. **Approximate feasibility.** PDHG's `theta` is only ~1e-4 feasible, so `is_optimal_plan`'s
   `contains(theta, 1e-6)` failed. Fixed by projecting the simplex block (`project_theta`); `delta`
   keeps the solver's value.
3. `_load_pycupdlpx` must `import torch` first (the extension links `libc10.so`).
4. Two pre-existing pyright errors in `build_cupdlpx_lp` were fixed.

## 3. Pruning is not the fix (the deeper reason)

The shipped loop's `tol = 1e-9` certificate and `tau_prune = 1e-12` pruning both require master
slacks far more accurate than PDHG's ~1e-4. Two runs at `n=200` show this is not repairable by a
matched pruning threshold:

| K | config | converged | iters | max \|F\| | maxviol | `is_optimal` | wall s |
|---|---|---|---|---|---|---|---|
| 4 | highs-shipped (`tau=1e-12`) | yes | 14 | 1322 | −3.5e−5 | true | 0.58 |
| 4 | highs-matched (`tau=1e-3`) | yes | 14 | 1323 | −2.1e−4 | true | 0.51 |
| 4 | cupdlpx-matched (`tau=1e-3`) | **no** | 50 | 1191 | 2.8e−1 | **false** | 79.5 |
| 4 | cupdlpx-prune-off | yes | 9 | **2211** | −1.5e−4 | true | 20.8 |
| 8 | highs-shipped | yes | 13 | 1322 | −8.0e−5 | true | 0.56 |
| 8 | cupdlpx-matched | **no** | 50 | 1191 | 7.7e−1 | **false** | 13.2 |
| 8 | cupdlpx-prune-off | yes | 10 | **2104** | −4.8e−3 | true | 4.86 |

- Matched-τ pruning does **not** make CuPDLPx converge; the loop runs to budget with
  `maxviol ≈ 0.3–0.9` and can return a non-optimal `theta`.
- Pruning does control the active set (HiGHS `|F| ≈ 1322`), but `prune=False` lets `|F|` grow to
  `≈ 2100–2900`; that is the budget blow-up to watch.
- The cause is that PDHG's restricted-LP duals are too inaccurate to point at the binding edges
  and to meet the `1e-9` termination test. A genuine fix would change the loop's tolerance/
  stagnation rule or its cut selection, not `tau_prune`.

## 4. Measurements

### 4.1 Diagonal, `n=200`, CuPDLPx `prune=False` vs HiGHS shipped: the gap closes with K

| K | HiGHS iters / wall s | CuPDLPx iters / wall s | wall ratio | CuPDLPx max \|F\| |
|---|---|---|---|---|
| 4 | 14 / 0.58 | 9 / 20.8 | 36× | 2211 |
| 8 | 13 / 0.56 | 10 / 4.86 | 8.6× | 2104 |
| 16 | 11 / 0.84 | 8 / 5.67 | 6.7× | 2004 |
| 32 | 11 / 1.25 | 10 / 7.28 | 5.8× | 2218 |
| 64 | 14 / 2.81 | 10 / 7.55 | 2.7× | 2393 |
| 128 | 14 / 6.72 | 10 / 13.8 | 2.1× | 2590 |
| 256 | 15 / 19.9 | 11 / 26.5 | **1.3×** | 2885 |

HiGHS's master cost grows with `K` (0.58 → 19.9 s), while PDHG's is dominated by its iteration
count and stays ~5–27 s. So CuPDLPx nearly catches HiGHS by `K ≈ 256`. Two caveats: `K=256` is not
a realistic inverse-OT feature count (the families below have `K ≤ 18`), and CuPDLPx needs
`prune=False` (≈2× larger `|F|`).

### 4.2 Diagonal, `n=800` (CuPDLPx `prune=False` vs HiGHS shipped)

| K | HiGHS wall s | CuPDLPx wall s | ratio |
|---|---|---|---|
| 4 | 5.91 | 98.5 | 16.7× |
| 8 | 6.75 | 19.6 | 2.9× |

(`n=800, K=4` is the outlier; PDHG is sensitive to LP degeneracy. Single seed.)

### 4.3 Harder LP families (`n=60/120`, 1 seed, CuPDLPx prune-off vs exact HiGHS)

| family | K | n | exact wall s | CuPDLPx wall s | ratio | `L0`/`R`/`is_optimal` |
|---|---|---|---|---|---|---|
| quadratic | 6 | 60 | 0.12 | 4.77 | 40× | 0 / 0 / true |
| quadratic | 6 | 120 | 0.27 | 8.16 | 30× | 0 / 0 / true |
| quadratic | 10 | 60 | 0.10 | 3.00 | 30× | 0 / 0 / true |
| quadratic | 10 | 120 | 0.30 | 9.16 | 31× | 0 / 0 / true |
| lowrank | 8 | 60 | 0.10 | 2.96 | 29× | 0 / 0 / true |
| lowrank | 8 | 120 | 0.28 | 2.01 | 7.2× | 0 / 0 / true |
| lowrank | 16 | 60 | 0.13 | 2.48 | 19× | 0 / 0 / true |
| lowrank | 16 | 120 | 0.34 | 2.47 | 7.2× | 0 / 0 / true |

Every CuPDLPx run converges and certifies, but it is 7–40× slower than HiGHS at these sizes. Cross
terms (quadratic) do not change the conclusion. The signed families are excluded (they are
trivially admissible), and PSD/off-support noise have no LP master at all.

## 5. Verdict

- **Correct but not a win at the current sizes.** `delta` agrees with HiGHS to ~1e-4 and every arm
  certifies via `is_optimal_plan`; CuPDLPx often needs *fewer* outer iterations. But the master LP
  is tiny and HiGHS is milliseconds, so a GPU first-order solver does not attack the real
  bottleneck (the `O(mnK)` oracle scan / `n^{2.1}` LP at large `n`).
- **The relative gap is a K story, not an n story.** It falls from ~36× (`K=4`) to ~1.3× (`K=256`)
  because HiGHS's master grows with `K` while PDHG's does not. If a future cost family has very
  large `K` (hundreds) and the rest of the loop is near-linear, CuPDLPx could become competitive —
  but the loop would need an approximate-dual-aware tolerance/cut rule and a safe active-set bound.
- **Active-set risk is real.** `prune=False` grows `|F|` to ~2× HiGHS and unboundedly; the shipped
  pruning does not work with first-order slacks. This alone blocks a drop-in replacement today.

## 6. Reproduce

```bash
python iot_experiments/legacy/methods/exact/build_pycupdlpx_cuda13.py
python -m pytest tests/iot_cupdlpx_master_test.py -q

# fair pruning comparison (n=200)
python iot_experiments/benchmarks/cupdlpx_master_benchmark.py --sizes 200 --k 4 --seeds 0 \
  --max-iterations 50 --out .../cupdlpx_fair_n200_k4.json

# K sweep
python iot_experiments/benchmarks/cupdlpx_master_benchmark.py --sizes 200 --k 256 --seeds 0 \
  --max-iterations 60 --cupdlpx-modes prune-off --out .../cupdlpx_ksweep_n200_k256.json

# harder families (quadratic / lowrank) with the CuPDLPx arm
python iot_experiments/benchmarks/harder_benchmark.py --arm families --families quadratic \
  --sizes 60 120 --seeds 0 --methods exact cupdlpx --max-iterations 300 --point-dimension 3 \
  --out .../cupdlpx_harder_quadratic_d3_k8.json
```

Backing data: `iot_experiments/legacy/methods/exact/out/cutting_plane/cupdlpx_*.json` (git-ignored).

## 7. Caveats

- This is a **CUDA 13.4 / torch 2.14** local build, not the documented cu118 stack; the
  `-include thrust/tuple.h` flag is the compatibility shim.
- Single seed, unpinned threads; the `n=800, K=4` number is slow enough to warrant a re-run.
- `tie_break="none"` only: CuPDLPx has no lexicographic re-solve, so it is outside the termination
  proof (same as the LP `none` mode). Correctness is always read from `is_optimal_plan`, never from
  the solver status.
- The K sweep uses the diagonal family to isolate the master's K-scaling; real families have
  `K ≤ 18` here.
