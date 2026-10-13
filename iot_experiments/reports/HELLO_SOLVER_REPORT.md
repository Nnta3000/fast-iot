# HELLO-OT as the exact-OT inner solver — ready, mixed speedup with the portable backend

**Status: CURRENT, empirical (2026-10-09).** `src/iot/hello_solver.py` exposes
`make_hello_solver(problem, device=...)`, an optional `ExactPlanSolver` for the subgradient /
cutting-plane methods: it rescales the point clouds by `sqrt(theta)` and calls HELLO-OT's
squared-Euclidean OT. It is **correct** (matches POT's optimal cost to solver tolerance and drives
the subgradient method to `L0 = 0`) but its speedup is mixed with the **portable torch backend**:
~2× faster at `n=800`, ~3× *slower* at `n=1600`.

Method label: `inner=hello` (optional, non-default; POT is the default).

## 1. Why it is exact for `DiagonalWeightFeatures`

$$
c_{ij}(\theta)=\sum_k\theta_k(x_{ik}-y_{jk})^2
=\big\|\sqrt{\theta}\odot x_i-\sqrt{\theta}\odot y_j\big\|^2,
$$
so rescaled squared-Euclidean HELLO-OT solves the same OT problem. Other families raise; use
`make_pot_solver`.

## 2. Subgradient method, `inv_sqrt` lr 0.5, seed 0 (`max_iterations=2000`, tol 1e-7)

| $n$ | inner | converged | iters | wall s | s/solve | best gap |
|---|---|---|---|---|---|---|
| 200 | POT | yes | 480 | 1.63 | 0.0034 | 0 |
| 200 | HELLO | yes | 480 | 1.66 | 0.0035 | −7e−17 |
| 800 | POT | yes | 314 | 32.13 | 0.102 | 0 |
| 800 | HELLO | yes | 314 | **15.46** | 0.049 | −3e−17 |
| 1600 | POT | yes | 210 | **77.05** | 0.367 | 0 |
| 1600 | HELLO | yes | 207 | 233.15 | 1.126 | 5e−11 |

Identical iteration counts (the inner solver returns an optimal plan, so the subgradient is
unchanged); only wall time differs.

## 3. Verdict

- **Correct, and it is the right seam.** The inner solver is injectable, so `make_hello_solver`
  can replace POT for `DiagonalWeightFeatures` with no change to the learner.
- **Mixed speed with `backend="torch"`.** HELLO's portable torch backend wins at `n=800` (2.1×) but
  loses at `n=1600` (0.29×): its full solve has hierarchy/LP overhead, and the dense CPU network
  simplex (POT) is already competitive at these sizes.
- **The native backend is the missing piece.** Only `pycupdlpx` was built locally (via the CUDA-13
  shim); the native scan/solver extensions (`inner_product_scan`, `norm_cost_scan`) were not, so
  `backend="native"` is unavailable. The documented cu118 native build should move this decisively
  in HELLO's favour.
- **Not a current speed argument.** Like the CuPDLPx master, the HELLO inner solver becomes a win
  only once the native extensions are built and the sizes are large; the portable path is not
  enough.

## 4. Reproduce

```bash
python iot_experiments/benchmark/hello_solver_benchmark.py \
  --sizes 200 800 1600 --k 8 --seeds 0 --device cuda --max-iterations 2000 \
  --out iot_experiments/benchmark/out/hello_scan/hello_solver.json
```

Backing data: `iot_experiments/benchmark/out/hello_scan/hello_solver.json` (git-ignored).
Tests: `tests/iot_hello_solver_test.py`.

## 5. Caveats

- Single seed; `torch` backend on GPU, not the native one.
- HELLO-OT returns a plan to its own tolerance; the subgradient method's gap is still exact to
  ~1e-10 (best gap column), and correctness is always read from `is_optimal_plan` / the gap.
- The first HELLO call pays a one-time import/CUDA warmup; the benchmark warms up first.
