# gpu_bench batches

Run everything from the repo root. Each command prints a device line and writes to
`iot_experiments/gpu_bench/out/<batch>/`. Use `--gpu auto` (first idle A100) or a fixed
`--gpu <id>`. To pin the whole session to one card, `export CUDA_VISIBLE_DEVICES=<id>` — with
`--gpu auto` the scripts now **respect** an already-set value.

**Always pass time limits** (`--highs-time-limit`, `--cupdlpx-time-limit`). Large `|F|` can make
HiGHS iterate for minutes-to-hours, and CuPDLPx can stall; without a cap a batch runs away. Both
limits are recorded per record, and the wall is kept even when a solve fails.

## Batch 0 — `env` (seconds, run first)

```bash
python -m iot_experiments.gpu_bench.run_batch --batch env --gpu auto
```

Checks GPU/VRAM, torch+CUDA, POT, `pycupdlpx`, HELLO GPU scan, and one tiny `is_optimal` run.
**Abort if `pycupdlpx` is false** (the `cupdlpx` arms need it).

## Batch 1 — `gpu_master_k` (isolated master LP: HiGHS vs CuPDLPx)

Same problem and same edge set `F` for both backends; `K` from the family (default `lowrank`).

```bash
python -m iot_experiments.gpu_bench.run_batch --batch gpu_master_k --gpu auto \
  --family lowrank --marginal gmm --n 800 --ks 1024 2048 4096 --extra-edges 0 --repeats 2 \
  --backends highs cupdlpx --highs-time-limit 60 --cupdlpx-time-limit 60 \
  --out iot_experiments/gpu_bench/out/gpu_master_k/Ksweep
```

### The decisive follow-up: grow `|F|` (this is where HiGHS degrades)
At `|F|=1600`, HiGHS is ~1 s while CuPDLPx is ~8 s. At `|F|≈9500` HiGHS does **not** converge in
30 s (1261 iterations, time limit hit) — HiGHS, not the GPU, becomes the wall. So the real test
is `|F|` growth, with both solvers capped:

```bash
for E in 2000 4000 8000 16000; do
  python -m iot_experiments.gpu_bench.run_batch --batch gpu_master_k --gpu 3 \
    --family lowrank --marginal gmm --n 800 --ks 1024 --extra-edges $E --repeats 1 \
    --backends highs cupdlpx --highs-time-limit 60 --cupdlpx-time-limit 60 \
    --out iot_experiments/gpu_bench/out/gpu_master_k/F$E
done
```

Look for: the `|F|` where the HiGHS wall saturates at the cap while CuPDLPx stays below it — that
is the crossover, if it exists. `delta_abs_diff_vs_highs` sanity-checks they solve the same LP.

## `lp_build_probe.py` — build-only scaling (no solve)

Answers "is building the LP itself the bottleneck?" — times family sampling, the cost matrix,
POT, mandatory edges, and `build_master_system`, and reports `nnz = |F|·(K+3)`.

```bash
python iot_experiments/gpu_bench/lp_build_probe.py --family lowrank --n 800 \
  --ks 256 1024 4096 --extra-edges 0 8000
```

## Batch 2 — `e2e` (full loop; only if the master story pans out)

```bash
python -m iot_experiments.gpu_bench.run_batch --batch e2e --gpu auto \
  --family diagonal --marginal gmm --sizes 800 3200 --scans numpy hello --masters highs
```

The HELLO GPU scan only supports `DiagonalWeightFeatures`; `lowrank`/`quadratic` fall back to
NumPy. Use `--family diagonal` to see the GPU scan engage.

## Budget discipline
- One A100, one batch at a time; Batch 0 → Batch 1 (`Ksweep`) → `F*` sweep.
- Never omit the time limits. `--repeats 2` is enough; raise only if noisy.
- `nnz ≈ |F|·(K+3)`; do not raise `K` past ~4096 or `|F|` past ~`B+32000` without telling me.

## What to send back
`iot_experiments/gpu_bench/out/` (the `env/`, `gpu_master_k/**` result dirs + `run.log`). The
`run.log` now includes a per-`(K, backend)` progress line.
