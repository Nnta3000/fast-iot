# gpu_bench — A100 batches (SLURM-less)

Runs the three-axis studies on an A100 with **GPU-vs-CPU** backends, and isolates the master
LP to test the "GPU LP wins as `K` grows" premise. Kept deliberately small: the goal is a few
decisive measurements per GPU session, not a broad sweep.

## GPU pinning without SLURM

There is no scheduler, so we pin at the process level:

- `gpu.py::configure("auto")` picks the **first idle GPU** (0 MiB used, 0% util) via
  `nvidia-smi` and sets `CUDA_VISIBLE_DEVICES=<id>` **before any torch import**. Only one
  device is exposed, so nothing can spill onto other cards.
- Use `--gpu auto` (default) or `--gpu <physical-id>` (e.g. `--gpu 2`). If you pin a busy GPU
  the script prints a warning.
- Never run two batches on the same GPU at once (batch 0's `env` output records the device).

## Setup (once, on the server)

```bash
pip install -e .                 # in the env that has the native CUDA stack + POT
python -m iot_experiments.gpu_bench.run_batch --batch env --gpu auto
```

`--batch env` prints and writes `out/env/env.json`: GPU name, torch/CUDA/POT versions,
whether `pycupdlpx` loads, whether the HELLO GPU scan supports `DiagonalWeightFeatures`, and
one tiny validated run. **Do not proceed if this fails.**

## Files

| file | role |
| --- | --- |
| `gpu.py` | idle-GPU selection + `CUDA_VISIBLE_DEVICES` + env manifest |
| `run_batch.py` | the single entry point (`--batch env|gpu_master_k|e2e`) |
| `master_microbench.py` | isolated master LP: HiGHS vs CuPDLPx on the same `F` |
| `run_batch.sh` | thin wrapper: `./run_batch.sh --batch ... --gpu auto` |
| `BATCHES.md` | the recommended batches, configs, and cost |
| `out/<batch>/` | git-ignored results (`manifest`/`records` json, csv, logs) |

## Sync back

Everything under `iot_experiments/gpu_bench/out/` is git-ignored; **sync that whole folder**
back to me, plus a `server_env.txt` if you have one. Nothing else needs syncing.

## Notes

- A GPU LP returns *approximate* duals, so a `cupdlpx` run must use `prune=False`; correctness
  is read from `is_optimal`/the loop's `max_violation`, never the solver status.
- The master microbenchmark deliberately uses `F = B` (mandatory) plus a fixed number of random
  extra edges, so it is independent of the loop trajectory and reproducible across backends.
