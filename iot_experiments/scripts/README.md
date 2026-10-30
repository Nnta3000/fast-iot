# Scripts — what each one does and why it is here

Two launcher systems exist in this repository and they are **not** alternatives to each other.
This file says plainly what each file is for, and records that one of them is currently broken.

---

## 1. `iot_experiments/scripts/` — the local launchers

These are what I use on **this machine** (a single workstation with one RTX 4060). They are small,
they are used daily, and all three are current.

| script | what it does | why it is here |
| --- | --- | --- |
| `run_job.sh` | **The launcher every experiment goes through.** Runs a command under `nice`/`ionice`; records the exit code to `cache/jobs/<name>.state`; **refuses to rerun a command that failed last time** unless given `--ack`; checks `MemAvailable` against a declared budget plus a 2 GB reserve before starting; writes the job log to `out/_jobs/`. | It is the enforcement point for the job-hygiene rule (AGENTS.md §0.2): one launch, no silent retry, memory checked first. Without it, an OOM gets retried automatically and the machine freezes. |
| `safe_run.sh` | The stricter variant: additionally requires **exclusivity** (only one job at a time) and runs inside a **systemd scope with a hard `MemoryMax`**, so an over-budget job is killed by the cgroup rather than by the kernel OOM killer. | It documents why a hard cap is unavailable on this machine: `systemd-run --user` has no user bus here and cgroup child creation is denied, so the cgroup path cannot actually be used locally. Kept because it is the correct design for any host where a user scope *is* available. |
| `wipe_cache.sh` | Deletes rebuildable caches (preprocessed slices, pooled DINOv2 features). Supports `--dry-run`. | Disk hygiene. It lives in `iot_experiments/scripts/` and **not** in the repo-root `scripts/`, because the latter is HELLO-OT's read-only area (AGENTS.md §0). |

**Do not** add files to the repository-root `scripts/` — that directory belongs to HELLO-OT and is
read-only for this project.

---

## 2. `gpu_queue.sh` — the server launcher (moved here on 2026-10-30)

`iot_experiments/scripts/gpu_queue.sh` is the A100 launcher: one job per idle GPU, atomic per-GPU
lock files so several queue instances do not collide. It was `legacy/gpu_bench/gpu_queue.sh`; it
belongs here because it launches commands and knows nothing about torch.

```bash
scripts/gpu_queue.sh jobs.txt            # one job per idle GPU
scripts/gpu_queue.sh --parallel --jobs 4 jobs.txt
```

Its job lists now live in `benchmarks/jobs/`, and its output goes to `out/queue/<timestamp>/`
(previously `gpu_bench/out/queue/`, which contradicted the "all output in `out/`" rule).
It does **not** run the preflight itself; `run_batch` used to. Run
`python -m iot_experiments.harness.preflight --need-gpu` first on a fresh server.

## 3. `legacy/gpu_bench/` — what remains, and why you cannot run it

These are for the **A100 server**, which has several GPUs and no scheduler. This is a different
problem from the local one: here there is one GPU and a memory budget; there are many GPUs and the
question is which one a job gets.

**Status: does not currently run.** Moved to `legacy/` during the 2026-10-27 cleanup, but every
path inside it still refers to `iot_experiments.gpu_bench.*`. The import fails:

```
$ python -c "import iot_experiments.gpu_bench"
ModuleNotFoundError: No module named 'iot_experiments.gpu_bench'
```

So the A100 batches cannot be launched from this repository as it stands. They need either a move
back to `iot_experiments/gpu_bench/` or a rewrite of their internal imports.

| file | what it does | why it is here |
| --- | --- | --- |
| `gpu.py` | Picks the **first idle GPU** (0 MiB used, 0 % util) via `nvidia-smi` and sets `CUDA_VISIBLE_DEVICES` **before any torch import**; refuses a busy card unless `FASTIOT_ALLOW_BUSY_GPU=1`; writes an env manifest. | No SLURM on the server, so GPU pinning must happen at the process level. Pinning after a torch import does nothing. |
| `run_batch.py` | The single entry point: `--batch env` (self-check: GPU/driver/CuPDLPx/scan support + one tiny validated run) and `--batch e2e` (the actual three-axis runs). | `env` is the "do not proceed if this fails" gate. **Its `e2e` path imports `iot_experiments.core`, which no longer exists** (moved to `legacy/core/`), so even after the name fix the e2e batches have stale imports. |
| `run_batch.sh` | Thin wrapper: `./run_batch.sh --batch … --gpu auto`. | Convenience around the `python -m` invocation. |
| `gpu_queue.sh` | **User-space GPU queue**: one job per idle GPU, atomic per-GPU lock files so several queue instances do not collide. | This is the local replacement for SLURM. It is the most reusable file here: it does not depend on `iot_experiments.core`, only on job files of shell commands. |
| `jobs/*.txt` (15 files) | Curated job lists, one shell command per line. Windows A (n-scaling), B (dimension/marginal), C (ablation), plus pruning, planting and ladder sweeps. | They are the **record of what was run on the server**, and the provenance for `SERVER_TESTS_REPORT.md`. They name drivers that survived in `legacy/core/` (`brenier_toy`, `degenerate_natural`, `fractional_sweep`, `planting_sweep`) but by their old import paths. |
| `BATCHES.md` | The recommended batches, their configs and cost. | Tells the operator what to run first on a fresh server. |
| `QUEUE.md` | How to queue on a SLURM-less box. | Operational note for `gpu_queue.sh`. |
| `archive_stale_results.sh` | Moves old `out/` results aside. | Housekeeping for the server-side output tree. |

### Which is "strictly more important"?

Neither, because they solve different problems — the local one protects RAM on one GPU, the server
one distributes across idle GPUs. What was wrong was **filing a live entry point under `legacy/`**,
which is why it could not be imported at all. That is now fixed by the split:

| launcher | machine | protects |
| --- | --- | --- |
| `run_job.sh` | workstation, 1 GPU | RAM budget, no auto-retry after failure, **runs the preflight** |
| `gpu_queue.sh` | server, N GPUs | one job per idle GPU, per-GPU locks |

### Note on the local/server split

`run_job.sh` and `gpu_queue.sh` overlap in intent (memory safety, no accidental concurrency) but not
in mechanism: the former caps one job on one machine, the latter spreads jobs across GPUs. A future
consolidation would keep `run_job.sh`'s memory check and add `gpu_queue.sh`'s GPU selection, but that
is not proposed until `gpu_bench` runs again.
