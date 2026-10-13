# Queuing jobs without SLURM

User-space queue: sends each job to an **idle GPU**, one job per GPU, and lets you walk away.

## Quick start (from the repo root)
```bash
# sequential (one job at a time):
bash iot_experiments/gpu_bench/gpu_queue.sh iot_experiments/gpu_bench/jobs/gpu_only.txt

# parallel — one job per idle GPU (recommended for a batch):
bash iot_experiments/gpu_bench/gpu_queue.sh --parallel iot_experiments/gpu_bench/jobs/gpu_only.txt

# safer overnight: cap each job at 1h so a stuck solve can't hold a card forever
bash iot_experiments/gpu_bench/gpu_queue.sh --parallel --timeout 3600 \
  iot_experiments/gpu_bench/jobs/gpu_only.txt
```
Every job runs from the repo root with `CUDA_VISIBLE_DEVICES=<idle id>`. Outputs:
- `out/queue/<timestamp>/summary.tsv` — `index, gpu, status, wall_s, command` (read this first);
- `out/queue/<timestamp>/NN_<slug>.log` — per-job stdout/stderr;
- `out/queue/<timestamp>/queue.log` — live queue log.

## How it behaves
- **Idle test:** a GPU is idle if `memory.used < 1 GiB` and `utilization < 20%` (matching the iot
  GPU binding). Override with `FASTIOT_GPU_MAX_USED_MB` / `FASTIOT_GPU_MAX_UTIL`.
- **Atomic per-GPU lock** (`/tmp/fastiot-gpu-locks-$USER/gpu_<id>`, `mkdir`): so **several queue
  processes cooperate** — you can run one queue per tmux window and they will never share a card.
- **Parallel mode** keeps up to `--jobs N` jobs running (default `N = #GPUs`); jobs block on a
  free card when none is idle.
- **tmux-friendly:** run each queue with `&`/in its own window; logs and `summary.tsv` make it
  obvious what finished.

## Flags
| flag | meaning |
| --- | --- |
| `--parallel` | run up to `--jobs N` concurrently (one per idle GPU); default serial |
| `--jobs N` | max concurrent jobs (default = number of GPUs) |
| `--poll S` | seconds between idle checks (default 30) |
| `--timeout S` | per-job wall limit (default 0 = none) |
| `--out DIR` | override the log/summary directory |

## Writing a jobs file
One command per line; `#` comments and blank lines ignored. Keep commands **self-contained**
(they run from the repo root) and give each its own `--out`. The curated GPU-only set is
`jobs/gpu_only.txt`; a minimal line looks like:
```
python -m iot_experiments.gpu_bench.run_batch --batch e2e --gpu auto --family diagonal \
  --marginal gmm --sizes 6400 --point-dimension 8 --scans hello --masters cupdlpx \
  --plan-backend hello --cupdlpx-time-limit 10 --cupdlpx-warm-start --max-iterations 20 \
  --out iot_experiments/gpu_bench/out/e2e/myrun
```
(`--gpu auto` is fine: the queue already pinned `CUDA_VISIBLE_DEVICES`, which the runner respects.)

## Split across tmux windows
The locks make this safe — **use disjoint job files** (never the same file twice, or jobs run
twice). The curated set is split into three:
| window | file | covers |
| --- | --- | --- |
| A | `jobs/gpu_only_a_scaling.txt` | n-scaling (gmm, d=8) |
| B | `jobs/gpu_only_b_dim_marginal.txt` | d-scaling + uniform marginal |
| C | `jobs/gpu_only_c_ablation.txt` | warm-start/cap ablations + n=25600 stretch |

## Limits (honest)
No reservation: a human can still start something on a card the queue is using, and the queue
cannot preempt. It also cannot express dependencies (job B needs job A) — order them in the file
(serial mode) or run two queues. For real scheduling you'd need an admin-installed SLURM/PBS.
