#!/usr/bin/env bash
# CN: 用户态 GPU 队列（无 SLURM）：每个空闲 GPU 跑一个作业，原子锁让多个队列实例互不抢占。
# EN: User-space GPU queue (no SLURM): one job per idle GPU, atomic per-GPU locks so several
#     queue instances cooperate.
#
# CN: 用法 / EN: usage:
#   gpu_queue.sh [--parallel] [--jobs N] [--poll S] [--timeout S] [--out DIR] jobs.txt
#   jobs.txt: one shell command per line ('#' comments and blanks ignored).
#   --parallel       keep up to N jobs running (default N = number of GPUs); default is serial.
#   --poll S         seconds between idle-GPU checks (default 30).
#   --timeout S      per-job wall limit in seconds, 0 = none (default 0).
#   --out DIR        logs + summary (default iot_experiments/out/queue/<ts>).
#
# CN: 每个作业在 `CUDA_VISIBLE_DEVICES=<idle id>` 下从仓库根目录运行；日志逐作业保存，
#     并写 summary.tsv。空闲判定与 iot 的 GPU 绑定一致（内存 <1GiB 且利用率 <20%）。
# EN: each job runs from the repo root with `CUDA_VISIBLE_DEVICES=<idle id>`; per-job logs plus a
#     summary.tsv. Idle test matches the iot GPU pinning (mem <1GiB and util <20%).
set -uo pipefail

POLL=30
JOBS_FILE=""
PARALLEL=0
MAX_JOBS=0
TIMEOUT=0
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LOCK_DIR="${FASTIOT_GPU_LOCK_DIR:-${TMPDIR:-/tmp}/fastiot-gpu-locks-${USER:-user}}"
MAX_USED_MB="${FASTIOT_GPU_MAX_USED_MB:-1024}"
MAX_UTIL="${FASTIOT_GPU_MAX_UTIL:-20}"
OUT_DIR=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --parallel) PARALLEL=1; shift ;;
    --jobs) MAX_JOBS="$2"; shift 2 ;;
    --poll) POLL="$2"; shift 2 ;;
    --timeout) TIMEOUT="$2"; shift 2 ;;
    --out) OUT_DIR="$2"; shift 2 ;;
    -h|--help) sed -n '2,17p' "$0"; exit 0 ;;
    *) JOBS_FILE="$1"; shift ;;
  esac
done
if [[ -z "$JOBS_FILE" ]]; then
  echo "usage: gpu_queue.sh [--parallel] [--jobs N] [--poll S] [--timeout S] [--out DIR] jobs.txt" >&2
  exit 2
fi
[[ -n "$OUT_DIR" ]] || OUT_DIR="$REPO_ROOT/iot_experiments/out/queue/$(date +%F_%H%M%S)"
mkdir -p "$LOCK_DIR" "$OUT_DIR"

gpu_free_indices() {
  nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits 2>/dev/null \
    | awk -F', *' -v mu="$MAX_USED_MB" -v mu2="$MAX_UTIL" '$2 < mu && $3 < mu2 {print $1}'
}

acquire_gpu() {
  local i
  while :; do
    for i in $(gpu_free_indices); do
      if mkdir "$LOCK_DIR/gpu_$i" 2>/dev/null; then echo "$i"; return 0; fi
    done
    echo "[queue] no idle GPU; waiting ${POLL}s" >&2
    sleep "$POLL"
  done
}
release_gpu() { rmdir "$LOCK_DIR/gpu_$1" 2>/dev/null || true; }
slug() { echo "$1" | tr -cs 'A-Za-z0-9' '_' | cut -c1-70; }

run_one() {
  local idx="$1" cmd="$2" gpu log started wall status runner
  gpu="$(acquire_gpu)"
  log="$OUT_DIR/$(printf '%02d' "$idx")_$(slug "$cmd").log"
  started=$(date +%s)
  echo "[queue] job $idx -> GPU $gpu  ($log)" | tee -a "$OUT_DIR/queue.log"
  if [[ "$TIMEOUT" -gt 0 ]]; then
    ( cd "$REPO_ROOT" && CUDA_VISIBLE_DEVICES="$gpu" timeout "$TIMEOUT" bash -c "$cmd" ) >"$log" 2>&1
  else
    ( cd "$REPO_ROOT" && CUDA_VISIBLE_DEVICES="$gpu" bash -c "$cmd" ) >"$log" 2>&1
  fi
  status=$?
  wall=$(( $(date +%s) - started ))
  if [[ "$status" -eq 0 ]]; then status="ok"; else status="fail($status)"; fi
  printf '%s\t%s\t%s\t%s\t%s\n' "$idx" "$gpu" "$status" "$wall" "$cmd" >> "$OUT_DIR/summary.tsv"
  echo "[queue] job $idx $status  gpu=$gpu  ${wall}s" | tee -a "$OUT_DIR/queue.log"
  release_gpu "$gpu"
}

mapfile -t JOBS < <(grep -vE '^[[:space:]]*(#|$)' "$JOBS_FILE")
echo "[queue] ${#JOBS[@]} jobs from $JOBS_FILE; parallel=$PARALLEL; logdir=$OUT_DIR" | tee -a "$OUT_DIR/queue.log"

if [[ "$PARALLEL" == "1" ]]; then
  [[ "$MAX_JOBS" -gt 0 ]] || MAX_JOBS=$(nvidia-smi -L 2>/dev/null | wc -l)
  running=0
  idx=0
  for cmd in "${JOBS[@]}"; do
    while [[ "$running" -ge "$MAX_JOBS" ]]; do wait -n; running=$((running - 1)); done
    run_one "$idx" "$cmd" &
    running=$((running + 1))
    idx=$((idx + 1))
  done
  wait
else
  idx=0
  for cmd in "${JOBS[@]}"; do run_one "$idx" "$cmd"; idx=$((idx + 1)); done
fi
echo "[queue] done; summary: $OUT_DIR/summary.tsv" | tee -a "$OUT_DIR/queue.log"
