#!/usr/bin/env bash
# CN: **唯一的作业启动器**。任何实验都必须通过它启动，绝不直接调用 python。
#
#     它做四件事，任何一件不满足就**拒绝启动**（退出码非零，不产生副作用）：
#       1. **可用内存检查**：`MemAvailable` 必须 >= budget + 2 GB，否则拒绝。
#          （`MemAvailable` 而不是 `MemFree`：前者已把可回收的页缓存算进去，是真正能用的量。）
#       2. **互斥检查**：如果已经有本脚本启动的作业在跑，拒绝（同一时刻只允许一个）。
#       3. 在 **systemd scope** 里启动，带硬 `MemoryMax` 与 `MemorySwapMax=0`：超出预算会被 cgroup
#          直接杀掉，**不会**拖垮整机。
#       4. 低优先级：`nice -n 19 ionice -c3`，让用户的交互式工作优先。
#
#     CN: **不用 `ulimit -v`**：它限制的是地址空间，CUDA 驱动会预留巨大的地址空间，导致即使
#         很小的分配也失败（此前实测 `ulimit -v 8G` 连 6.4 MB 都不给分配）。
#
# EN: **The only job launcher.** Every experiment must go through it; never call python directly.
#
#     It refuses to start (non-zero exit, no side effects) unless:
#       1. **memory**: `MemAvailable` >= budget + 2 GB (`MemAvailable`, not `MemFree`, because it already
#          counts reclaimable page cache -- it is the amount actually usable);
#       2. **exclusivity**: no other job started by this script is running;
#       3. it runs inside a **systemd scope** with a hard `MemoryMax` and `MemorySwapMax=0`, so exceeding the
#          budget kills the job via cgroup instead of overwhelming the machine;
#       4. it runs at low priority (`nice -n 19 ionice -c3`) so interactive work wins.
#
#     **No `ulimit -v`**: that limits address space, and the CUDA driver reserves a huge address range, so
#     even a 6.4 MB allocation fails (measured earlier with `ulimit -v 8G`).
#
# usage / 用法
#     scripts/safe_run.sh --budget-gb 3 --name pixel-baseline -- python -u -m pkg.module --flags
#
#   --budget-gb N   hard RAM budget in GB (cgroup MemoryMax), also used for the free-memory check
#   --name STR      short job name, used for logging and the exclusivity check
#   --              everything after is the command to run
#
#   CN: `--gpu` 目前**不支持**：本机预算不含 GPU，且需要 GPU 的作业应当由人明确批准。
#   EN: `--gpu` is **not supported**: the budget excludes the GPU, and GPU jobs need explicit approval.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LOCK_DIR="${REPO_ROOT}/cache/locks"
LOG_DIR="${REPO_ROOT}/iot_experiments/out/_jobs"
RESERVE_GB=2

BUDGET_GB=""
JOB_NAME=""
COMMAND=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --budget-gb) BUDGET_GB="${2:-}"; shift 2 ;;
        --name)      JOB_NAME="${2:-}";  shift 2 ;;
        --gpu)       echo "REFUSED: --gpu is not supported (GPU work needs explicit human approval)" >&2; exit 3 ;;
        --)          shift; COMMAND=("$@"); break ;;
        *)           echo "REFUSED: unknown argument '$1'" >&2; exit 2 ;;
    esac
done

[[ -n "${BUDGET_GB}" ]] || { echo "REFUSED: --budget-gb is required" >&2; exit 2; }
[[ -n "${JOB_NAME}"  ]] || { echo "REFUSED: --name is required" >&2; exit 2; }
[[ ${#COMMAND[@]} -gt 0 ]] || { echo "REFUSED: no command after --" >&2; exit 2; }
[[ "${BUDGET_GB}" =~ ^[0-9]+$ ]] || { echo "REFUSED: --budget-gb must be an integer number of GB" >&2; exit 2; }

mkdir -p "${LOCK_DIR}" "${LOG_DIR}"

# ---- 1) exclusivity ---------------------------------------------------------
# CN: 用一个原子锁文件表示"本启动器有一个作业在跑"。若已存在且 pid 仍存活，则拒绝。
# EN: an atomic lock file means "this launcher already has a job running"; refuse if its pid is alive.
LOCK_FILE="${LOCK_DIR}/${JOB_NAME}.lock"
for existing in "${LOCK_DIR}"/*.lock; do
    [[ -e "${existing}" ]] || continue
    other_pid="$(cat "${existing}" 2>/dev/null || echo "")"
    if [[ -n "${other_pid}" ]] && kill -0 "${other_pid}" 2>/dev/null; then
        echo "REFUSED: another job is already running (${existing##*/}, pid ${other_pid})." >&2
        echo "         Only one job at a time. Wait for it or remove the lock if it is stale." >&2
        exit 4
    fi
    rm -f "${existing}"        # CN: 陈旧锁，清理。/ EN: stale lock, clean it.
done

# ---- 2) memory --------------------------------------------------------------
avail_kb=$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)
avail_gb=$(( avail_kb / 1024 / 1024 ))
need_gb=$(( BUDGET_GB + RESERVE_GB ))
if (( avail_gb < need_gb )); then
    echo "REFUSED: not enough memory." >&2
    echo "         available ${avail_gb} GB, need ${need_gb} GB (budget ${BUDGET_GB} GB + ${RESERVE_GB} GB reserve)." >&2
    echo "         Close something (ComfyUI, browser) or lower the budget." >&2
    exit 5
fi

# ---- 3) systemd scope available? --------------------------------------------
# CN: 本机此前没有可用的 user bus；若 systemd-run 不可用则拒绝，而不是退回到无上限运行。
# EN: this machine had no usable user bus before; if systemd-run is unavailable, refuse rather than fall back
#     to running without a limit.
if ! systemd-run --user --scope -p MemoryMax=1M --quiet /bin/true 2>/dev/null; then
    echo "REFUSED: 'systemd-run --user --scope' is not usable on this machine (no user bus)." >&2
    echo "         A hard MemoryMax cannot be enforced, so this launcher will not start the job." >&2
    exit 6
fi

# ---- 4) launch --------------------------------------------------------------
echo "${JOB_NAME}: available ${avail_gb} GB, budget ${BUDGET_GB} GB, reserve ${RESERVE_GB} GB -- starting" >&2
stamp="$(date +%Y%m%d_%H%M%S)"
job_log="${LOG_DIR}/${JOB_NAME}_${stamp}.log"
echo "log: ${job_log}" >&2

write_lock() { echo "$1" > "${LOCK_FILE}"; }
cleanup() { rm -f "${LOCK_FILE}"; }
trap cleanup EXIT

# CN: 用 bash -c 包一层，以便在 scope 内写入锁并清理；命令与参数通过位置参数传递，避免注入。
# EN: wrap in bash -c so the lock is written and cleaned inside the scope; the command and its arguments are
#     passed positionally, avoiding injection.
write_lock "$$"
systemd-run --user --scope --quiet \
    -p "MemoryMax=${BUDGET_GB}G" \
    -p "MemorySwapMax=0" \
    -p "CPUQuota=400%" \
    nice -n 19 ionice -c3 \
    "${COMMAND[@]}" 2>&1 | tee "${job_log}"
status="${PIPESTATUS[0]}"

echo "${JOB_NAME}: finished with status ${status}" >&2
exit "${status}"
