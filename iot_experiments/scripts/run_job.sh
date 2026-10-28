#!/usr/bin/env bash
# CN: **作业启动器** —— 所有实验都从这里启动。它记录退出码、拒绝重跑上次失败的命令（除非 `--ack`）、
#     先检查可用内存，并以低优先级运行。
#
#     CN: 本机**没有**可用的硬内存上限（实测 `prlimit --rss` 不生效、`ulimit -v` 破坏 CUDA、
#         `systemd-run --user` 无 user bus、cgroup 子组无权限），所以这里**不假装**能强制限制；
#         它做的是"启动前检查 + 启动后记账 + 失败后不自动重试"。真正的运行期保护由驱动里的
#         `harness/watchdog.py` 线程提供（超预算或系统可用内存过低时主动中止）。
#
#     CN: 用法：
#           scripts/run_job.sh --name pixel --budget-gb 3 -- python -u -m pkg.mod --flags
#           scripts/run_job.sh --name pixel --budget-gb 3 --ack -- python -u -m pkg.mod --flags
#
#         `--ack` 表示"我知道上次失败了，仍然要跑"。没有它、且上次失败时，本脚本会拒绝启动。
#
# EN: **Job launcher** -- every experiment starts here. It records the exit code, refuses to rerun a command
#     that failed last time unless `--ack` is given, checks available memory first, and runs at low priority.
#
#     This machine has **no** usable hard memory cap (measured: `prlimit --rss` does not enforce; `ulimit -v`
#     breaks CUDA; `systemd-run --user` has no user bus; cgroup child creation is denied), so this script does
#     **not** pretend to enforce one. It provides: a pre-flight check, a post-run record, and no automatic
#     retry after a failure. Runtime protection comes from `harness/watchdog.py` inside the driver, which
#     aborts on budget overrun or low system memory.
#
#     `--ack` means "I know the last run failed and still want to run it". Without it, and with a previous
#     failure recorded, this script refuses to start.

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
STATE_DIR="${REPO_ROOT}/cache/jobs"
LOG_DIR="${REPO_ROOT}/iot_experiments/out/_jobs"
RESERVE_GB=2

BUDGET_GB=""
JOB_NAME=""
ACK=0
COMMAND=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --budget-gb) BUDGET_GB="${2:-}"; shift 2 ;;
        --name)      JOB_NAME="${2:-}";  shift 2 ;;
        --ack)       ACK=1; shift ;;
        --)          shift; COMMAND=("$@"); break ;;
        *)           echo "REFUSED: unknown argument '$1'" >&2; exit 2 ;;
    esac
done

[[ -n "${BUDGET_GB}" ]] || { echo "REFUSED: --budget-gb is required" >&2; exit 2; }
[[ -n "${JOB_NAME}"  ]] || { echo "REFUSED: --name is required" >&2; exit 2; }
[[ ${#COMMAND[@]} -gt 0 ]] || { echo "REFUSED: no command after --" >&2; exit 2; }
[[ "${BUDGET_GB}" =~ ^[0-9]+$ ]] || { echo "REFUSED: --budget-gb must be an integer" >&2; exit 2; }

mkdir -p "${STATE_DIR}" "${LOG_DIR}"
STATE_FILE="${STATE_DIR}/${JOB_NAME}.state"
cmd_key="$(printf '%s\0' "${COMMAND[@]}" | sha256sum | cut -c1-16)"

# ---- refuse to repeat a failed command --------------------------------------
# CN: 记录格式 `<cmd_key> <exit_code>`。若上次同一个命令非零退出且没有 --ack，则拒绝。
# EN: the record is `<cmd_key> <exit_code>`; refuse if the same command exited non-zero and `--ack` was
#     not given.
if [[ -f "${STATE_FILE}" && "${ACK}" -eq 0 ]]; then
    read -r prev_key prev_code < "${STATE_FILE}" || true
    if [[ "${prev_key:-}" == "${cmd_key}" && "${prev_code:-0}" != "0" ]]; then
        echo "REFUSED: this exact command last exited ${prev_code} (log: ${STATE_FILE})." >&2
        echo "         Re-run with --ack if you want to try again, or fix the cause first." >&2
        exit 7
    fi
fi

# ---- pre-flight memory -------------------------------------------------------
avail_gb=$(awk '/^MemAvailable:/ {print int($2/1024/1024)}' /proc/meminfo)
need_gb=$(( BUDGET_GB + RESERVE_GB ))
if (( avail_gb < need_gb )); then
    echo "REFUSED: available ${avail_gb} GB, need ${need_gb} GB (budget ${BUDGET_GB} + ${RESERVE_GB} reserve)." >&2
    echo "         Free memory first, or lower the budget." >&2
    exit 5
fi

stamp="$(date +%Y%m%d_%H%M%S)"
job_log="${LOG_DIR}/${JOB_NAME}_${stamp}.log"

echo "${JOB_NAME}: available ${avail_gb} GB, budget ${BUDGET_GB} GB -- starting" >&2
echo "log: ${job_log}" >&2
echo "cmd: ${COMMAND[*]}" >&2

# CN: `nice`/`ionice` 降低优先级，让交互式工作优先；不给进程设地址空间上限（会破坏 CUDA）。
# EN: `nice`/`ionice` lower the priority so interactive work wins; no address-space limit (it would break
#     CUDA).
nice -n 19 ionice -c3 "${COMMAND[@]}" 2>&1 | tee "${job_log}"
status="${PIPESTATUS[0]}"

printf '%s %s\n' "${cmd_key}" "${status}" > "${STATE_FILE}"
echo "${JOB_NAME}: exit ${status} (recorded in ${STATE_FILE})" >&2

if [[ "${status}" -ne 0 ]]; then
    echo "----------------------------------------------------------------------" >&2
    echo "FAILED with exit ${status}. Last 20 log lines:" >&2
    tail -n 20 "${job_log}" >&2
    echo "----------------------------------------------------------------------" >&2
    echo "STOP: do not relaunch without --ack and without a stated reason." >&2
fi

exit "${status}"
