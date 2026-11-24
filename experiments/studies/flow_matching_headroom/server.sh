#!/usr/bin/env bash
# EN: run the coupling-headroom study on a server GPU. Stage 0 is minutes; Stage 1 is the long job.
# CN: 在服务器 GPU 上运行耦合 headroom 研究。Stage 0 几分钟；Stage 1 是长任务。
#
#   bash experiments/studies/flow_matching_headroom/server.sh stage0
#   bash experiments/studies/flow_matching_headroom/server.sh cifar
#   bash experiments/studies/flow_matching_headroom/server.sh all
#
# EN: the script installs only `pytorch-fid` (for FID); nothing else is added to the environment.
# CN: 脚本只安装 `pytorch-fid`（用于 FID），不向环境添加其它依赖。
set -euo pipefail
cd "$(dirname "$0")/../../.."
export PYTHONPATH="$PWD"
STAGE="${1:-all}"

if ! python -c "import pytorch_fid" >/dev/null 2>&1; then
  python -m pip install -q pytorch-fid
fi

case "$STAGE" in
  stage0)
    python -m experiments.studies.flow_matching_headroom.toys --steps 10000 --tag stage0
    ;;
  cifar)
    # EN: 50k steps, batch 128, three seeds, unconditional CIFAR-10. Adjust --steps to the GPU budget.
    # CN: 50k 步、batch 128、三个种子、无条件 CIFAR-10。按 GPU 预算调整 --steps。
    python -m experiments.studies.flow_matching_headroom.cifar \
      --steps 50000 --batch 128 --seeds 0 1 2 --tag cifar
    ;;
  all)
    python -m experiments.studies.flow_matching_headroom.toys --steps 10000 --tag stage0
    python -m experiments.studies.flow_matching_headroom.cifar \
      --steps 50000 --batch 128 --seeds 0 1 2 --tag cifar
    ;;
  *)
    echo "usage: $0 [stage0|cifar|all]" >&2
    exit 2
    ;;
esac
