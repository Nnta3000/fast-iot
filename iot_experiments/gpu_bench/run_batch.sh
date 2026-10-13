#!/usr/bin/env bash
# CN: 便捷包装；真正的 GPU 绑定在 run_batch.py 内完成（在任何 torch import 之前）。
# EN: Thin wrapper; the actual GPU pin happens inside run_batch.py (before any torch import).
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo="$(cd "$here/../.." && pwd)"
cd "$repo"
exec python -m iot_experiments.gpu_bench.run_batch "$@"
