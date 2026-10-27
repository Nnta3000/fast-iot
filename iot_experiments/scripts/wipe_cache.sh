#!/usr/bin/env bash
# CN: 清空实验缓存（预处理切片、池化的 DINOv2 特征）。
#
#     CN: 缓存**可删除、可重建**，清理是安全的：下一次运行会重新生成，代价是几分钟。
#         默认只删缓存内容、保留目录与 README；`--dry-run` 只报告将要删除的内容。
#
#     CN: 放在 `iot_experiments/scripts/` 而不是仓库根的 `scripts/`——后者是 HELLO-OT 的只读区域
#         （`AGENTS.md` §0），本项目不得在该处新增文件。
#
# EN: Wipe the experiment caches (preprocessed slices, pooled DINOv2 features).
#
#     The caches are **rebuildable**, so wiping is safe: the next run regenerates them at a cost of a few
#     minutes. By default only the cache contents are removed, the directories and README kept;
#     `--dry-run` reports only.
#
#     It lives in `iot_experiments/scripts/` rather than the repository's `scripts/`, which is HELLO-OT's
#     read-only area (AGENTS.md §0) and may not gain this project's files.
#
# usage / 用法
#     iot_experiments/scripts/wipe_cache.sh
#     iot_experiments/scripts/wipe_cache.sh --dry-run

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CACHE_ROOT="${REPO_ROOT}/cache"
DRY_RUN=0
[[ "${1:-}" == "--dry-run" ]] && DRY_RUN=1

if [[ ! -d "${CACHE_ROOT}" ]]; then
    echo "no cache directory at ${CACHE_ROOT}; nothing to do"
    exit 0
fi

echo "cache root: ${CACHE_ROOT}"
echo "contents:"
# CN: 列出缓存内容与体积；README.md 与脚本本身不受影响。
# EN: list cache contents with sizes; README.md and this script are unaffected.
du -sh "${CACHE_ROOT}"/* 2>/dev/null | grep -v "README.md" || echo "  (empty)"

if [[ "${DRY_RUN}" -eq 1 ]]; then
    echo "--dry-run: nothing deleted"
    exit 0
fi

# CN: 删除缓存目录下的所有内容，但保留 README.md 与目录结构。
# EN: delete everything under the cache root, keeping README.md and the directory layout.
find "${CACHE_ROOT}" -mindepth 1 -maxdepth 1 ! -name "README.md" -exec rm -rf {} +

echo "wiped. remaining:"
ls -A "${CACHE_ROOT}"
