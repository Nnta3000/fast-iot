#!/usr/bin/env bash
# CN: 在服务器上归档「失效/离岗」结果：GMM 结果（旧公式错误）与 CPU 参照（numpy 扫描 / HiGHS）。
#     不删除任何文件；只是移动到 out/_legacy_results/ 下，便于核对。
# EN: Archive off-target results ON THE SERVER: GMM results (old, wrong formulation) and CPU
#     references (numpy scan / HiGHS). Nothing is deleted; files are moved under out/_legacy_results/.
#
# 用法 / EN: usage (from the repo root on the server):
#     bash iot_experiments/gpu_bench/archive_stale_results.sh
#
# 分类原则 / EN: classification rules
#   1) STALE-GMM  : any run whose marginal is gmm / gmm_floored  -> the gmm construction was WRONG
#                   (mixture applied along generation order, not geometry). Delete-tier.
#   2) CPU-REF    : runs produced with the numpy scan or a HiGHS (regularized) master, unless the
#                   run was explicitly a GPU-vs-CPU comparison. Here that is `gpu_master_k/`.
#   3) KEEP       : everything else (GPU scan hello + cupdlpx master, non-gmm marginals).

set -euo pipefail
OUT="iot_experiments/gpu_bench/out"
DEST="$OUT/_legacy_results"
mkdir -p "$DEST/stale_gmm" "$DEST/cpu_reference"

echo "[archive] repo root: $(pwd)"
echo "[archive] target    : $DEST"

move() {
  local src="$1" bucket="$2"
  if [ -e "$src" ]; then
    mv "$src" "$DEST/$bucket/$(echo "$src" | sed 's|^iot_experiments/gpu_bench/out/||; s|/|__|g')"
    echo "  moved  $src  ->  $bucket/"
  fi
}

# --- 1) STALE-GMM: old (wrong) gmm / gmm_floored instances -------------------
move "$OUT/e2e/E3_n12800_gmm_d8"                stale_gmm
move "$OUT/e2e/E4_n6400_gmm_d32"                stale_gmm
move "$OUT/e2e_gpu"                             stale_gmm
move "$OUT/no_prune/N1_n3200_gmm_d8"            stale_gmm
move "$OUT/no_prune/N2_n6400_gmm_d8"            stale_gmm
move "$OUT/no_prune/N3_n12800_gmm_d8"           stale_gmm
move "$OUT/no_prune/N4_n25600_gmm_d8"           stale_gmm
move "$OUT/validated/V1_n3200_gmm_d8"           stale_gmm
move "$OUT/validated/V2_n6400_gmm_d8"           stale_gmm
move "$OUT/validated/V5_n3200_gmm_seeds123"     stale_gmm
move "$OUT/validated/V7_n12800_gmm_d8"          stale_gmm
move "$OUT/validated/V9_n25600_gmm_d8"          stale_gmm

# --- 2) CPU-REF: HiGHS vs CuPDLPx master ladder (an explicit GPU-vs-CPU comparison) ---
# CN: 这是一个**明确**的比较实验，保留其结论，但归档到 cpu_reference（不是生产路径）。
# EN: this is an EXPLICIT GPU-vs-CPU comparison; keep the conclusion but file it under
#     cpu_reference (it is not the production path).
move "$OUT/gpu_master_k"                        cpu_reference

echo
echo "[archive] KEEP (current, GPU scan hello + cupdlpx, non-gmm):"
find "$OUT/e2e" "$OUT/no_prune" "$OUT/validated" -maxdepth 1 -mindepth 1 -type d 2>/dev/null | sort

cat <<'NOTE'

[archive] done.

Notes:
  - Nothing deleted. Inspect _legacy_results/ before removing.
  - `queue/` logs are left in place (they are provenance for what ran, including the stale runs).
    Move them too if you want them out of the current view.
  - The GMM results are delete-tier: the gmm formula was wrong (mixture applied along the sample
    generation order instead of the geometry). Re-run anything you need with the corrected gmm.
NOTE
