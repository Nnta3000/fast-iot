"""
CN: **GATE 1 / E1（Phase 2）** —— 在真实 SynthRAD brain 探针切片上，先**固定度量**，再比较配对方法。

    CN: 顺序是硬性的：
      (a) **判别力检查**：度量必须在**同一中心内**把"同 z"与"不同 z"分开 ≥ 0.5 SD。
          通过不了的度量**不参与**评分——否则用噪声度量评估方法只会得到噪声。
      (b) **配对方法**：MR（行）对 CT（列），**跨患者**、同样大小（每侧 1000 片），
          每个患者最多 10 片、间隔 ≥ 8 层。用 (a) 保留下来的度量给每个方法打分。
      (c) **仅作健全性检查**：隐藏配对恢复（同患者注册伙伴），**不是**头条。

    CN: **预注册的门（GATE 1）**：cosine+OT 通过 ⇔ 它在保留度量上
        在全部 3 个 draw 上都比 random 高 ≥ 1 SD，且不比 z-only 基线差。
        失败时按顺序尝试：另一个 CT 窗、G1/G2/G3、K——不开始 E4。

    CN: **中心捷径必须失败**：因此在任何中心版之外，还要报**限定同中心**的配对变体。
        若只在 any-centre 上通过、同中心却不行，说明方法学的是"哪个扫描仪"。

EN: **GATE 1 / E1 (Phase 2)** -- fix the metrics first, then compare pairing methods, on real SynthRAD brain
    probe slices.

    The order is mandatory:
    (a) **Discrimination check**: a metric must separate same-z from different-z **within the same centre**
        by >= 0.5 SD. Metrics that fail are **not used** for scoring -- evaluating a method with a noise metric
        only yields noise.
    (b) **Pairing methods**: MR (rows) vs CT (columns), **cross-patient**, equal pools (1000 slices per side),
        at most 10 slices per patient and >= 8 apart. Every method is scored with the metrics kept in (a).
    (c) **Sanity only**: hidden-partner recovery (same-patient registered partner). Not a headline.

    **Pre-registered gate (GATE 1):** cosine+OT passes iff, on the kept metrics, it beats random by >= 1 SD on
    all 3 draws AND is not worse than the z-only baseline. On failure, try in order: the other CT window,
    G1/G2/G3, K. Do not start E4.

    **A centre shortcut must fail**, so alongside the any-centre matching we report a **same-centre-restricted**
    variant. Passing only on any-centre means the method learned the scanner, not the anatomy.
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import ot
from jaxtyping import Bool, Float, Int

_REPO_ROOT = Path(__file__).resolve().parents[4]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from iot_experiments.harness.cache import read_slices, save_slices, slices_path  # noqa: E402
from iot_experiments.harness.config import RunConfig, build_parser  # noqa: E402
from iot_experiments.harness.e1_metrics import mask_iou, nmi, psnr, ssim  # noqa: E402
from iot_experiments.harness.records import RunLog, RunRecord, format_table, summarise, write_records  # noqa: E402
from iot_experiments.harness.registry import apply_geometry, fit_geometry  # noqa: E402
from iot_experiments.harness.synthrad import (  # noqa: E402
    CT_WINDOW,
    IMAGE_SIZE,
    MIN_MASK_AREA,
    MR_PERCENTILES,
    PatientPaths,
    SynthRadSlices,
    discover_patients,
    preprocess_patient,
)

PROVENANCE: dict[str, object] = {
    "dataset": "SynthRAD2023 Task 1 brain, real files; probe patients only (test set untouched)",
    "preprocessing": "fixed: axial, 192x192, native-grid mask area >= 5%, CT fixed HU window, MR per-volume 1/99 pct",
    "features": "frozen DINOv2 pool4 -> PCA (label-free, both modalities) -> G2 or G3",
    "metrics_gate": "a metric is kept only if same-z vs different-z WITHIN the same centre separates by >= 0.5 SD",
    "gate": "cosine+OT passes iff beats random by >= 1 SD on all 3 draws on every kept metric AND >= z-only",
    "chance": "see the random floor; never compare across cells",
}

SECOND_CT_WINDOW: tuple[float, float] = (-200.0, 1500.0)      # bone-inclusive
SLICES_PER_SIDE: int = 1000
MAX_PER_PATIENT: int = 10
MIN_Z_SEPARATION: int = 8


class GateError(RuntimeError):
    """CN: Phase 2 的前置条件失败。EN: a Phase 2 precondition failed."""


# ---------------------------------------------------------------------------------------
# CN: 切片池构造：每患者最多 10 片且彼此间隔 >= 8 层（避免近乎重复的切片主导结果）。
# EN: pool construction: at most 10 slices per patient, >= 8 apart (so near-duplicate slices cannot
#     dominate).
# ---------------------------------------------------------------------------------------
def pick_slices(slices: SynthRadSlices, *, max_per_patient: int, min_gap: int, rng: np.random.Generator,
                width: int = IMAGE_SIZE) -> Int[np.ndarray, "picked"]:
    """
    CN: 在一个患者内部选出互隔 `min_gap` 的切片下标（返回在 `SynthRadSlices` 中的行号）。
    EN: pick within one patient slices at least `min_gap` apart (returning row indices into `SynthRadSlices`).
    """
    n = len(slices)
    if n == 0:
        return np.zeros(0, dtype=np.int64)
    order = rng.permutation(n)
    chosen: list[int] = []
    for idx in order:
        if all(abs(int(idx) - c) >= int(min_gap) for c in chosen):
            chosen.append(int(idx))
            if len(chosen) >= int(max_per_patient):
                break
    return np.sort(np.asarray(chosen, dtype=np.int64))


def build_pool(paths_list: list[PatientPaths], *, ct_window: tuple[float, float], cache: bool,
               max_per_patient: int, min_gap: int, seed: int
               ) -> tuple[Float[np.ndarray, "n pixels"], Float[np.ndarray, "n pixels"],
                          Bool[np.ndarray, "n pixels"], Float[np.ndarray, "n"],
                          Int[np.ndarray, "n"], list[str], list[str]]:
    """
    CN: 把若干患者展开成一个大池：返回 MR、CT、掩膜、`z_normalised`、原始层号、患者 id、中心。
    EN: flatten patients into one pool: MR, CT, mask, `z_normalised`, native z index, patient id, centre.
    """
    rng = np.random.default_rng(int(seed))
    mr, ct, mask, z, zidx, pid, ctr = [], [], [], [], [], [], []
    for paths in paths_list:
        slices = load_cached(paths, ct_window=ct_window, cache=cache)
        picked = pick_slices(slices, max_per_patient=max_per_patient, min_gap=min_gap, rng=rng)
        for i in picked:
            mr.append(slices.mr[i]); ct.append(slices.ct[i]); mask.append(slices.mask[i])
            z.append(float(slices.z_normalised[i])); zidx.append(int(slices.z_index[i]))
            pid.append(slices.patient); ctr.append(slices.centre)
    if not mr:
        raise GateError("empty pool")
    return (np.stack(mr), np.stack(ct), np.stack(mask), np.asarray(z, dtype=np.float64),
            np.asarray(zidx, dtype=np.int64), pid, ctr)


def load_cached(paths: PatientPaths, *, ct_window: tuple[float, float], cache: bool) -> SynthRadSlices:
    """CN: 读缓存或预处理并写缓存（键含 CT 窗，故两个窗各自缓存）。EN: cached read or preprocess+store."""
    target = slices_path(paths.patient, size=IMAGE_SIZE, ct_window=tuple(ct_window),
                         mr_percentiles=MR_PERCENTILES, min_mask_area=MIN_MASK_AREA,
                         anatomy=paths.anatomy.lower())
    if cache and target.is_file():
        data = read_slices(target)
        return SynthRadSlices(patient=str(data["meta"]["patient"]), centre=str(data["meta"]["centre"]),
                              mr=np.asarray(data["mr"]), ct=np.asarray(data["ct"]),
                              mask=np.asarray(data["mask"]), z_normalised=np.asarray(data["z_normalised"]),
                              z_index=np.asarray(data["z_index"]), meta=dict(data["meta"]))
    slices = preprocess_patient(paths, ct_window=tuple(ct_window))
    if cache:
        save_slices(target, mr=slices.mr, ct=slices.ct, mask=slices.mask,
                    z_normalised=slices.z_normalised, z_index=slices.z_index,
                    patient=slices.patient, centre=slices.centre, meta=slices.meta)
    return slices


# ---------------------------------------------------------------------------------------
# CN: (a) 判别力检查的四组：同患者同层 / 同中心同 z / 同中心不同 z / 其他中心。
# EN: (a) the four groups: same-patient same-slice, same-centre same-z, same-centre different-z, other-centre.
# ---------------------------------------------------------------------------------------
def metric_rows(mr: Float[np.ndarray, "n pixels"], ct: Float[np.ndarray, "n pixels"],
                mask: Bool[np.ndarray, "n pixels"], z: Float[np.ndarray, "n"],
                pid: list[str], ctr: list[str]) -> list[dict[str, np.ndarray]]:
    """
    CN: `discrimination_check` 需要的行：`a` 侧带 `truth`（该 MR 的真实 CT），`b` 侧是候选 CT。
    EN: rows for `discrimination_check`: the `a` side carries `truth` (the MR's own CT), `b` is a candidate.
    """
    return [{"mr": mr[i], "mask": mask[i], "z": np.asarray(z[i]), "truth": ct[i],
             "ct": ct[i], "ctmask": mask[i], "pid": pid[i], "ctr": ctr[i]} for i in range(len(mr))]


def group_pairs(rows: list[dict[str, np.ndarray]], *, seed: int, n_per_group: int
                ) -> dict[str, list[tuple[int, int]]]:
    """
    CN: 构造四组配对下标。**关键在于同中心对照**：同中心同 z 才是"同一台扫描仪"的选择性对照组。
        同中心不同 z 用 `|dz| > 0.2`，同中心同 z 用 `|dz| < 0.05`。
    EN: build the four index groups. The **same-centre control** is the point: same-centre same-z is the
        "same scanner" selective control. Different-z within a centre uses `|dz| > 0.2`, same-z uses
        `|dz| < 0.05`.
    """
    rng = np.random.default_rng(int(seed))
    groups: dict[str, list[tuple[int, int]]] = {"same_patient_same_slice": [],
                                                "same_centre_same_z": [],
                                                "same_centre_diff_z": [],
                                                "other_centre": []}
    n = len(rows)
    for i in range(n):
        groups["same_patient_same_slice"].append((i, i))
    for _ in range(int(n_per_group) * 4):
        i = int(rng.integers(0, n)); j = int(rng.integers(0, n))
        if i == j:
            continue
        same_centre = rows[i]["ctr"] == rows[j]["ctr"]
        dz = abs(float(rows[i]["z"]) - float(rows[j]["z"]))
        same_patient = rows[i]["pid"] == rows[j]["pid"]
        if same_patient:
            continue                                  # CN: 配对方法一律跨患者，这里也排除。
        if same_centre and dz < 0.05 and len(groups["same_centre_same_z"]) < n_per_group:
            groups["same_centre_same_z"].append((i, j))
        elif same_centre and dz > 0.2 and len(groups["same_centre_diff_z"]) < n_per_group:
            groups["same_centre_diff_z"].append((i, j))
        elif (not same_centre) and len(groups["other_centre"]) < n_per_group:
            groups["other_centre"].append((i, j))
    return {k: v for k, v in groups.items() if v}


def metric_fns() -> dict[str, object]:
    """
    CN: 候选度量。`mask_iou`、`z`（= 1-|dz|）、`nmi` 不需要真值；`ssim`/`psnr`/`ct_mask_iou`
        用"选中 CT 对 MR 的真实 CT"——它们**只用于评估**，绝不能当配对代价（会泄漏答案）。
    EN: candidate metrics. `mask_iou`, `z` (= 1-|dz|) and `nmi` need no truth; `ssim`/`psnr`/`ct_mask_iou`
        compare the chosen CT against the MR's own true CT -- **evaluation only**, never a pairing cost.
    """
    return {
        "mask_iou": lambda a, b: mask_iou(np.asarray(a["mask"], bool), np.asarray(b["ctmask"], bool)),
        "z": lambda a, b: float(np.clip(1.0 - abs(float(a["z"]) - float(b["z"])), 0.0, 1.0)),
        "nmi": lambda a, b: nmi(np.asarray(a["mr"]), np.asarray(b["ct"])),
        "ssim": lambda a, b: ssim(np.asarray(b["ct"]), np.asarray(a["truth"])),
        "psnr": lambda a, b: min(psnr(np.asarray(b["ct"]), np.asarray(a["truth"])), 99.0),
        # CN: `ct_mask_iou` 已删除——它与 `mask_iou` 是同一个比较（同一对掩膜），实测数值完全相同。
        # EN: `ct_mask_iou` is gone -- it computed the same comparison as `mask_iou` over the same mask pair
        #     and produced numerically identical values.
    }


def run_metric_gate(rows: list[dict[str, np.ndarray]], *, seed: int, n_per_group: int,
                    threshold: float = 0.5) -> tuple[list[str], dict[str, dict[str, float]], str]:
    """
    CN: 对每个度量，在同中心内比较"同 z"与"不同 z"的效应量。保留判决见返回值第一项。
        这里**不用** `discrimination_check` 的通用三组接口，因为本阶段需要**四组**且判据是
        "同 z vs 不同 z（同中心）"，这正是中心捷径的判据。
    EN: for each metric, compare same-z against different-z **within a centre**. The kept list is the first
        return value. The generic three-group `discrimination_check` is not used here because this phase needs
        **four groups** and the criterion is "same-z vs different-z within a centre", which is precisely the
        centre-shortcut test.
    """
    groups = group_pairs(rows, seed=seed, n_per_group=n_per_group)
    fns = metric_fns()
    effects: dict[str, dict[str, float]] = {}
    for name, fn in fns.items():
        values: dict[str, list[float]] = {g: [] for g in groups}
        for gname, pairs in groups.items():
            for i, j in pairs:
                values[gname].append(float(fn(rows[i], rows[j])))  # type: ignore[operator]
        same = np.asarray(values.get("same_centre_same_z", []), dtype=np.float64)
        diff = np.asarray(values.get("same_centre_diff_z", []), dtype=np.float64)
        other = np.asarray(values.get("other_centre", []), dtype=np.float64)
        truth = np.asarray(values.get("same_patient_same_slice", []), dtype=np.float64)
        sd = float(diff.std()) if diff.size > 1 else float("nan")
        eff = float((same.mean() - diff.mean()) / sd) if np.isfinite(sd) and sd > 0 else float("nan")
        effects[name] = {
            "mean_same_patient": round(float(truth.mean()), 4) if truth.size else float("nan"),
            "mean_same_centre_same_z": round(float(same.mean()), 4) if same.size else float("nan"),
            "mean_same_centre_diff_z": round(float(diff.mean()), 4) if diff.size else float("nan"),
            "mean_other_centre": round(float(other.mean()), 4) if other.size else float("nan"),
            "sd_diff_z": round(sd, 4),
            "effect_same_z_vs_diff_z": round(eff, 3),
        }
    kept = [m for m, e in effects.items()
            if np.isfinite(e["effect_same_z_vs_diff_z"]) and e["effect_same_z_vs_diff_z"] >= float(threshold)]
    verdict_lines = [f"{m:<12} effect(same-z vs diff-z, same centre) = {e['effect_same_z_vs_diff_z']:+.2f} SD"
                     + ("  KEPT" if m in kept else "  rejected") for m, e in effects.items()]
    return kept, effects, "\n".join(verdict_lines)


# ---------------------------------------------------------------------------------------
# CN: (b) 配对方法。全部跨患者；相等大小 1000 vs 1000。
# EN: (b) pairing methods, all cross-patient, equal pools of 1000 vs 1000.
# ---------------------------------------------------------------------------------------
@dataclass(frozen=True)
class PairingResult:
    """CN: 一个方法的配对结果与耗时。EN: one method's pairing and its timing."""

    name: str
    row_of: Int[np.ndarray, "m"]        # CN: 行池里哪些行参与（同中心变体是子集）/ EN: which rows participate
    col_of: Int[np.ndarray, "m"]
    perm: Int[np.ndarray, "m"]          # CN: 行 i 配到列的本地下标 / EN: row i matched to a local column
    seconds: float


def random_pairing(n: int, *, seed: int) -> Int[np.ndarray, "n"]:
    """CN: 均匀随机置换（下界）。EN: uniform random permutation (the floor)."""
    return np.random.default_rng(int(seed)).permutation(n).astype(np.int64)


def z_only_pairing(z_rows: Float[np.ndarray, "m"], z_cols: Float[np.ndarray, "m"]
                   ) -> Int[np.ndarray, "m"]:
    """
    CN: 只用 `z_normalised` 的配对：代价 = |dz|，Hungarian 求解。它是"只用元数据"的基线。
    EN: pair on `z_normalised` alone: cost = |dz|, solved by Hungarian. The metadata-only baseline.
    """
    from scipy.optimize import linear_sum_assignment  # noqa: PLC0415

    cost = np.abs(z_rows[:, None] - z_cols[None, :])
    _rows, cols = linear_sum_assignment(cost)
    return cols.astype(np.int64)


def mask_iou_pairing(mask_rows: Bool[np.ndarray, "m pixels"], mask_cols: Bool[np.ndarray, "m pixels"]
                     ) -> Int[np.ndarray, "m"]:
    """CN: 只用身体掩膜 IoU 的配对（元数据基线）。EN: pair on body-mask IoU (a metadata baseline)."""
    from scipy.optimize import linear_sum_assignment  # noqa: PLC0415

    a = np.asarray(mask_rows, dtype=np.float64); b = np.asarray(mask_cols, dtype=np.float64)
    inter = a @ b.T
    union = a.sum(axis=1)[:, None] + b.sum(axis=1)[None, :] - inter
    iou = inter / np.maximum(union, 1.0)
    _rows, cols = linear_sum_assignment(-iou)
    return cols.astype(np.int64)


def nmi_cost_matrix(mr: Float[np.ndarray, "m pixels"], ct: Float[np.ndarray, "n pixels"], *,
                    size: int = 128, bins: int = 32, block: int = 64) -> Float[np.ndarray, "m n"]:
    """
    CN: GPU 分块的像素 NMI 代价矩阵（**负**互信息，越小越好）。`size=128, bins=32` ⇒ 16 样本/格子。
        实测：1000×1000 约 0.1 分钟（RTX 4060 Laptop），因此可以完整跑。
    EN: GPU-batched pixel-NMI cost matrix (**negative** MI, lower is better). `size=128, bins=32` gives 16
        samples per cell. Measured: a full 1000x1000 takes about 0.1 min on the RTX 4060 Laptop, so it runs.
    """
    import torch  # noqa: PLC0415

    from iot_experiments.harness.e1_metrics import _square  # noqa: PLC0415

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    a = torch.tensor(np.stack([_square(mr[i], size=size).ravel() for i in range(mr.shape[0])]),
                     dtype=torch.float32, device=dev)
    b = torch.tensor(np.stack([_square(ct[j], size=size).ravel() for j in range(ct.shape[0])]),
                     dtype=torch.float32, device=dev)

    def codes(x: "object") -> "object":
        qs = torch.linspace(0, 1, int(bins) + 1, device=x.device, dtype=x.dtype)[1:-1]
        edges = torch.quantile(x, qs, dim=1)
        return (x.unsqueeze(1) > edges.t().unsqueeze(-1)).sum(1).long()

    ca, cb = codes(a), codes(b)
    R, C, P = a.shape[0], b.shape[0], a.shape[1]
    out = np.empty((R, C), dtype=np.float64)
    for r0 in range(0, R, int(block)):
        r1 = min(r0 + int(block), R)
        idx = (ca[r0:r1, None, :] * int(bins) + cb[None, :, :]).reshape((r1 - r0) * C, P)
        joint = torch.zeros((r1 - r0) * C, int(bins) * int(bins), device=dev).scatter_add_(
            1, idx, torch.ones_like(idx, dtype=a.dtype)).reshape(r1 - r0, C, int(bins), int(bins)) / P
        pa = joint.sum(-1, keepdim=True); pb = joint.sum(-2, keepdim=True)
        t = torch.where(joint > 0, joint * torch.log(joint / (pa * pb + 1e-30) + 1e-30),
                        torch.zeros_like(joint))
        out[r0:r1] = t.sum(dim=(-1, -2)).double().cpu().numpy()
    return -out


def hungarian(cost: Float[np.ndarray, "m n"]) -> Int[np.ndarray, "m"]:
    """
    CN: 精确指派，返回 `perm[i] = j`。**要求方阵**：矩形代价下"指派"不再是置换，配对就不再可比，
        因此这里显式拒绝，而不是悄悄只解一半。
    EN: exact assignment, returning `perm[i] = j`. **Square only**: on a rectangular cost the assignment is no
        longer a permutation and pairings stop being comparable, so this refuses explicitly instead of quietly
        solving only part of it.
    """
    from scipy.optimize import linear_sum_assignment  # noqa: PLC0415

    matrix = np.asarray(cost, dtype=np.float64)
    if matrix.shape[0] != matrix.shape[1]:
        raise GateError(f"hungarian needs a square cost, got {matrix.shape}")
    rows, cols = linear_sum_assignment(matrix)
    perm = np.empty(matrix.shape[0], dtype=np.int64)
    perm[rows] = cols
    return perm


def emd(cost: Float[np.ndarray, "m n"]) -> Int[np.ndarray, "m"]:
    """
    CN: 精确 OT（POT 网络单纯形），均匀边际；方阵时耦合即置换。**注意**：POT 的计划元素是 1/m，
        乘 m 才是与指派可比的目标值——这里只需要置换，故不受影响。
    EN: exact OT (POT network simplex) with uniform marginals; for a square cost the coupling is a permutation.
        Note POT's plan entries are 1/m, so objective values need a factor of m; only the permutation is used.
    """
    a = np.asarray(cost, dtype=np.float64)
    m, n = a.shape
    if m != n:
        raise GateError(f"emd pairing needs a square cost, got {a.shape}")
    weights = np.ones(m) / m
    plan = ot.emd(weights, weights, a, numItermax=50_000_000)
    return np.argmax(plan, axis=1).astype(np.int64)


def evaluate_pairing(perm: Int[np.ndarray, "m"], *, rows: list[dict[str, np.ndarray]],
                     row_index: Int[np.ndarray, "m"], col_index: Int[np.ndarray, "m"], kept: list[str]
                     ) -> dict[str, float]:
    """
    CN: 用保留下来的度量给一个配对打分（均值）。**本地下标 ↔ 池下标**必须显式映射：
        第 `k` 个本地行是池里的 `row_index[k]`，它配到的本地列是 `perm[k]`，对应池里的 `col_index[perm[k]]`。
        （早先直接用本地下标索引 `rows`，池大于本地子集时就会越界。）
    EN: score a pairing with the kept metrics (means). The **local-to-pool mapping must be explicit**: the
        `k`-th local row is pool index `row_index[k]`, matched to local column `perm[k]`, i.e. pool index
        `col_index[perm[k]]`. (An earlier version indexed `rows` with the local index, which goes out of bounds
        whenever the pool is larger than the local subset.)
    """
    fns = metric_fns()
    out: dict[str, float] = {}
    for name in kept:
        fn = fns[name]
        values = [float(fn(rows[int(row_index[k])], rows[int(col_index[int(perm[k])])]))  # type: ignore[operator]
                  for k in range(len(perm))]
        out[name] = float(np.mean(values))
    out["same_patient_fraction"] = float(np.mean([
        rows[int(row_index[k])]["pid"] == rows[int(col_index[int(perm[k])])]["pid"]
        for k in range(len(perm))]))
    return out


@dataclass(frozen=True)
class GateConfig(RunConfig):
    """CN: Phase 2 的旋钮。EN: Phase 2's knobs."""

    section = "synthrad_gate"
    study: str = "synthrad_gate"
    probe_per_centre: int = 10
    split_seed: int = 7
    ks: tuple[int, ...] = (128, 256)
    geometries: tuple[str, ...] = ("G2", "G3")
    ct_windows: tuple[str, ...] = ("soft", "bone")
    encoders: tuple[str, ...] = ("dinov2", "pixpca")
    draw_seeds: tuple[int, ...] = (0, 1, 2)
    cache: bool = True

    @classmethod
    def add_study_arguments(cls, parser: argparse.ArgumentParser) -> None:
        """CN: 只加本阶段特有的参数。EN: only this phase's flags."""
        parser.add_argument("--probe-per-centre", type=int, default=10)
        parser.add_argument("--split-seed", type=int, default=7)
        parser.add_argument("--ks", type=int, nargs="+", default=[128, 256])
        parser.add_argument("--geometries", nargs="+", default=["G2", "G3"], choices=["G1", "G2", "G3"])
        parser.add_argument("--ct-windows", nargs="+", default=["soft", "bone"],
                            choices=["soft", "bone"],
                            help="soft = -200..300 HU; bone = -200..1500 HU")
        parser.add_argument("--encoders", nargs="+", default=["dinov2", "pixpca"],
                            choices=["dinov2", "pixpca"],
                            help="dinov2 = frozen DINOv2 pool4 (as specified); pixpca = PCA on raw pixels")
        parser.add_argument("--draw-seeds", type=int, nargs="+", default=[0, 1, 2])
        parser.add_argument("--cache", action=argparse.BooleanOptionalAction, default=True)


def window_of(name: str) -> tuple[float, float]:
    """CN: 窗名 → HU 区间。EN: window name -> HU interval."""
    return CT_WINDOW if name == "soft" else SECOND_CT_WINDOW


def main(argv: list[str] | None = None) -> int:
    """CN: 入口。EN: entry point."""
    parser = build_parser(GateConfig)
    args = parser.parse_args(list(argv) if argv is not None else None)
    config = GateConfig.from_namespace(args, parser)

    log = RunLog(config.run_dir / "run.log")
    log.line(f"config: encoders={config.encoders} ks={config.ks} geometries={config.geometries} "
             f"windows={config.ct_windows} "
             f"draws={config.draw_seeds} probe_per_centre={config.probe_per_centre} seed={config.split_seed}")

    manifest_path = config.run_dir.parent / "synthrad_probe" / "split_manifest.json"
    if manifest_path.is_file():
        import json  # noqa: PLC0415

        probe_ids = set(json.loads(manifest_path.read_text())["probe"])
        log.line(f"reusing the Phase 1 split manifest ({len(probe_ids)} probe patients); test set untouched")
    else:
        probe_ids = set()
        log.line("WARNING: no Phase 1 split manifest found; will rebuild the split here")

    patients = discover_patients(_REPO_ROOT / "data" / "Task1", anatomy="brain")
    if probe_ids:
        patients = [p for p in patients if p.patient in probe_ids]

    records: list[RunRecord] = []
    for window_name in config.ct_windows:
        window = window_of(window_name)
        t0 = time.perf_counter()
        mr, ct, mask, z, zidx, pid, ctr = build_pool(patients, ct_window=window, cache=bool(config.cache),
                                                     max_per_patient=MAX_PER_PATIENT,
                                                     min_gap=MIN_Z_SEPARATION, seed=int(config.split_seed))
        log.line(f"[window={window_name} {window}] pool: {len(mr)} slices from {len(set(pid))} patients "
                 f"({time.perf_counter() - t0:.1f}s)")
        rows = metric_rows(mr, ct, mask, z, pid, ctr)

        # CN: **(a) 先固定度量**，再评分。判定用的是同中心内 same-z vs diff-z。
        # EN: **(a) fix the metrics first**, then score. The criterion is same-z vs diff-z within a centre.
        kept, effects, verdict_text = run_metric_gate(rows, seed=int(config.split_seed), n_per_group=400)
        log.line("metric gate (same-centre control):")
        for line in verdict_text.splitlines():
            log.line(f"  {line}")
        log.line(f"kept metrics: {kept}")
        (config.run_dir / f"metric_gate_{window_name}.json").write_text(
            __import__("json").dumps(effects, indent=2) + "\n", encoding="utf-8")
        if not kept:
            log.line("NO metric separated same-z from different-z within a centre; stopping before scoring.")
            continue

        # CN: 特征用**两模态共享**的 encoder 与 label-free PCA，再按几何标准化。
        # EN: features use **one shared encoder** and label-free PCA, then the geometry.
        for encoder in config.encoders:
          for k in config.ks:
            for geometry_name in config.geometries:
                # CN: PCA 基拟合在两模态全部池切片上（无标签）；G2/G3 的统计量取自**池**（未标注）。
                # EN: the PCA basis is fitted on all pool slices of both modalities (no labels); the G2/G3
                #     statistics come from the **pool** (unlabelled).
                feat_mr, feat_ct = _features(mr, ct, k=int(k), geometry_name=str(geometry_name),
                                              encoder=str(encoder))
                for draw in config.draw_seeds:
                    row_sel, col_sel = _split_pools(rows, n=SLICES_PER_SIDE, seed=int(draw), same_centre=None)
                    if len(row_sel) < SLICES_PER_SIDE:
                        log.line(f"  note: pool yields only {len(row_sel)} slices per side (30 probe patients "
                                 f"x <= {MAX_PER_PATIENT}/patient), not the {SLICES_PER_SIDE} target")
                    zr = z[row_sel]; zc = z[col_sel]
                    methods: dict[str, PairingResult] = {}
                    t1 = time.perf_counter()
                    # CN: 一律用**实际池大小**（`len(row_sel)`），不是目标上限 `SLICES_PER_SIDE`：
                    #     30 个探针患者 × 每人 ≤10 片只能给出 300 片，两侧各 150 片。
                    # EN: always use the **actual pool size** (`len(row_sel)`), never the target cap
                    #     `SLICES_PER_SIDE`: 30 probe patients at <=10 slices each yield 300 slices, i.e. 150
                    #     per side.
                    methods["random"] = PairingResult("random", row_sel, col_sel,
                                                      random_pairing(len(row_sel), seed=int(draw)),
                                                      time.perf_counter() - t1)
                    t1 = time.perf_counter()
                    methods["z_only"] = PairingResult("z_only", row_sel, col_sel,
                                                      z_only_pairing(zr, zc), time.perf_counter() - t1)
                    t1 = time.perf_counter()
                    methods["mask_iou"] = PairingResult("mask_iou", row_sel, col_sel,
                                                        mask_iou_pairing(mask[row_sel], mask[col_sel]),
                                                        time.perf_counter() - t1)
                    t1 = time.perf_counter()
                    nmi_cost = nmi_cost_matrix(mr[row_sel], ct[col_sel])
                    methods["nmi"] = PairingResult("nmi", row_sel, col_sel, hungarian(nmi_cost),
                                                   time.perf_counter() - t1)
                    t1 = time.perf_counter()
                    cos_cost = _cosine_cost(feat_mr[row_sel], feat_ct[col_sel])
                    methods["cosine_ot"] = PairingResult("cosine_ot", row_sel, col_sel, emd(cos_cost),
                                                         time.perf_counter() - t1)
                    t1 = time.perf_counter()
                    methods["cosine_nn"] = PairingResult("cosine_nn", row_sel, col_sel,
                                                         np.argmin(cos_cost, axis=1).astype(np.int64),
                                                         time.perf_counter() - t1)
                    for name, res in methods.items():
                        col_index = col_sel
                        metrics = evaluate_pairing(res.perm, rows=rows, row_index=res.row_of,
                                                   col_index=col_index, kept=kept)
                        records.append(RunRecord(
                            study=str(config.study),
                            geometry=f"{encoder}|{geometry_name}|K{k}|{window_name}", arm=name, k=int(k), m=0,
                            draw=int(draw), metrics=metrics,
                            config=config.to_dict(),
                            extra={"window": window_name, "ct_window": list(window), "draw": int(draw),
                                   "seconds": round(res.seconds, 3),
                                   "n_rows": int(len(row_sel)), "n_cols": int(len(col_sel))}))
                        log.line(f"[{encoder} {geometry_name} K={k} {window_name} draw={draw}] {name:<11} "
                                 + " ".join(f"{m}={metrics[m]:.4f}" for m in kept)
                                 + f" same_patient={metrics['same_patient_fraction']:.3f} "
                                   f"({res.seconds:.2f}s)")

                    # CN: **同中心限定**变体——中心捷径若存在，这里会失效。
                    # EN: **same-centre-restricted** variant: a centre shortcut fails here.
                    for centre in sorted({rows[i]["ctr"] for i in row_sel}):
                        r_sub, c_sub = _split_pools(rows, n=min(300, SLICES_PER_SIDE), seed=int(draw),
                                                    same_centre=centre)
                        if len(r_sub) < 20 or len(c_sub) < 20:
                            continue
                        cos_sub = _cosine_cost(feat_mr[r_sub], feat_ct[c_sub])
                        perm_sub = emd(cos_sub)
                        m_sub = evaluate_pairing(perm_sub, rows=rows, row_index=r_sub,
                                                 col_index=c_sub, kept=kept)
                        m_sub["rand"] = evaluate_pairing(random_pairing(len(r_sub), seed=int(draw)),
                                                         rows=rows, row_index=r_sub,
                                                         col_index=c_sub, kept=kept)[kept[0]]
                        records.append(RunRecord(
                            study=str(config.study),
                            geometry=f"{encoder}|{geometry_name}|K{k}|{window_name}|centre{centre}",
                            arm="cosine_ot_same_centre", k=int(k), m=0, draw=int(draw), metrics=m_sub,
                            config=config.to_dict(),
                            extra={"window": window_name, "centre": centre, "draw": int(draw),
                                   "n_rows": int(len(r_sub)), "n_cols": int(len(c_sub))}))
                        log.line(f"  same-centre {centre}: cosine_ot "
                                 + " ".join(f"{m}={m_sub[m]:.4f}" for m in kept)
                                 + f" | random {m_sub['rand']:.4f}")

    rows_out = summarise(records, group=("geometry", "arm"), metrics=tuple(
        f for f in ("mask_iou", "z", "nmi", "ssim", "psnr", "ct_mask_iou") if any(
            f in r.metrics for r in records)))
    log.line("\n" + format_table(rows_out, ("geometry", "arm", "n_draws") + tuple(
        f"{f}_mean" for f in ("mask_iou", "z", "nmi", "ssim", "psnr", "ct_mask_iou")
        if any(f in r.metrics for r in records))))

    write_records(config.run_dir / "records.json", records,
                  provenance={**PROVENANCE, "metric_effects": "see metric_gate_*.json"})
    log.line(f"wrote {config.run_dir / 'records.json'}")
    log.close()
    return 0


def _features(mr: Float[np.ndarray, "n pixels"], ct: Float[np.ndarray, "n pixels"], *, k: int,
              geometry_name: str, encoder: str) -> tuple[Float[np.ndarray, "n k"],
                                                         Float[np.ndarray, "n k"]]:
    """
    CN: 用一个**共享 encoder** 把两模态变成特征，然后套用几何并逐行 L2。
        `encoder="dinov2"` 用冻结的 DINOv2 pool4（与 T1/T2 那条路径相同）；`encoder="pixpca"` 是
        直接在像素上做 PCA（早期版本用的就是这个，保留用于对照）。
        PCA 在**两模态全部池切片**上拟合（无标签）；几何统计量同样取自池。
    EN: turn both modalities into features with **one shared encoder**, then apply the geometry and L2-normalise
        each row. `encoder="dinov2"` uses frozen DINOv2 pool4 (the same path as the T1/T2 work);
        `encoder="pixpca"` does PCA on raw pixels (what the earlier version used, kept for comparison).
        PCA is fitted on **all pool slices of both modalities** (label-free); the geometry statistics likewise
        come from the pool.
    """
    from iot_experiments.harness.dinov2 import encode_pool4, pca_apply, pca_fit  # noqa: PLC0415

    if encoder == "dinov2":
        side = int(round(float(np.asarray(mr).shape[1]) ** 0.5))
        pool_mr = np.asarray(mr).reshape(-1, side, side)
        pool_ct = np.asarray(ct).reshape(-1, side, side)
        feats_mr = encode_pool4(pool_mr, device="cuda").pooled.astype(np.float64)
        feats_ct = encode_pool4(pool_ct, device="cuda").pooled.astype(np.float64)
        basis = pca_fit(np.concatenate([feats_mr, feats_ct], axis=0), k=int(k))
        proj_mr, proj_ct = pca_apply(feats_mr, basis), pca_apply(feats_ct, basis)
    elif encoder == "pixpca":
        pooled = np.concatenate([mr, ct], axis=0)
        mean = pooled.mean(axis=0)
        _u, _s, vt = np.linalg.svd(pooled - mean[None, :], full_matrices=False)
        kk = int(min(k, vt.shape[0]))
        basis_px = vt[:kk].T
        proj_mr, proj_ct = (mr - mean) @ basis_px, (ct - mean) @ basis_px
    else:
        raise GateError(f"unknown encoder {encoder!r}")
    stats = fit_geometry(proj_mr, proj_ct, str(geometry_name))
    out_mr, out_ct = apply_geometry(proj_mr, proj_ct, stats)
    return np.asarray(out_mr, dtype=np.float64), np.asarray(out_ct, dtype=np.float64)


def _cosine_cost(a: Float[np.ndarray, "m k"], b: Float[np.ndarray, "n k"]) -> Float[np.ndarray, "m n"]:
    """CN: 余弦距离 `1 - <a,b>`（行已 L2 归一化）。EN: cosine distance `1 - <a,b>` (rows already unit-norm)."""
    return 1.0 - np.asarray(a, dtype=np.float64) @ np.asarray(b, dtype=np.float64).T


def _split_pools(rows: list[dict[str, np.ndarray]], *, n: int, seed: int, same_centre: str | None
                 ) -> tuple[Int[np.ndarray, "m"], Int[np.ndarray, "m"]]:
    """
    CN: 从池中抽**跨患者**、等大的行/列下标。所有行与列必须来自**不同患者**（同一患者不得同时出现在两侧）。
        `same_centre` 给定时只在那个中心内抽。
    EN: draw **cross-patient**, equal-size row/column indices. No patient may appear on both sides.
        With `same_centre`, restrict to that centre.
    """
    rng = np.random.default_rng(int(seed))
    eligible = [i for i, r in enumerate(rows)
                if (same_centre is None or r["ctr"] == same_centre)]
    by_patient: dict[str, list[int]] = {}
    for i in eligible:
        by_patient.setdefault(str(rows[i]["pid"]), []).append(i)
    patients = sorted(by_patient)
    if len(patients) < 2:
        raise GateError("need at least two patients to build a cross-patient pool")
    rng.shuffle(patients)
    half = len(patients) // 2
    left_patients, right_patients = patients[:half], patients[half:]

    def take(group: list[str], target: int) -> list[int]:
        picked: list[int] = []
        cursor = 0
        while len(picked) < target and group:
            progressed = False
            for pid_name in group:
                items = by_patient[pid_name]
                if cursor < len(items):
                    picked.append(items[cursor]); progressed = True
                    if len(picked) >= target:
                        break
            if not progressed:
                break
            cursor += 1
        return picked

    left = take(left_patients, int(n))
    right = take(right_patients, int(n))
    # CN: **强制等大**：配对是指派，两侧不等大就不再是置换。取两侧的较小者。
    # EN: **force equal sides**: pairing is an assignment, so unequal sides are no longer a permutation. Keep
    #     the smaller of the two.
    size = min(len(left), len(right))
    if size < 2:
        raise GateError(f"could not build a cross-patient pool of equal sides (left={len(left)}, right={len(right)})")
    return (np.asarray(left[:size], dtype=np.int64), np.asarray(right[:size], dtype=np.int64))


if __name__ == "__main__":
    raise SystemExit(main())
