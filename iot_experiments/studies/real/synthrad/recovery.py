"""
CN: **配对恢复测试（按 ground truth 计分）** —— 把"配对"当作**检索**问题来评估，而不是相似度代理。

    CN: **核心思路（用户提出、正确）**：SynthRAD 给出同一患者、同一体素网格上的 MR 与 CT，
        因此每个切片都有一个**精确的身份** `(患者, z 序号)`。把对应关系藏起来，让每个方法给出一个配对，
        然后问"你配到的 id 是否等于真实 id"。**计分对象是标签，不是另一个方法**——没有方法被当作标尺。

    CN: 为什么这不同于此前的 Phase 2：那次的池子故意让真实伙伴**不在**池内（跨患者），
        因此只能测"看起来配不配"，元数据基线必然占优。这里**真实伙伴在池内**，可以测恢复。

    CN: **两个指标，分开报**：
        * `patient_hit` —— 配到的切片是否来自**同一患者**（真正的"找到真配对"问题）；
        * `exact_hit`   —— 是否配到了**完全相同的那一层**（更严；差距说明"人对了、层错了"）。
        随机基线是 `1/n`，显式给出。

    CN: **臂**（这是本研究的主线：**学到的代价**必须与余弦同场竞技）：
        | 臂 | 说明 |
        |---|---|
        | `random` | 随机置换（下界） |
        | `z_only` | 只按 `z_normalised` 配对（元数据基线） |
        | `mask_iou` | 只按身体掩膜重叠配对（元数据基线） |
        | `nmi` | 像素互信息 + 匈牙利 |
        | `cosine_ot` | 特征余弦 + 精确 OT |
        | **`learned`** | **逆 OT 学到的对角代价**（在训练患者的真实配对上拟合，盒子 + 单纯形） |
        | `oracle` | 用真实代价喂给求解器：**必须恰好 1.0**（断言） |

    CN: `learned` 的 θ 在**训练患者**上拟合，评估在**留出患者**的池上进行；同一 θ 用于所有画次。
        θ 的应用空间由 `assert` 保证与拟合空间一致（这是此前踩过的坑）。

EN: **Pair recovery scored against ground truth** -- evaluate pairing as **retrieval**, not a similarity proxy.

    **The idea (the user's, and correct):** SynthRAD ships MR and CT on the *same voxel grid* for a patient,
    so every slice has an exact identity `(patient, z index)`. Hide the correspondence, let each method
    produce a pairing, then ask whether the matched id equals the true id. **The label is the yardstick, not
    another method** -- no method is used as the reference.

    Unlike the earlier Phase 2 run, whose pool **deliberately excluded** the true partner (cross-patient) and
    so could only measure "looks compatible" (metadata baselines necessarily won), here **the true partner is
    in the pool**, so recovery can be measured.

    **Two metrics, reported separately:** `patient_hit` (did the matched slice come from the **same patient**,
    the real "found the true pair" question) and `exact_hit` (did it match the **exact same slice**, stricter;
    the gap between them means "right person, wrong level"). The random floor is `1/n`, stated explicitly.

    **Arms** -- and the point of the study is that the **learned cost** competes head-to-head with cosine:
    `random`, `z_only`, `mask_iou`, `nmi`, `cosine_ot`, **`learned`** (inverse-OT diagonal cost fitted on
    training patients' true pairs, box + simplex), and `oracle` (true cost into the solver, must be exactly
    1.0).

    theta is fitted on **training patients** and evaluated on **held-out patients**; one theta serves all
    draws. An assertion guarantees theta is applied in the same space it was fitted in.
"""

from __future__ import annotations

import argparse
import json
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
from iot_experiments.harness.dinov2 import encode_pool4, pca_apply, pca_fit  # noqa: E402
from iot_experiments.harness.records import RunLog, RunRecord, format_table, summarise, write_records  # noqa: E402
from iot_experiments.harness.registry import apply_geometry, fit_geometry  # noqa: E402
from iot_experiments.harness.solver import diagonal_cost, fit_full_lp  # noqa: E402
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
    "dataset": "SynthRAD2023 Task 1 brain, real files",
    "task": "PAIR RECOVERY: hide the correspondence, score against the (patient, z_index) label",
    "why_different_from_phase2": "the true partner is IN the pool; Phase 2 excluded it deliberately",
    "scoring": "patient_hit (same patient) and exact_hit (same slice); chance = 1/n",
    "learned_cost": "inverse-OT diagonal theta fitted on TRAINING patients' true pairs, box+simplex",
    "features": "frozen DINOv2 pool4 -> label-free PCA -> G2/G3, one shared encoder for both modalities",
}


class RecoveryError(RuntimeError):
    """CN: 恢复测试的前置条件失败。EN: a recovery-test precondition failed."""


# ---------------------------------------------------------------------------------------
# CN: 数据装载：池内保留**真实伙伴**（同一患者、同一 z），这是与 Phase 2 的根本区别。
# EN: data loading: the pool keeps the **true partner** (same patient, same z) -- the fundamental difference
#     from Phase 2.
# ---------------------------------------------------------------------------------------
def load_cached(paths: PatientPaths, *, cache: bool) -> SynthRadSlices:
    """CN: 读缓存或预处理并写缓存（CT 窗固定为 soft）。EN: cached read, else preprocess and store."""
    target = slices_path(paths.patient, size=IMAGE_SIZE, ct_window=CT_WINDOW,
                         mr_percentiles=MR_PERCENTILES, min_mask_area=MIN_MASK_AREA,
                         anatomy=paths.anatomy.lower())
    if cache and target.is_file():
        data = read_slices(target)
        return SynthRadSlices(patient=str(data["meta"]["patient"]), centre=str(data["meta"]["centre"]),
                              mr=np.asarray(data["mr"]), ct=np.asarray(data["ct"]),
                              mask=np.asarray(data["mask"]), z_normalised=np.asarray(data["z_normalised"]),
                              z_index=np.asarray(data["z_index"]), meta=dict(data["meta"]))
    slices = preprocess_patient(paths)
    if cache:
        save_slices(target, mr=slices.mr, ct=slices.ct, mask=slices.mask,
                    z_normalised=slices.z_normalised, z_index=slices.z_index,
                    patient=slices.patient, centre=slices.centre, meta=slices.meta)
    return slices


def take_slices(slices: SynthRadSlices, *, max_per_patient: int, min_gap: int, rng: np.random.Generator
                ) -> Int[np.ndarray, "picked"]:
    """CN: 每患者最多 `max_per_patient` 片、互隔 `min_gap`；返回行号。EN: up to N slices >= gap apart."""
    order = rng.permutation(len(slices))
    chosen: list[int] = []
    for idx in order:
        if all(abs(int(idx) - c) >= int(min_gap) for c in chosen):
            chosen.append(int(idx))
            if len(chosen) >= int(max_per_patient):
                break
    return np.sort(np.asarray(chosen, dtype=np.int64))


@dataclass(frozen=True)
class SliceTable:
    """
    CN: 评估用的切片表。**`mr_id` 与 `ct_id` 是同一个东西**：第 i 行的 MR 与第 i 行的 CT 是同一患者同一层。
        因此"真实配对"就是恒等映射，`id = (patient, z_index)`。
    EN: the evaluation table. **`mr_id` and `ct_id` are the same object**: row i's MR and row i's CT are the
        same patient at the same slice, so the true pairing is the identity and `id = (patient, z_index)`.
    """

    mr: Float[np.ndarray, "n pixels"]
    ct: Float[np.ndarray, "n pixels"]
    mask: Bool[np.ndarray, "n pixels"]
    z: Float[np.ndarray, "n"]
    patient: list[str]
    z_index: Int[np.ndarray, "n"]
    centre: list[str]

    def __len__(self) -> int:
        """CN: 切片数。EN: number of slices."""
        return int(self.mr.shape[0])


def build_table(patients: list[PatientPaths], *, cache: bool, max_per_patient: int, min_gap: int,
                seed: int) -> SliceTable:
    """CN: 把患者展开成一张表，MR 与 CT 行对齐（同一患者同一 z）。EN: flatten patients; MR/CT rows aligned."""
    rng = np.random.default_rng(int(seed))
    mr, ct, mask, z, pid, zidx, ctr = [], [], [], [], [], [], []
    for paths in patients:
        slices = load_cached(paths, cache=cache)
        for i in take_slices(slices, max_per_patient=max_per_patient, min_gap=min_gap, rng=rng):
            mr.append(slices.mr[i]); ct.append(slices.ct[i]); mask.append(slices.mask[i])
            z.append(float(slices.z_normalised[i])); zidx.append(int(slices.z_index[i]))
            pid.append(slices.patient); ctr.append(slices.centre)
    if not mr:
        raise RecoveryError("empty table")
    return SliceTable(mr=np.stack(mr), ct=np.stack(ct), mask=np.stack(mask),
                      z=np.asarray(z, dtype=np.float64), patient=pid,
                      z_index=np.asarray(zidx, dtype=np.int64), centre=ctr)


# ---------------------------------------------------------------------------------------
# CN: 计分：把方法与**标签**比，不与别的方法比。
# EN: scoring: compare the method to the **label**, never to another method.
# ---------------------------------------------------------------------------------------
def score_permutation(perm: Int[np.ndarray, "n"], *, patient_rows: list[str], patient_cols: list[str],
                      z_rows: Int[np.ndarray, "n"], z_cols: Int[np.ndarray, "n"]) -> dict[str, float]:
    """
    CN: 给定置换 `perm[i] = j`（第 i 行配到第 j 列），返回
        `patient_hit` = 同一患者的比例，`exact_hit` = 同一患者**且**同一层的比例。
    EN: given `perm[i] = j`, return `patient_hit` (same patient) and `exact_hit` (same patient **and** same z).
    """
    n = int(len(perm))
    same_patient = np.asarray([patient_rows[i] == patient_cols[int(perm[i])] for i in range(n)], dtype=bool)
    same_z = np.asarray([int(z_rows[i]) == int(z_cols[int(perm[i])]) for i in range(n)], dtype=bool)
    return {"patient_hit": float(same_patient.mean()), "exact_hit": float((same_patient & same_z).mean())}


def oracle_permutation(*, patient_rows: list[str], patient_cols: list[str],
                       z_rows: Int[np.ndarray, "n"], z_cols: Int[np.ndarray, "n"]) -> Int[np.ndarray, "n"]:
    """
    CN: **oracle**：直接用 id 构造正确置换。必须得到 1.0/1.0，否则 id 对齐本身有错。
    EN: **oracle**: build the correct permutation from the ids directly. It must score 1.0/1.0, otherwise the
        ids themselves are misaligned.
    """
    index: dict[tuple[str, int], int] = {}
    for j in range(len(patient_cols)):
        index[(patient_cols[j], int(z_cols[j]))] = j
    perm = np.empty(len(patient_rows), dtype=np.int64)
    for i in range(len(patient_rows)):
        key = (patient_rows[i], int(z_rows[i]))
        if key not in index:
            raise RecoveryError(f"true partner for {key} is not in the pool; ids are not aligned")
        perm[i] = index[key]
    return perm


def hungarian(cost: Float[np.ndarray, "n n"]) -> Int[np.ndarray, "n"]:
    """CN: 精确指派（方阵）。EN: exact assignment (square)."""
    from scipy.optimize import linear_sum_assignment  # noqa: PLC0415

    m = np.asarray(cost, dtype=np.float64)
    if m.shape[0] != m.shape[1]:
        raise RecoveryError(f"need a square cost, got {m.shape}")
    rows, cols = linear_sum_assignment(m)
    perm = np.empty(m.shape[0], dtype=np.int64)
    perm[rows] = cols
    return perm


def exact_ot(cost: Float[np.ndarray, "n n"]) -> Int[np.ndarray, "n"]:
    """CN: POT 精确 OT（均匀边际）；方阵时耦合即置换。EN: POT exact OT with uniform marginals."""
    m = np.asarray(cost, dtype=np.float64)
    if m.shape[0] != m.shape[1]:
        raise RecoveryError(f"need a square cost, got {m.shape}")
    w = np.ones(m.shape[0]) / m.shape[0]
    plan = ot.emd(w, w, m, numItermax=50_000_000)
    return np.argmax(plan, axis=1).astype(np.int64)


def nmi_cost(mr: Float[np.ndarray, "n pixels"], ct: Float[np.ndarray, "n pixels"], *, size: int = 128,
             bins: int = 32, block: int = 64) -> Float[np.ndarray, "n n"]:
    """CN: 像素互信息代价（负 MI，GPU 分块）。EN: pixel mutual-information cost (negative MI, GPU-batched)."""
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


def mask_iou_cost(mask_rows: Bool[np.ndarray, "n pixels"], mask_cols: Bool[np.ndarray, "n pixels"]
                  ) -> Float[np.ndarray, "n n"]:
    """CN: 掩膜 IoU 代价（`1 - IoU`，越小越好）。EN: mask IoU cost (`1 - IoU`, lower is better)."""
    a = np.asarray(mask_rows, dtype=np.float64); b = np.asarray(mask_cols, dtype=np.float64)
    inter = a @ b.T
    union = a.sum(axis=1)[:, None] + b.sum(axis=1)[None, :] - inter
    return 1.0 - inter / np.maximum(union, 1.0)


# ---------------------------------------------------------------------------------------
# CN: 特征：一个共享 encoder，label-free PCA，几何，逐行 L2。
# EN: features: one shared encoder, label-free PCA, geometry, row-wise L2.
# ---------------------------------------------------------------------------------------
def features(mr: Float[np.ndarray, "n pixels"], ct: Float[np.ndarray, "n pixels"], *, k: int,
             geometry_name: str, encoder: str) -> tuple[Float[np.ndarray, "n k"], Float[np.ndarray, "n k"]]:
    """
    CN: 两模态共享一个 encoder 得到特征，再套几何并逐行 L2。`dinov2` 或 `pixpca`。
    EN: one shared encoder for both modalities, then the geometry and row-wise L2. `dinov2` or `pixpca`.
    """
    if encoder == "dinov2":
        side = int(round(float(np.asarray(mr).shape[1]) ** 0.5))
        fm = encode_pool4(np.asarray(mr).reshape(-1, side, side), device="cuda").pooled.astype(np.float64)
        fc = encode_pool4(np.asarray(ct).reshape(-1, side, side), device="cuda").pooled.astype(np.float64)
        basis = pca_fit(np.concatenate([fm, fc], axis=0), k=int(k))
        proj_mr, proj_ct = pca_apply(fm, basis), pca_apply(fc, basis)
    elif encoder == "pixpca":
        pooled = np.concatenate([mr, ct], axis=0)
        mean = pooled.mean(axis=0)
        _u, _s, vt = np.linalg.svd(pooled - mean[None, :], full_matrices=False)
        kk = int(min(k, vt.shape[0]))
        b = vt[:kk].T
        proj_mr, proj_ct = (mr - mean) @ b, (ct - mean) @ b
    else:
        raise RecoveryError(f"unknown encoder {encoder!r}")
    stats = fit_geometry(proj_mr, proj_ct, str(geometry_name))
    out_mr, out_ct = apply_geometry(proj_mr, proj_ct, stats)
    return np.asarray(out_mr, dtype=np.float64), np.asarray(out_ct, dtype=np.float64)


def cosine_cost(a: Float[np.ndarray, "n k"], b: Float[np.ndarray, "n k"]) -> Float[np.ndarray, "n n"]:
    """CN: 余弦距离 `1 - <a,b>`。EN: cosine distance `1 - <a,b>`."""
    return 1.0 - np.asarray(a) @ np.asarray(b).T


# ---------------------------------------------------------------------------------------
# CN: **学到的代价**：在训练患者的**真实配对**上解逆 OT（盒子 ∩ 单纯形），再把 θ 用于留出患者。
# EN: **the learned cost**: solve inverse OT on the TRAINING patients' **true pairs** (box ∩ simplex), then
#     apply theta to held-out patients.
# ---------------------------------------------------------------------------------------
def fit_learned(mr: Float[np.ndarray, "n pixels"], ct: Float[np.ndarray, "n pixels"], *, k: int,
                geometry_name: str, encoder: str, solver: str, solve_limit_s: float,
                pairs_per_patient: int, seed: int) -> tuple[Float[np.ndarray, "k"], dict[str, object]]:
    """
    CN: 在训练患者上拟合对角 θ。**监督信号是真实配对**：把同一患者的多层切片排成一个块对角置换，
        使逆 OT 学到"同患者同层"的代价。θ 的作用空间与拟合空间由同一 `features()` 保证一致。
    EN: fit diagonal theta on training patients. The supervision is the **true pairing**: block the slices per
        patient into a block-diagonal permutation so inverse OT learns a cost that ranks same-patient same-slice
        first. `features()` is used for both fit and evaluation, so the space is identical by construction.
    """
    x, y = features(mr, ct, k=int(k), geometry_name=str(geometry_name), encoder=str(encoder))
    # CN: 用真实配对构造正边：这里直接取对角（第 i 片配第 i 片）作为训练计划。
    # EN: the training plan is the identity over aligned rows (slice i with slice i).
    fit = fit_full_lp(x, y, solver=solver, solve_limit_s=float(solve_limit_s))
    if fit.status != "ok":
        raise RecoveryError(f"learned fit failed: {fit.status}")
    theta = np.asarray(fit.theta, dtype=np.float64)
    if not np.all(np.isfinite(theta)) or abs(float(theta.sum()) - 1.0) > 1e-6:
        raise RecoveryError(f"learned theta is not a valid simplex point: sum={float(theta.sum())!r}")
    meta = {"R_prime": float(fit.R_prime), "R_uniform": float(fit.R_uniform),
            "R_ratio": float(fit.R_ratio), "n_pairs": int(x.shape[0]), "solver": str(solver)}
    return theta, meta


@dataclass(frozen=True)
class RecoveryConfig(RunConfig):
    """CN: 恢复测试的旋钮。EN: the recovery test's knobs."""

    section = "synthrad_recovery"
    study: str = "synthrad_recovery"
    probe_per_centre: int = 10
    split_seed: int = 7
    pool_patients: int = 20
    slices_per_patient: int = 5
    min_gap: int = 8
    ks: tuple[int, ...] = (128, 256)
    geometries: tuple[str, ...] = ("G2", "G3")
    encoders: tuple[str, ...] = ("dinov2", "pixpca")
    draw_seeds: tuple[int, ...] = (0, 1, 2)
    cache: bool = True

    @classmethod
    def add_study_arguments(cls, parser: argparse.ArgumentParser) -> None:
        """CN: 只加本研究特有的参数。EN: only this study's flags."""
        parser.add_argument("--probe-per-centre", type=int, default=10)
        parser.add_argument("--split-seed", type=int, default=7)
        parser.add_argument("--pool-patients", type=int, default=20,
                            help="patients in the evaluation pool; their true partners ARE in the pool")
        parser.add_argument("--slices-per-patient", type=int, default=5)
        parser.add_argument("--min-gap", type=int, default=8)
        parser.add_argument("--ks", type=int, nargs="+", default=[128, 256])
        parser.add_argument("--geometries", nargs="+", default=["G2", "G3"], choices=["G1", "G2", "G3"])
        parser.add_argument("--encoders", nargs="+", default=["dinov2", "pixpca"],
                            choices=["dinov2", "pixpca"])
        parser.add_argument("--draw-seeds", type=int, nargs="+", default=[0, 1, 2])
        parser.add_argument("--cache", action=argparse.BooleanOptionalAction, default=True)


def main(argv: list[str] | None = None) -> int:
    """CN: 入口。EN: entry point."""
    parser = build_parser(RecoveryConfig)
    args = parser.parse_args(list(argv) if argv is not None else None)
    config = RecoveryConfig.from_namespace(args, parser)

    log = RunLog(config.run_dir / "run.log")
    log.line(f"config: pool_patients={config.pool_patients} slices/patient={config.slices_per_patient} "
             f"ks={config.ks} geometries={config.geometries} encoders={config.encoders} "
             f"draws={config.draw_seeds}")

    manifest = config.run_dir.parent / "synthrad_probe" / "split_manifest.json"
    probe_ids: set[str] = set()
    if manifest.is_file():
        probe_ids = set(json.loads(manifest.read_text())["probe"])
        log.line(f"using the Phase 1 probe split ({len(probe_ids)} patients); test set untouched")
    patients = discover_patients(_REPO_ROOT / "data" / "Task1", anatomy="brain")
    if probe_ids:
        patients = [p for p in patients if p.patient in probe_ids]
    if len(patients) < config.pool_patients + 1:
        raise RecoveryError(f"need more than {config.pool_patients} probe patients, have {len(patients)}")

    # CN: **训练/评估按患者切分**：`learned` 的 θ 只在训练患者上拟合，评估在其余患者的池上。
    # EN: **split by patient**: the learned theta is fitted on training patients only and evaluated on a pool
    #     of different patients.
    rng = np.random.default_rng(int(config.split_seed))
    order = rng.permutation(len(patients))
    train_patients = [patients[i] for i in order[: len(patients) // 2]]
    eval_patients = [patients[i] for i in order[len(patients) // 2:]]
    log.line(f"train patients {len(train_patients)}, evaluation patients {len(eval_patients)}")

    records: list[RunRecord] = []
    learned_by_cell: dict[tuple[str, str, int], tuple[Float[np.ndarray, "k"], dict[str, object]]] = {}

    # CN: 每个配置先准备特征，再跑 3 个画次的池。
    # EN: for each configuration prepare features once, then run the 3 draws of pools.
    for encoder in config.encoders:
        for k in config.ks:
            for geometry_name in config.geometries:
                cell = f"{encoder}|{geometry_name}|K{k}"
                train_table = build_table(train_patients, cache=bool(config.cache),
                                          max_per_patient=int(config.slices_per_patient),
                                          min_gap=int(config.min_gap), seed=int(config.split_seed))
                t0 = time.perf_counter()
                theta, fit_meta = fit_learned(train_table.mr, train_table.ct, k=int(k),
                                              geometry_name=str(geometry_name), encoder=str(encoder),
                                              solver=str(config.solver),
                                              solve_limit_s=float(config.solve_limit_s),
                                              pairs_per_patient=int(config.slices_per_patient),
                                              seed=int(config.split_seed))
                log.line(f"[{cell}] learned theta fitted on {fit_meta['n_pairs']} training slices in "
                         f"{time.perf_counter() - t0:.1f}s; R'/R_uniform={fit_meta['R_ratio']:.3f}; "
                         f"theta range [{theta.min():.6f}, {theta.max():.6f}]")
                learned_by_cell[(encoder, geometry_name, int(k))] = (theta, fit_meta)

                for draw in config.draw_seeds:
                    pool = build_table(eval_patients[: int(config.pool_patients)],
                                       cache=bool(config.cache),
                                       max_per_patient=int(config.slices_per_patient),
                                       min_gap=int(config.min_gap),
                                       seed=int(config.split_seed) + int(draw) + 1)
                    n = len(pool)
                    log.line(f"[{cell} draw={draw}] pool: {n} slices from "
                             f"{len(set(pool.patient))} patients (chance patient_hit ~ "
                             f"{1.0 / max(len(set(pool.patient)), 1):.3f})")
                    feat_mr, feat_ct = features(pool.mr, pool.ct, k=int(k), geometry_name=str(geometry_name),
                                                encoder=str(encoder))

                    # CN: **列必须打乱**。表中第 i 行的 MR 与第 i 行的 CT 是同一层，若列保持原序，
                    #     对称代价（|dz|、掩膜 IoU）的对角线恒为零，配对就被平凡地解出——实测 z_only 与
                    #     mask_iou 因此都得到 1.0/1.0。这里用固定种子打乱列，使方法只能看内容。
                    # EN: **columns must be shuffled.** Row i's MR and row i's CT are the same slice, so with
                    #     columns in their original order a symmetric cost (|dz|, mask IoU) is zero on the
                    #     diagonal and the pairing is solved trivially -- measured: z_only and mask_iou both hit
                    #     1.0/1.0. Shuffle the columns with a fixed seed so a method sees only content.
                    col_order = np.random.default_rng(10_000 + int(draw)).permutation(n)
                    z_cols = pool.z[col_order]
                    mask_cols = pool.mask[col_order]
                    ct_cols = pool.ct[col_order]
                    feat_ct_shuf = feat_ct[col_order]
                    patient_cols = [pool.patient[i] for i in col_order]
                    zidx_cols = pool.z_index[col_order]

                    arms: dict[str, tuple[Int[np.ndarray, "n"], float]] = {}
                    t1 = time.perf_counter()
                    arms["random"] = (np.random.default_rng(int(draw)).permutation(n).astype(np.int64),
                                      time.perf_counter() - t1)
                    t1 = time.perf_counter()
                    arms["z_only"] = (hungarian(np.abs(pool.z[:, None] - z_cols[None, :])),
                                      time.perf_counter() - t1)
                    t1 = time.perf_counter()
                    arms["mask_iou"] = (hungarian(mask_iou_cost(pool.mask, mask_cols)),
                                        time.perf_counter() - t1)
                    t1 = time.perf_counter()
                    arms["nmi"] = (hungarian(nmi_cost(pool.mr, ct_cols)), time.perf_counter() - t1)
                    t1 = time.perf_counter()
                    arms["cosine_ot"] = (exact_ot(cosine_cost(feat_mr, feat_ct_shuf)),
                                         time.perf_counter() - t1)
                    t1 = time.perf_counter()
                    arms["learned"] = (exact_ot(diagonal_cost(feat_mr, feat_ct_shuf, theta)),
                                       time.perf_counter() - t1)
                    t1 = time.perf_counter()
                    arms["oracle"] = (oracle_permutation(patient_rows=pool.patient,
                                                         patient_cols=patient_cols,
                                                         z_rows=pool.z_index, z_cols=zidx_cols),
                                      time.perf_counter() - t1)

                    # CN: oracle 断言——id 对齐若错，这里会立刻暴露。
                    # EN: oracle assertion -- if the ids are misaligned this fires immediately.
                    oracle_score = score_permutation(arms["oracle"][0], patient_rows=pool.patient,
                                                     patient_cols=patient_cols, z_rows=pool.z_index,
                                                     z_cols=zidx_cols)
                    if oracle_score["patient_hit"] != 1.0 or oracle_score["exact_hit"] != 1.0:
                        raise RecoveryError(f"oracle must be exactly 1.0/1.0, got {oracle_score}")
                    log.line(f"[{cell} draw={draw}] oracle check passed (1.0/1.0)")

                    for arm, (perm, seconds) in arms.items():
                        metrics = score_permutation(perm, patient_rows=pool.patient,
                                                    patient_cols=patient_cols, z_rows=pool.z_index,
                                                    z_cols=zidx_cols)
                        records.append(RunRecord(study=str(config.study), geometry=cell, arm=arm,
                                                 k=int(k), m=int(config.pool_patients),
                                                 draw=int(draw), metrics=metrics,
                                                 config=config.to_dict(),
                                                 extra={"seconds": round(seconds, 3), "n": int(n),
                                                        "n_patients": int(len(set(pool.patient))),
                                                        "encoder": encoder, "geometry": geometry_name,
                                                        "learned_R_ratio": fit_meta["R_ratio"]}))
                        log.line(f"[{cell} draw={draw}] {arm:<10} patient_hit={metrics['patient_hit']:.4f} "
                                 f"exact_hit={metrics['exact_hit']:.4f} ({seconds:.2f}s)")

    rows = summarise(records, group=("geometry", "arm"), metrics=("patient_hit", "exact_hit"))
    log.line("\n" + format_table(rows, ("geometry", "arm", "n_draws", "patient_hit_mean",
                                        "patient_hit_std", "exact_hit_mean", "exact_hit_std")))
    write_records(config.run_dir / "records.json", records, provenance=PROVENANCE)
    log.line(f"wrote {config.run_dir / 'records.json'}")
    log.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
