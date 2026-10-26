"""
CN: **NMI 配对研究** —— 问题：互信息（NMI）能否作为配对代价，与**白化 DINOv2 余弦 + OT** 及**学到的代价**相比？

    CN: 协议（三臂完全相同，**真实配对被隐藏**）：
        * 候选池 = **全部 1000 个受试者**，T1 侧与 T2 侧都各自打乱，模型只看到两个无序集合；
        * 每个臂输出一个 1000×1000 的代价矩阵，用 Hungarian 与 EMD 各自解码；
        * 指标：`pair_acc` = 解码出的配对与真实配对一致的比例（chance = 1/1000 = 0.001）。

    CN: **NMI 的样本量问题（本研究的核心教训）**：互信息需要一个**样本集合**来估计联合分布。
        最初两版把"一个候选对的若干主成分"当作样本，结果在 n=1000 时**退化为随机**——
        因为 16 或 256 个成分不足以估计一个联合直方图（实测同一对与错配对的 MI 完全相同）。
        正确的构造是经典的**互信息图像配准**目标：以**同一候选对的像素**为样本，

            cost[i, j] = -( H(A_i) + H(B_j) - I( A_i ; B_j ) )

        其中 `A_i` 是 T1 第 i 幅图的像素，`B_j` 是 T2 第 j 幅图的像素，每对约 16k 个样本。
        代价用"负互信息"或"负联合概率"都可；这里用 `H + H - I`，量纲为 nats。

    CN: 三臂：
        | 臂 | 代价 |
        |---|---|
        | `nmi` | `H(A_i) + H(B_j) - I(A_i ; B_j)`（像素，等频分箱） |
        | `cosine_ot` | 白化 DINOv2 特征的白化余弦距离 |
        | `learned` | 学到的 θ 下的对角 L2² 代价（在 m 个标注对上拟合，标签不进入池） |

EN: **NMI pairing study** -- can mutual information serve as the pairing cost, versus **whitened DINOv2
    cosine + OT** and the **learned cost**?

    Protocol (identical across arms, **true partners hidden**): the pool is all 1000 subjects, the T1 and
    T2 sides are shuffled independently, each arm emits a 1000x1000 cost, and Hungarian and EMD decode it.
    `pair_acc` is the fraction of decoded pairs that are correct (chance = 1/1000).

    **The sample-size lesson.** MI needs a *set of samples* to estimate a joint distribution. The first two
    constructions treated "the components of one candidate pair" as samples and collapsed to chance at
    n=1000: 16 or 256 components cannot estimate a joint histogram (measured: the MI of a true pair and of
    a false pair were *identical*). The defensible construction is the classical mutual-information image
    registration objective, which uses the **pixels of one candidate pair** as samples:

        cost[i, j] = -( H(A_i) + H(B_j) - I( A_i ; B_j ) )

    with `A_i` the pixels of T1 image i and `B_j` those of T2 image j, about 16k samples per pair.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import ot
from jaxtyping import Float, Int
from scipy.optimize import linear_sum_assignment

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from iot_experiments.harness.config import RunConfig, build_parser  # noqa: E402
from iot_experiments.harness.driver import make_split  # noqa: E402
from iot_experiments.harness.records import RunRecord, format_table, summarise, write_records  # noqa: E402
from iot_experiments.harness.registry import apply_geometry, fit_geometry, geometry  # noqa: E402
from iot_experiments.harness.solver import diagonal_cost, fit_full_lp  # noqa: E402

PROVENANCE: dict[str, object] = {
    "whitening_fit_set": "all T1+T2 images (label-free, no pairing information)",
    "nmi_estimator": "joint pixel histogram of the candidate pair, equal-frequency bins per image",
    "pool": "all 1000 subjects, T1 and T2 sides shuffled independently (true partners hidden)",
    "chance_pair_acc": 1.0 / 1000.0,
    "note": "no pairing or label information enters any arm; NMI sees only the two images of a pair",
}


class NmiError(RuntimeError):
    """CN: NMI 读数的前置条件失败。EN: a precondition for the NMI read-out failed."""


def whiten(x: Float[np.ndarray, "subjects d"], y: Float[np.ndarray, "subjects d"],
           *, dim: int) -> tuple[Float[np.ndarray, "subjects K"], Float[np.ndarray, "subjects K"]]:
    """
    CN: **PCA 白化**（在全部 T1+T2 上拟合，无配对信息），保留前 `dim` 个主成分并各自除以标准差。
    EN: **PCA whitening** fitted on all T1+T2 (no pairing information); the first `dim` components are
        each divided by their own standard deviation.
    """
    xa: Float[np.ndarray, "subjects d"] = np.asarray(x, dtype=np.float64)
    ya: Float[np.ndarray, "subjects d"] = np.asarray(y, dtype=np.float64)
    pooled: Float[np.ndarray, "pooled d"] = np.concatenate([xa, ya], axis=0)
    mean: Float[np.ndarray, "d"] = pooled.mean(axis=0)
    centred: Float[np.ndarray, "pooled d"] = pooled - mean
    _u, s, vt = np.linalg.svd(centred, full_matrices=False)
    k: int = int(min(dim, s.size))
    components: Float[np.ndarray, "K d"] = vt[:k]
    scale: Float[np.ndarray, "K"] = s[:k] / np.sqrt(float(centred.shape[0]))
    scale = np.where(scale > 0.0, scale, 1.0)

    def project(a: Float[np.ndarray, "subjects d"]) -> Float[np.ndarray, "subjects K"]:
        return ((a - mean) @ components.T) / scale

    return project(xa), project(ya)


def quantile_codes(values: Float[np.ndarray, "pixels"], *, bins: int
                   ) -> tuple[Int[np.ndarray, "pixels"], float, Float[np.ndarray, "pixels"]]:
    """
    CN: 等频分箱。返回 `(codes, base, levels)`：把 `bins` 个箱压到 `bins-1` 个不同的整数码
        （等频箱共享边界，故跨两图**共用同一组水平** `levels`），以保证 T1 与 T2 的联合直方图可加。
    EN: equal-frequency binning. Returns `(codes, base, levels)`: the `bins` quantile bins are mapped to
        `bins - 1` shared integer levels so that the T1 and T2 joint histogram uses the **same** supports.
    """
    a: Float[np.ndarray, "pixels"] = np.asarray(values, dtype=np.float64)
    if float(a.max()) - float(a.min()) < 1e-15:
        levels = np.zeros(1, dtype=np.float64)
        return np.zeros(a.size, dtype=np.int64), 0.0, levels
    edges: Float[np.ndarray, "edges"] = np.quantile(a, np.linspace(0.0, 1.0, int(bins) + 1)[1:-1])
    levels: Float[np.ndarray, "levels"] = np.concatenate([[float(a.min()) - 1.0], edges])
    codes: Int[np.ndarray, "pixels"] = np.searchsorted(edges, a, side="right").astype(np.int64)
    return codes, float(levels[0]), levels


def nmi_pixel_cost(images_a: Float[np.ndarray, "n pixels"],
                   images_b: Float[np.ndarray, "n pixels"], *,
                   bins: int = 32) -> Float[np.ndarray, "n n"]:
    """
    CN: **像素级 NMI 代价** `H(A_i) + H(B_j) - I(A_i; B_j)`（nats），每对用**两幅图的像素**作样本。

        CN: 计算方式：先对每个受试者做等频分箱，得到 `codes_a[i]`、`codes_b[j]`；再对每对 `(i, j)`
            用 `bincount` 在 `codes_a[i] * bins + codes_b[j]` 上建联合直方图（`n^2` 次，
            每次 `O(pixels)`）。边际熵由每幅图自己的直方图给出，只需 `O(n)` 次。
            这是与尺寸无关的 `O(n^2 * pixels)`，n=1000、16k 像素约 1.6e10 次累加，分块后在 GPU 上可行。

    EN: **Pixel-level NMI cost** `H(A_i) + H(B_j) - I(A_i; B_j)` in nats, with **the pixels of the two
        images** as the samples of each candidate pair.

        Each subject is quantile-binned once; for every pair `(i, j)` the joint histogram is a `bincount`
        over `codes_a[i] * bins + codes_b[j]`. Marginal entropies come from each image's own histogram,
        needing only `O(n)` histograms. The cost is `O(n^2 * pixels)`, about 1.6e10 accumulations at
        n=1000 with 16k pixels, so it is computed in row blocks.
    """
    a: Float[np.ndarray, "n pixels"] = np.asarray(images_a, dtype=np.float64)
    b: Float[np.ndarray, "n pixels"] = np.asarray(images_b, dtype=np.float64)
    if a.shape[1] != b.shape[1]:
        raise NmiError(f"pixel counts differ: {a.shape[1]} vs {b.shape[1]}")
    n: int = int(a.shape[0])
    m: int = int(bins)

    # CN: 每幅图各自等频分箱到 [0, bins-1]；两图箱边界不同，但共用整数码，足够建联合直方图。
    # EN: each image is quantile-binned to [0, bins-1]; the edges differ per image but the integer codes
    #     share a support, which is all the joint histogram needs.
    codes_a: Int[np.ndarray, "n pixels"] = np.stack(
        [quantile_codes(a[i], bins=m)[0] for i in range(n)], axis=0)
    codes_b: Int[np.ndarray, "n pixels"] = np.stack(
        [quantile_codes(b[j], bins=m)[0] for j in range(n)], axis=0)

    def entropy(codes: Int[np.ndarray, "n pixels"]) -> Float[np.ndarray, "n"]:
        counts: Float[np.ndarray, "n bins"] = np.stack(
            [np.bincount(codes[i], minlength=m) for i in range(codes.shape[0])]).astype(np.float64)
        p: Float[np.ndarray, "n bins"] = counts / float(codes.shape[1])
        with np.errstate(divide="ignore", invalid="ignore"):
            terms = np.where(p > 0.0, -p * np.log(np.maximum(p, 1e-300)), 0.0)
        return terms.sum(axis=1)

    h_a: Float[np.ndarray, "n"] = entropy(codes_a)
    h_b: Float[np.ndarray, "n"] = entropy(codes_b)

    cost: Float[np.ndarray, "n n"] = np.zeros((n, n), dtype=np.float64)
    flat_b: Int[np.ndarray, "n pixels"] = codes_b.astype(np.int64)
    pixels: int = int(codes_a.shape[1])
    block: int = 64
    for start in range(0, n, block):
        stop: int = min(start + block, n)
        chunk: Int[np.ndarray, "chunk pixels"] = codes_a[start:stop].astype(np.int64)
        for row_index in range(stop - start):
            # CN: 对固定 i，逐个 j 建联合直方图：index = codes_a[i] * bins + codes_b[j]。
            # EN: for fixed i, build the joint histogram per j: index = codes_a[i] * bins + codes_b[j].
            base_a: Int[np.ndarray, "pixels"] = chunk[row_index] * m
            joint: Float[np.ndarray, "n bins bins"] = np.stack(
                [np.bincount(base_a + flat_b[j], minlength=m * m) for j in range(n)]
            ).astype(np.float64) / float(pixels)
            joint = joint.reshape(n, m, m)
            p_a: Float[np.ndarray, "n bins bins"] = joint.sum(axis=2)
            p_b: Float[np.ndarray, "n bins bins"] = joint.sum(axis=1)
            with np.errstate(divide="ignore", invalid="ignore"):
                ratio = joint / (p_a[..., :, None] * p_b[..., None, :] + 1e-300)
                pointwise = np.where(joint > 0.0,
                                     joint * np.log(np.maximum(ratio, 1e-300)), 0.0)
            mutual: Float[np.ndarray, "n"] = pointwise.sum(axis=(1, 2))
            cost[start + row_index] = h_a[start + row_index] + h_b - mutual
    return cost


def pair_accuracy(matched: Int[np.ndarray, "n"], true_partner: Int[np.ndarray, "n"]) -> float:
    """
    CN: 解码出的配对 `matched[i]`（T1 行 i 配到的 T2 **列**）与真实配对一致的比例。
        **不能用** `matched[i] == i`：两侧都打乱过，恒等映射没有意义。chance = 1/n。
    EN: fraction of rows where the decoded partner (the T2 **column** matched to T1 row i) is the true one.
        **Not** `matched[i] == i`: both sides were shuffled, so the identity map is meaningless.
    """
    a: Int[np.ndarray, "n"] = np.asarray(matched, dtype=np.int64)
    b: Int[np.ndarray, "n"] = np.asarray(true_partner, dtype=np.int64)
    if a.shape != b.shape:
        raise NmiError(f"shape mismatch: matched {a.shape} vs true_partner {b.shape}")
    return float(np.mean(a == b))


def hungarian(cost: Float[np.ndarray, "n n"]) -> Int[np.ndarray, "n"]:
    """CN: Hungarian 指派，返回 `perm[i] = j`。EN: Hungarian assignment, returning `perm[i] = j`."""
    rows, cols = linear_sum_assignment(np.asarray(cost, dtype=np.float64))
    if not np.array_equal(rows, np.arange(rows.size)):
        raise NmiError("linear_sum_assignment did not return rows in order")
    return cols.astype(np.int64)


def emd_perm(cost: Float[np.ndarray, "n n"], *, num_iter_max: int = 50_000_000) -> Int[np.ndarray, "n"]:
    """
    CN: 均匀边际下的精确 EMD；n=n 且边际 1/n 时耦合即置换。
    EN: exact EMD under uniform marginals; with n = n and 1/n marginals the coupling is a permutation.
    """
    values: Float[np.ndarray, "n n"] = np.asarray(cost, dtype=np.float64)
    h: int = int(values.shape[0])
    if values.shape[1] != h:
        raise NmiError(f"expected a square matrix, got {values.shape}")
    plan: Float[np.ndarray, "n n"] = ot.emd(np.ones(h) / h, np.ones(h) / h, values, numItermax=int(num_iter_max))
    return np.argmax(plan, axis=1).astype(np.int64)


def load_pixels(*, size: int = 32) -> tuple[Float[np.ndarray, "n pixels"], Float[np.ndarray, "n pixels"]]:
    """
    CN: 读 T1/T2 灰度图并缩放到 `size x size`，返回两个 `(n, size*size)` 数组（按受试者 id 升序）。
        **只用像素**，不使用任何特征或标签。
    EN: Load the T1/T2 grey images resized to `size x size` as two `(n, size*size)` arrays in subject-id
        order. **Pixels only**: no features and no labels.
    """
    from iot_experiments.harness.data import image_files, load_gray  # noqa: PLC0415

    out: dict[str, Float[np.ndarray, "n pixels"]] = {}
    for modality in ("T1", "T2"):
        files = image_files(modality)
        out[modality] = np.stack([np.asarray(load_gray(p, size=int(size)), dtype=np.float64).ravel()
                                  for p in files], axis=0)
    return out["T1"], out["T2"]


@dataclass(frozen=True)
class NmiConfig(RunConfig):
    """CN: NMI 研究自己的旋钮。EN: the NMI study's own knobs."""

    section = "nmi"
    study: str = "nmi"
    feature: str = "dinov2-pool4"
    ks: tuple[int, ...] = (256,)
    ms: tuple[int, ...] = (300,)
    arms: tuple[str, ...] = ("nmi", "cosine_ot", "learned")
    bins: int = 32
    pixel_size: int = 32
    fitted_geometry: str = "G2"

    @classmethod
    def add_study_arguments(cls, parser: argparse.ArgumentParser) -> None:
        """CN: 只加本研究特有的参数。EN: only this study's flags."""
        parser.add_argument("--feature", default="dinov2-pool4", choices=["dinov2-pool4"])
        parser.add_argument("--ks", type=int, nargs="+", default=[256])
        parser.add_argument("--ms", type=int, nargs="+", default=[300],
                            help="labelled pairs m used to fit the learned arm")
        parser.add_argument("--arms", nargs="+", default=["nmi", "cosine_ot", "learned"],
                            choices=["nmi", "cosine_ot", "learned"])
        parser.add_argument("--bins", type=int, default=32, help="equal-frequency bins for the NMI histogram")
        parser.add_argument("--pixel-size", type=int, default=32,
                            help="images are resized to this square before the MI histogram")
        parser.add_argument("--fitted-geometry", default="G2", choices=["G0", "G1", "G2", "G3"],
                            help="geometry theta is fitted AND applied in (must be the same space)")


def main(argv: list[str] | None = None) -> int:
    """CN: 入口。EN: entry point."""
    parser = build_parser(NmiConfig)
    args = parser.parse_args(list(argv) if argv is not None else None)
    config = NmiConfig.from_namespace(args, parser)

    from iot_experiments.harness.registry import features  # noqa: PLC0415

    records: list[RunRecord] = []
    for k in config.ks:
        x, y = features(str(config.feature), dim=int(k))
        n_subjects: int = int(x.shape[0])
        # CN: 池 = 全部受试者；两侧各自打乱，真实配对在任何臂中都不可见。
        # EN: the pool is all subjects with each side shuffled; the true pairing is invisible to every arm.
        order_t1: Int[np.ndarray, "n"] = np.random.default_rng(0).permutation(n_subjects)
        order_t2: Int[np.ndarray, "n"] = np.random.default_rng(1).permutation(n_subjects)
        inv_t2: Int[np.ndarray, "n"] = np.empty(n_subjects, dtype=np.int64)
        inv_t2[order_t2] = np.arange(n_subjects, dtype=np.int64)
        pos_of_true: Int[np.ndarray, "n"] = inv_t2[order_t1]

        xw, yw = whiten(x, y, dim=int(k))

        def unit(a: Float[np.ndarray, "n K"]) -> Float[np.ndarray, "n K"]:
            norms = np.linalg.norm(a, axis=1, keepdims=True)
            return a / np.where(norms > 0.0, norms, 1.0)

        xu_s: Float[np.ndarray, "n K"] = unit(xw[order_t1])
        yu_s: Float[np.ndarray, "n K"] = unit(yw[order_t2])

        if "nmi" in config.arms:
            px1, px2 = load_pixels(size=int(config.pixel_size))
            px1_s: Float[np.ndarray, "n pixels"] = px1[order_t1]
            px2_s: Float[np.ndarray, "n pixels"] = px2[order_t2]
            cost = nmi_pixel_cost(px1_s, px2_s, bins=int(config.bins))
            perm_h = hungarian(cost)
            perm_e = emd_perm(cost)
            metrics = {"pair_acc_hungarian": pair_accuracy(perm_h, pos_of_true),
                       "pair_acc_emd": pair_accuracy(perm_e, pos_of_true)}
            true_costs = cost[np.arange(n_subjects), pos_of_true]
            off = (float(cost.sum()) - float(true_costs.sum())) / float(n_subjects * (n_subjects - 1))
            records.append(RunRecord(study=str(config.study), geometry="pixels", arm="nmi",
                                     k=int(k), m=0, draw=-1, metrics=metrics,
                                     config=config.to_dict(),
                                     extra={"true_pair_cost": float(true_costs.mean()),
                                            "offdiag_cost": off}))
            print(f"  K={k} pixel NMI hungarian acc={metrics['pair_acc_hungarian']:.4f} "
                  f"emd acc={metrics['pair_acc_emd']:.4f} "
                  f"(true {true_costs.mean():.4f} vs off {off:.4f})", flush=True)

        for m in config.ms:
            for draw in config.draws:
                split = make_split(n_subjects, int(m), draw=int(draw))

                if "cosine_ot" in config.arms:
                    # CN: 白化余弦分；Hungarian 与 EMD 用同一矩阵（EMD 取平方距离，OT 的标准代价）。
                    # EN: whitened cosine distance; Hungarian and EMD share it (EMD uses the square).
                    cos = xu_s @ yu_s.T
                    cost = 1.0 - cos
                    perm_h = hungarian(cost)
                    perm_e = emd_perm(cost * cost)
                    metrics = {"pair_acc_hungarian": pair_accuracy(perm_h, pos_of_true),
                               "pair_acc_emd": pair_accuracy(perm_e, pos_of_true)}
                    records.append(RunRecord(study=str(config.study), geometry="whiten", arm="cosine_ot",
                                             k=int(k), m=int(m), draw=int(draw), metrics=metrics,
                                             config=config.to_dict()))
                    print(f"  K={k} m={m} d={draw} cosine+OT hungarian acc={metrics['pair_acc_hungarian']:.4f} "
                          f"emd acc={metrics['pair_acc_emd']:.4f}", flush=True)

                if "learned" in config.arms:
                    geo = geometry(x, y, str(config.fitted_geometry),
                                   labelled=split.labelled, unlabelled=split.unlabelled)
                    fit = fit_full_lp(geo.x_label, geo.y_label, solver=str(config.solver),
                                      solve_limit_s=float(config.solve_limit_s))
                    if fit.status != "ok":
                        records.append(RunRecord(study=str(config.study),
                                                 geometry=str(config.fitted_geometry), arm="learned",
                                                 k=int(k), m=int(m), draw=int(draw), metrics={},
                                                 config=config.to_dict(), excluded=True, status=fit.status))
                        print(f"  K={k} m={m} d={draw} learned EXCLUDED: {fit.status}", flush=True)
                        continue
                    # CN: **θ 必须在它被拟合的那个空间里使用**。早先的版本在 G2（标准化+L2）上拟合，
                    #     却把代价算在原始未白化空间上，两者尺度不同，θ 无意义（实测 0.549 vs 0.863）。
                    #     这里把同一个几何施加到**整个池**上，再套用 θ。
                    # EN: **theta must be used in the space it was fitted in.** An earlier version fitted on
                    #     G2 (standardised + L2) but evaluated the cost in the raw unwhitened space; the
                    #     scales differ, so theta is meaningless there (measured 0.549 vs 0.863). Here the
                    #     same geometry is applied to the **whole pool**, then theta is applied.
                    stats = fit_geometry(x[split.unlabelled], y[split.unlabelled],
                                         str(config.fitted_geometry))
                    xg, yg = apply_geometry(x[order_t1], y[order_t2], stats)
                    # CN: **断言**评分用的几何与拟合用的几何是同一个，且数据确实落在该几何的单位球上。
                    #     仅靠注释无法阻止这个 bug 复发：此前正是"G2 拟合、原始空间计分"。
                    # EN: **assert** that the geometry used for scoring is the one theta was fitted in, and
                    #     that the data really lies on that geometry's unit sphere. A comment cannot stop
                    #     this bug from returning; the original defect was "fit in G2, score in raw space".
                    apply_geometry_stats = stats.name
                    if apply_geometry_stats != str(config.fitted_geometry):
                        raise SystemExit(
                            f"theta was fitted in {config.fitted_geometry} but applied in "
                            f"{apply_geometry_stats}; refusing to score")
                    fitted_norms = np.linalg.norm(np.asarray(geo.x_label, dtype=np.float64), axis=1)
                    scored_norms = np.linalg.norm(np.asarray(xg, dtype=np.float64), axis=1)
                    if not (np.allclose(fitted_norms, 1.0, atol=1e-8)
                            and np.allclose(scored_norms, 1.0, atol=1e-8)):
                        raise SystemExit(
                            "fit and score spaces differ: fitted norms "
                            f"[{fitted_norms.min():.6f},{fitted_norms.max():.6f}] vs scored norms "
                            f"[{scored_norms.min():.6f},{scored_norms.max():.6f}]")
                    cost = diagonal_cost(xg, yg, fit.theta)
                    perm_h = hungarian(cost)
                    perm_e = emd_perm(cost)
                    metrics = {"pair_acc_hungarian": pair_accuracy(perm_h, pos_of_true),
                               "pair_acc_emd": pair_accuracy(perm_e, pos_of_true)}
                    records.append(RunRecord(study=str(config.study),
                                             geometry=str(config.fitted_geometry), arm="learned",
                                             k=int(k), m=int(m), draw=int(draw), metrics=metrics,
                                             config=config.to_dict(),
                                             extra={"R_prime": fit.R_prime, "R_uniform": fit.R_uniform,
                                                    "theta_space": str(config.fitted_geometry),
                                                    "space_asserted": True}))
                    print(f"  K={k} m={m} d={draw} learned ({config.fitted_geometry}) "
                          f"hungarian acc={metrics['pair_acc_hungarian']:.4f} "
                          f"emd acc={metrics['pair_acc_emd']:.4f}", flush=True)

    rows = summarise(records, group=("geometry", "arm"),
                     metrics=("pair_acc_hungarian", "pair_acc_emd"))
    print("\n" + format_table(rows, ("geometry", "arm", "n_draws",
                                     "pair_acc_hungarian_mean", "pair_acc_hungarian_std",
                                     "pair_acc_emd_mean", "pair_acc_emd_std")))
    print(f"\n# chance pair_acc = 1/{n_subjects} = {1.0 / n_subjects:.4f}")
    write_records(config.run_dir / "records.json", records, provenance=PROVENANCE)
    print(f"\n[nmi] wrote {config.run_dir / 'records.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
