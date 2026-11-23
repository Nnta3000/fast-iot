"""
EN: **FAUST's own protocol: ordered scan-to-scan pairs, two challenges.**

    The dataset README is explicit (MPI-FAUST, CVPR 2014): the task is *scan-to-scan* correspondence over a
    designated pair list, split into **intra-subject** (same person, different pose) and **inter-subject**
    (different people) challenges, scored separately. Each pair is **ordered**: for every vertex of the *first*
    shape you give a point on the *second*, and the error is measured on the second. There is no template and no
    symmetric pair.

    Our proxy: the 100 training **registrations** share one vertex ordering, so the ground truth is the identity
    and both normalisations can be computed exactly. That is the only deviation, and it is a real one -- FAUST
    scores *scans* (different tessellation, holes), which needs the withheld test registrations and the
    submission server. Anything this driver reports is therefore a **registration proxy**, never a leaderboard
    number.

    Two deliberate protocol choices, both fixing defects of the earlier pairwise proxy:

    1. **No transductive standardisation.** Descriptor statistics are fitted on the *training* shapes only
       (`DescriptorScaler`) and applied unchanged to the test shapes. The old `standardise(a, b)` pooled both
       sides of the evaluated pair, so the target's statistics entered the feature map.
    2. **Two normalisations, reported together.** FAUST's own metric divides the geodesic error by the shape's
       **diameter** (x100); the study's earlier scale was `sqrt(surface area)`. They differ by ~25 % on FAUST, so
       a table may contain only one of them. The diameter is the headline.

    Split discipline: θ is fitted on pairs drawn from **training subjects** and evaluated on pairs drawn from
    **test subjects**; the two sets are disjoint by assertion, not by convention. Poses are crossed so that both
    seen and unseen poses appear on the test side.

CN: **FAUST 自身的协议：有向的 scan-to-scan 配对，两个 challenge。** 数据集 README 写得很清楚：任务是**扫描到扫描**
    的对应，按指定配对表分为**同受试者**（同一人、不同姿态）与**跨受试者**（不同人）两个 challenge，分别评分。
    配对是**有向的**：给出第一个形状每个顶点在第二个上的点，误差在第二个上度量。没有模板，也没有对称配对。

    我们的代理：100 个训练**配准**共享同一顶点序，故真值是恒等映射，两种归一化都能精确计算。这是唯一偏差，而且
    是实质偏差——FAUST 评的是**扫描**（不同网格、有洞），需要被保留的测试配准与提交服务器。因此本 driver 的任何
    数字都是**配准代理**，绝不是排行榜数字。

    两个刻意的协议选择，都是修掉早先成对代理的缺陷：(1) **不做传导式标准化**——描述子统计只在**训练**形状上拟合，
    原样应用到测试形状；旧的 `standardise(a,b)` 把评估对两侧合并，测试侧统计进入了特征。(2) **同时报告两种归一化**
    ——FAUST 自身用**直径**（乘 100），研究早先用 `sqrt(表面积)`，在 FAUST 上相差约 25%，同一张表里只能放一种；直径
    是头条。划分纪律：θ 在**训练受试者**的对上拟合、在**测试受试者**的对上评估，两者由断言保证不交。
"""

from __future__ import annotations

import dataclasses
import json
import sys
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sparse
from jaxtyping import Float, Int

_REPO_ROOT = Path(__file__).resolve().parents[4]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from iot.core.parameters import (  # noqa: E402
    PruningParameters,
    RelaxedObjective,
    SolverParameters,
)
from iot.core.problem import InverseOTProblem  # noqa: E402
from iot.core.theta import SimplexTheta  # noqa: E402
from iot.relaxed.loop_regret import solve_inverse_ot_regret  # noqa: E402
from iot_experiments.harness.config import RunConfig, build_parser  # noqa: E402
from iot_experiments.harness.faust import (  # noqa: E402
    DescriptorScaler,
    default_registrations_root,
    diameter,
    geodesic_matrix,
    read_ply,
    shared_index,
    surface_area,
)
from iot_experiments.harness.faust_match import geodesic_errors, match  # noqa: E402
from iot_experiments.harness.master_probe import RecordingMaster  # noqa: E402
from iot_experiments.harness.preflight import check as preflight_check  # noqa: E402
from iot_experiments.harness.records import (  # noqa: E402
    RunLog,
    RunRecord,
    format_table,
    summarise,
    write_records,
)
from iot_experiments.studies.iot.real.faust.correspondence import descriptors, fit_theta  # noqa: E402
from iot_experiments.studies.iot.real.faust.joint import (  # noqa: E402
    PairProblem,
    _cost_rows,
    _family,
    _flatness,
    _solve_multi,
    _solve_shared_margin,
    _uniform_regret,
)

PROVENANCE: dict[str, object] = {
    "protocol": "FAUST scan-to-scan correspondence, ordered pairs, intra- and inter-subject challenges scored "
                "separately (dataset README, MPI-FAUST CVPR 2014)",
    "proxy": "REGISTRATION proxy: the 100 training registrations share a vertex ordering, so the ground truth is "
             "the identity. FAUST scores scans, which this driver cannot do (test registrations withheld).",
    "direction": "first -> second; the error is measured on the second shape",
    "normalisation": "diameter (FAUST's own, headline) AND sqrt(surface area) (the study's earlier scale); never "
                     "mixed in one table",
    "standardisation": "fitted on TRAINING shapes only (DescriptorScaler), applied unchanged to test shapes; the "
                       "old pooled standardise(a, b) leaked target statistics into the feature map",
    "split": "theta fitted on training-subject pairs, evaluated on test-subject pairs; disjointness asserted",
    "theta_sources": "uniform | shared (regret LP over training pairs) | shared_margin (soft-margin LP) | oracle "
                     "(fitted on the evaluated pair itself -- an upper bound, never a method result)",
}


def _poses(count: int = 10) -> tuple[int, ...]:
    """EN: FAUST training registrations are `10*subject + pose`. CN: 训练配准为 `10*subject + pose`。"""
    return tuple(range(int(count)))


def intra_pairs(subjects: tuple[int, ...], per_subject: int, *, seed: int) -> list[tuple[int, int]]:
    """
    EN: ordered **same-subject, different-pose** pairs, `per_subject` per subject. Deterministic given `seed`.

        Ordered because FAUST's submission is directional: the first shape supplies the vertices whose images are
        asked for, and the error is measured on the second.

    CN: 有序的**同受试者、不同姿态**配对，每个受试者 `per_subject` 对；给定 `seed` 后确定。方向性来自 FAUST 的提交
        格式：第一个形状提供顶点，误差在第二个上度量。
    """
    rng = np.random.default_rng(int(seed))
    poses = _poses()
    out: list[tuple[int, int]] = []
    for subject in subjects:
        candidates = [(p, q) for p in poses for q in poses if p != q]
        picked = rng.choice(len(candidates), size=int(per_subject), replace=False)
        out.extend((10 * int(subject) + candidates[index][0], 10 * int(subject) + candidates[index][1])
                   for index in picked)
    return out


def inter_pairs(subjects: tuple[int, ...], per_subject: int, *, seed: int) -> list[tuple[int, int]]:
    """
    EN: ordered **different-subject** pairs, `per_subject` per subject: for each subject, pick a different subject
        and a pose for each side. Coverage is balanced by construction (every subject contributes the same
        number), which is what the earlier hand-typed lists got wrong.

    CN: 有序的**跨受试者**配对，每个受试者 `per_subject` 对：为每个受试者选另一个受试者，两侧各选一个姿态。
        构造上覆盖均衡（每个受试者贡献相同数量）——这正是早先手写列表没做到的。
    """
    rng = np.random.default_rng(int(seed) + 1)
    poses = _poses()
    ordered = [int(v) for v in subjects]
    out: list[tuple[int, int]] = []
    for subject in ordered:
        others = [v for v in ordered if v != subject]
        for _ in range(int(per_subject)):
            partner = int(rng.choice(others))
            pose_a = int(rng.integers(0, len(poses)))
            pose_b = int(rng.integers(0, len(poses)))
            out.append((10 * int(subject) + pose_a, 10 * partner + pose_b))
    return out


@dataclasses.dataclass(frozen=True)
class ChallengeConfig(RunConfig):
    """EN: knobs for the protocol driver. CN: 协议 driver 的旋钮。"""

    study: str = "faust_challenge"
    section: str = "faust_challenge"
    root: str = dataclasses.field(default_factory=lambda: str(default_registrations_root()))
    n_sub: int = 800
    seeds: tuple[int, ...] = (0,)
    train_subjects: tuple[int, ...] = (0, 1, 2, 3, 4)
    test_subjects: tuple[int, ...] = (5, 6, 7, 8, 9)
    intra_per_subject: int = 6      # EN: 6 x 10 subjects = 60, FAUST's intra count. CN: 6x10=60，对齐 FAUST。
    inter_per_subject: int = 4      # EN: 4 x 10 subjects = 40, FAUST's inter count. CN: 4x10=40。
    max_iterations: int = 5
    family: str = "diag13"
    bands: int = 5
    basis_size: int = 8
    with_margin: bool = True
    per_pair_backend: str = "highs"

    def __post_init__(self) -> None:
        """EN: normalise and validate. CN: 归一化与校验。"""
        for name in ("seeds", "train_subjects", "test_subjects"):
            object.__setattr__(self, name, tuple(int(v) for v in getattr(self, name)))
        overlap = set(self.train_subjects) & set(self.test_subjects)
        if overlap:
            raise ValueError(f"train and test subjects overlap: {sorted(overlap)}")
        if int(self.n_sub) < 2:
            raise ValueError("--n-sub must be at least 2")
        if int(self.intra_per_subject) < 1 or int(self.inter_per_subject) < 1:
            raise ValueError("per-subject pair counts must be at least 1")

    @classmethod
    def add_study_arguments(cls, parser) -> None:
        """EN: this study's flags only. CN: 只加本研究的参数。"""
        parser.add_argument("--root", default=str(default_registrations_root()))
        parser.add_argument("--n-sub", type=int, default=800)
        parser.add_argument("--seeds", type=int, nargs="+", default=[0])
        parser.add_argument("--train-subjects", type=int, nargs="+", default=[0, 1, 2, 3, 4])
        parser.add_argument("--test-subjects", type=int, nargs="+", default=[5, 6, 7, 8, 9])
        parser.add_argument("--intra-per-subject", type=int, default=6)
        parser.add_argument("--inter-per-subject", type=int, default=4)
        parser.add_argument("--max-iterations", type=int, default=5)
        parser.add_argument("--family", default="diag13",
                            choices=["diag13", "hinge8", "profile8", "group+diag", "group_gauss",
                                     "gauss", "gauss+cos"])
        parser.add_argument("--bands", type=int, default=5)
        parser.add_argument("--basis-size", type=int, default=8)
        parser.add_argument("--with-margin", type=int, default=1)
        parser.add_argument("--per-pair-backend", default="highs", choices=["cupdlpx", "highs"])


def _append_rows(path: Path, rows: list[RunRecord]) -> None:
    """
    EN: append finished rows to a JSONL checkpoint **as they are produced**.

        Why: the driver previously wrote `records.json` only after the last test pair, so a run killed in the
        middle (OOM, session restart) left nothing -- that has already happened twice on this study. A partial
        file turns a kill from "no data" into "n rows", and `records.json` is still written at the end.

    CN: 每产出一批行就追加到 JSONL 检查点。原实现只在最后一对之后写 `records.json`，中途被杀（OOM、会话重启）
        就什么都不剩——本研究已经发生过两次。部分文件把"被杀"从"没有数据"变成"有 n 行"，结束时仍写完整文件。
    """
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row.to_dict(), sort_keys=True) + "\n")


def _evaluate(pair: tuple[int, int], *, thetas: dict[str, np.ndarray], scaler: DescriptorScaler,
              config: ChallengeConfig, index: Int[np.ndarray, "n"], geo: Float[np.ndarray, "n n"],
              norm_diameter: float, norm_area: float, kind: str) -> list[RunRecord]:
    """
    EN: score every supplied θ on one ordered pair, in **both** normalisations, with exact OT and the geodesic
        metric. The cost is built blockwise through the pair's own `FeatureMap` (so rich families are not scored
        as if they were diagonal) and the descriptors are already standardised by the training-fitted scaler.

    CN: 在一条有序对上评估所有给定 θ，**两种归一化**都报，精确 OT + 测地线度量。代价经该对自身的 `FeatureMap` 分块
        构造（避免把丰富族按对角族评分），描述子已用训练拟合的 scaler 标准化。
    """
    a_mesh = read_ply(Path(config.root) / f"tr_reg_{pair[0]:03d}.ply")
    b_mesh = read_ply(Path(config.root) / f"tr_reg_{pair[1]:03d}.ply")
    da = scaler.transform(descriptors(a_mesh)[index])
    db = scaler.transform(descriptors(b_mesh)[index])
    n = int(index.size)
    feature_map = _family(str(config.family), da, db, bands=int(config.bands),
                          basis_size=int(config.basis_size))
    rows: list[RunRecord] = []
    for name, theta in thetas.items():
        cost = _cost_rows(feature_map, theta, n)
        prediction = np.asarray(match(cost), dtype=np.int64)
        by_diameter = geodesic_errors(prediction, geo, norm_diameter)
        by_area = geodesic_errors(prediction, geo, norm_area)
        rows.append(RunRecord(
            study=str(config.study), geometry=f"{pair[0]:03d}->{pair[1]:03d}", arm=str(name),
            k=int(feature_map.n_features), m=n, draw=int(config.seeds[0]),
            metrics={"mean_diam": by_diameter["mean"], "median_diam": by_diameter["median"],
                     "miss_1pct_diam": by_diameter["miss_1pct"], "miss_5pct_diam": by_diameter["miss_5pct"],
                     "mean_area": by_area["mean"], "miss_5pct_area": by_area["miss_5pct"]},
            config=config.to_dict(),
            extra={"challenge": kind, "first": int(pair[0]), "second": int(pair[1]),
                   "first_subject": int(pair[0]) // 10, "second_subject": int(pair[1]) // 10,
                   "family": str(config.family), "n_features": int(feature_map.n_features),
                   "flatness": _flatness(cost)}))
    return rows


def main(argv: list[str] | None = None) -> int:
    """EN: entry point. CN: 入口。"""
    parser = build_parser(ChallengeConfig)
    args = parser.parse_args(list(argv) if argv is not None else None)
    config = ChallengeConfig.from_namespace(args, parser)
    run_dir = config.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    log = RunLog(run_dir / "run.log")
    preflight = preflight_check(run_tiny=False, need_gpu=False)
    preflight.write(run_dir / "preflight.json")
    log.line(f"preflight ok={preflight.ok} mem={preflight.mem_available_gb:.1f} GB")
    log.line(f"FAUST protocol (REGISTRATION PROXY): train subjects {config.train_subjects}, "
             f"test subjects {config.test_subjects}, family={config.family!r}, n_sub={config.n_sub}")

    train_pairs = ([(p, q, "intra") for p, q in
                    intra_pairs(config.train_subjects, int(config.intra_per_subject), seed=0)]
                   + [(p, q, "inter") for p, q in
                      inter_pairs(config.train_subjects, int(config.inter_per_subject), seed=0)])
    test_pairs = ([(p, q, "intra") for p, q in
                   intra_pairs(config.test_subjects, int(config.intra_per_subject), seed=0)]
                  + [(p, q, "inter") for p, q in
                     inter_pairs(config.test_subjects, int(config.inter_per_subject), seed=0)])
    log.line(f"{len(train_pairs)} training pairs, {len(test_pairs)} test pairs; "
             f"intra per subject={config.intra_per_subject}, inter per subject={config.inter_per_subject}")

    # EN: the scaler sees TRAINING shapes only. That is the protocol point, not an implementation detail.
    # CN: scaler 只看**训练**形状；这是协议要求，不是实现细节。
    index = shared_index(read_ply(Path(config.root) / "tr_reg_000.ply"), int(config.n_sub), seed=0)
    train_ids = sorted({pair[0] for pair in train_pairs} | {pair[1] for pair in train_pairs})
    scaler = DescriptorScaler.fit([descriptors(read_ply(Path(config.root) / f"tr_reg_{i:03d}.ply"))[index]
                                   for i in train_ids])

    # EN: fit theta on the training pairs: a per-pair fit for the shared LPs' edge sets, then the LPs.
    # CN: 在训练对上拟合 θ：先逐对拟合以得到共享 LP 的边集，再解 LP。
    built: list[PairProblem] = []
    started = time.perf_counter()
    for position, (a_id, b_id, kind) in enumerate(train_pairs):
        a_mesh = read_ply(Path(config.root) / f"tr_reg_{a_id:03d}.ply")
        b_mesh = read_ply(Path(config.root) / f"tr_reg_{b_id:03d}.ply")
        da = scaler.transform(descriptors(a_mesh)[index])
        db = scaler.transform(descriptors(b_mesh)[index])
        n = int(index.size)
        plan = sparse.csr_matrix((np.full(n, 1.0 / n), (np.arange(n), np.arange(n))), shape=(n, n))
        family = _family(str(config.family), da, db, bands=int(config.bands),
                         basis_size=int(config.basis_size))
        problem = InverseOTProblem(features=family, theta_set=SimplexTheta(int(family.n_features)),
                                   plan=plan)
        master = RecordingMaster(backend=str(config.per_pair_backend), tie_break=False)
        single = solve_inverse_ot_regret(
            problem,
            objective=RelaxedObjective(eps_r_rel=1e-6, check_pot=False, selection="none"),
            params=SolverParameters(tol=None, gamma=1, max_iterations=int(config.max_iterations),
                                    pruning=PruningParameters(enabled=False)),
            master=master)
        if master.last_edges is None:
            raise SystemExit(f"training pair {(a_id, b_id)}: no edge set recorded")
        built.append(PairProblem(features=family, problem=problem, edges=master.last_edges,
                                 theta_single=np.asarray(single.theta, dtype=np.float64),
                                 regret_single=float(single.regret), loss=None))
        if position % 10 == 0:
            log.line(f"  fitted {position + 1}/{len(train_pairs)} ({kind} {a_id}->{b_id}) "
                     f"|F|={int(master.last_edges.size)} in {time.perf_counter() - started:.0f}s")

    k = int(built[0].features.n_features)
    scales = [_uniform_regret(piece) for piece in built]
    shared = _solve_multi(built, k, scales=scales if all(np.isfinite(scales)) else None)
    if not shared.get("ok"):
        log.line(f"shared-regret LP failed: {shared.get('reason')}")
        shared_theta = None
    else:
        shared_theta = np.asarray(shared["theta"], dtype=np.float64)
        log.line(f"shared (regret) LP: shape {shared['shape']} nnz {shared['nnz']:,} "
                 f"objective {shared['objective']:.6g} nonzero={int((shared_theta > 1e-9).sum())}/{k}")

    margin_theta = None
    if int(config.with_margin):
        losses = []
        for a_id, b_id in [(p[0], p[1]) for p in train_pairs]:
            b_mesh = read_ply(Path(config.root) / f"tr_reg_{b_id:03d}.ply")
            geodesic = geodesic_matrix(b_mesh.vertices, b_mesh.faces, index)
            losses.append(np.asarray(geodesic, dtype=np.float64) / max(diameter(b_mesh.vertices), 1e-12))
        joint = _solve_shared_margin(built, k, losses)
        if not joint.get("ok"):
            log.line(f"shared-margin LP failed: {joint.get('reason')}")
        else:
            margin_theta = np.asarray(joint["theta"], dtype=np.float64)
            slack = np.asarray(joint["slack"], dtype=np.float64)
            log.line(f"shared (soft margin) LP: sum(xi)={float(slack.sum()):.4e} "
                     f"max={float(slack.max()):.4e} nonzero={int((margin_theta > 1e-9).sum())}/{k}")

    rows: list[RunRecord] = []
    checkpoint = run_dir / "records.partial.jsonl"
    checkpoint.write_text("", encoding="utf-8")
    started = time.perf_counter()
    for position, (a_id, b_id, kind) in enumerate(test_pairs):
        a_mesh = read_ply(Path(config.root) / f"tr_reg_{a_id:03d}.ply")
        b_mesh = read_ply(Path(config.root) / f"tr_reg_{b_id:03d}.ply")
        geo = geodesic_matrix(b_mesh.vertices, b_mesh.faces, index)
        norm_diameter = diameter(b_mesh.vertices)
        norm_area = float(np.sqrt(surface_area(b_mesh.vertices, b_mesh.faces)))
        # EN: the oracle is fitted on THIS pair and is labelled as such; it is an upper bound, not a method.
        # CN: oracle 在本对上拟合并如此标注；它是上界，不是方法结果。
        a_desc = scaler.transform(descriptors(a_mesh)[index])
        b_desc = scaler.transform(descriptors(b_mesh)[index])
        oracle_fit = fit_theta(_family(str(config.family), a_desc, b_desc, bands=int(config.bands),
                                       basis_size=int(config.basis_size)),
                               rel=None, max_iterations=int(config.max_iterations))
        thetas: dict[str, np.ndarray] = {
            "uniform": np.full(k, 1.0 / k),
            "oracle": np.asarray(oracle_fit["theta"], dtype=np.float64),
        }
        if shared_theta is not None:
            thetas["shared"] = shared_theta
        if margin_theta is not None:
            thetas["shared_margin"] = margin_theta
        produced = _evaluate((a_id, b_id), thetas=thetas, scaler=scaler, config=config, index=index,
                             geo=geo, norm_diameter=norm_diameter, norm_area=norm_area, kind=kind)
        rows.extend(produced)
        _append_rows(checkpoint, produced)
        if position % 5 == 0 or position == len(test_pairs) - 1:
            log.line(f"  scored {position + 1}/{len(test_pairs)} ({kind} {a_id}->{b_id}) "
                     f"in {time.perf_counter() - started:.0f}s")

    table = summarise(rows, group=("challenge", "arm"), metrics=("mean_diam", "miss_5pct_diam"))
    log.line("\n" + format_table(table, ("challenge", "arm", "n_draws", "mean_diam_mean",
                                         "miss_5pct_diam_mean")))
    log.line("headline metric: diameter normalisation (x100 of mean_diam). "
             "The registration proxy is NOT a leaderboard number.")
    write_records(run_dir / "records.json", rows, provenance=PROVENANCE)
    log.line(f"wrote {run_dir / 'records.json'} ({len(rows)} rows, {time.perf_counter() - started:.0f}s)")
    log.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
