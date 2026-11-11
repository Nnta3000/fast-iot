"""
EN: **FAUST correspondence -- the actual experiment: does a theta fitted on ONE pair transfer to OTHERS?**

    Gates 6.0-6.3 passed (`data_check.py`, `gates.py`). This module runs the experiment those gates cleared
    the way for.

      train on pair (train_a, train_b): fit `theta` on `pi_hat = I/n`, exactly as the oracle in `gates.py`
          did -- the difference is that the training pair is then **discarded**;
      evaluate on each held-out pair: solve forward OT with the learned cost and report the error.

    Arms, all evaluated on the same held-out pairs with the same descriptor standardisation:

      `position_only`   the naive cost;
      `uniform_theta`   equal weights over the descriptor dictionary (the honest baseline);
      `grouped_uniform` equal weight per descriptor GROUP (K = 4, a capacity control);
      `learned`         the theta fitted on the training pair ONLY -- this is the experiment;
      `oracle_theta`    a theta fitted on the held-out pair itself. A CEILING, not a result;
      `nn_<arm>`        per-row argmin instead of OT, to separate the weights from the coupling.

    **Metric.** The FAUST protocol: geodesic distance on the target between predicted and true vertex,
    normalised by `sqrt(surface area)` of the target. Mean, median, and the share of vertices beyond 1 % and
    5 % (cumulative-error curve points). Geodesics come from `potpourri3d`'s heat method.

    **Statistics.** A paired bootstrap over held-out **pairs**, not vertices: vertices inside a mesh are not
    independent, so a vertex-level bootstrap would manufacture significance.

    CN: **FAUST 对应关系 —— 真正的实验**：在一对上拟合 θ，在其余对上评估；测地误差按 `sqrt(area)` 归一；
        显著性用**按对**的自助法，不按顶点。

    Run:
        .venv/bin/python -u -m iot_experiments.studies.real.faust.correspondence --train 17 65
"""

from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sparse
from jaxtyping import Float, Int

_REPO_ROOT = Path(__file__).resolve().parents[4]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from iot.core.features import SquaredDifferenceFeatures  # noqa: E402
from iot.core.parameters import PruningParameters, RelaxedObjective, SolverParameters  # noqa: E402
from iot.core.problem import InverseOTProblem  # noqa: E402
from iot.core.theta import SimplexTheta  # noqa: E402
from iot.relaxed.loop_regret import solve_inverse_ot_regret  # noqa: E402
from iot_experiments.harness.config import RunConfig, build_parser  # noqa: E402
from iot_experiments.harness.faust import (  # noqa: E402
    FaustMesh,
    descriptors,
    geodesic_matrix,
    mirror_indices,
    read_ply,
    shared_index,
    standardise,
    surface_area,
)
from iot_experiments.harness.faust_match import (  # noqa: E402
    cost_matrix as _cost_matrix,
    geodesic_errors,
    match,
)
from iot_experiments.harness.preflight import check as preflight_check  # noqa: E402
from iot_experiments.harness.records import (  # noqa: E402
    RunLog,
    RunRecord,
    format_table,
    summarise,
    write_records,
)
from iot_experiments.harness.solver import box_simplex  # noqa: E402

DEFAULT_ROOT = "/home/agent/Documents/fast-iot/data/faust_extract/MPI-FAUST/training/registrations"
GROUPS: dict[str, slice] = {"sparse": slice(0, 3), "area": slice(3, 4), "curv": slice(4, 5),
                            "hks": slice(5, 13)}
# CN: HKS 列在 13 维描述子里的位置。`LATE` 取 t = 32/64/128，正是拟合出的 θ 加载的那几列——
#     因此"等权晚 HKS"是**最关键的对照**：如果它已经和学到的 θ 一样好，那么学习只发现了"哪一块重要"。
# EN: where the HKS columns sit in the 13-dim descriptor. `LATE` is t = 32/64/128, which is exactly where the
#     fitted theta puts its mass -- so "uniform late HKS" is the crucial control: if it already matches the
#     learned theta, learning only discovered WHICH BLOCK matters, not any weighting inside it.
HKS: slice = slice(5, 13)
HKS_LATE: slice = slice(10, 13)

PROVENANCE: dict[str, object] = {
    "question": "does a theta fitted on one registered FAUST pair recover the true vertex correspondence on "
                "HELD-OUT pairs, better than uniform weights?",
    "plan": "pi_hat = I/n on a shared index subsample of both meshes (the registered bijection)",
    "pairing": "inter = pairs across two different subjects (the hard case); intra = the same subject in two "
               "poses (the easy case). Subject-disjointness holds either way",
    "split": "SUBJECT-DISJOINT: training and evaluation subjects do not intersect (checked by `check_split`, "
             "which exits on violation). Training pair ids come from the training subjects only and every "
             "evaluation pair from the test subjects only. An earlier revision of this module had NO such "
             "check and its training pairs shared subjects with the evaluation pairs -- that earlier result is "
             "pose transfer within seen subjects, not subject transfer",
    "box_selection": "the box width (rel) AND simplex-only are compared by mean miss@5% on VALIDATION pairs "
                     "drawn from the TRAINING subjects (disjoint from the test subjects and from the training "
                     "pair). The selected arm is copied to `learned_selected`; every other width is reported "
                     "on the test pairs too, but took no part in the selection",
    "box_meaning": "rel is the INVERSE box width: theta in [rel/K, 1/(rel*K)], so rel=0.05 is the widest box "
                   "and rel=0.9 the narrowest; rel=None (learned_simplex) is the whole simplex",
    "vertex_subsamples": "several index subsamples (default seeds 0,1,2); theta is REFITTED per seed because it "
                         "depends on the sampled vertices",
    "bootstrap_averaged_theta": "learned_avg_<rel>: the per-coordinate mean of theta across the vertex "
                                "subsamples, evaluated on the test pairs. It reduces the variance of how much "
                                "mass each HKS scale takes, which the corner analysis showed is not identified",
    "support_matched_control": "support_matched_<rel>: uniform weights over the NONZERO support of the averaged "
                               "theta. Same selected descriptors, no fitted magnitudes -- separates 'right block' "
                               "from 'right weights inside the block'",
    "metric": "GEODESIC distance on the target (potpourri3d heat method) between predicted and true vertex, "
              "normalised by sqrt(surface area) of the target -- the FAUST protocol. Miss rates at 1% and 5%",
    "descriptors": "K=13: extrinsic xyz (centred, scaled by diameter), log 1-ring area, log curvature, 8 HKS "
                   "columns; standardised with the pooled source+target mean/std",
    "groups": "sparse (xyz), area, curv, hks -- used by the grouped_uniform capacity control",
    "block_controls": "uniform_hks (8 HKS columns, equal weight) and uniform_late_hks (t = 32/64/128 only, "
                      "equal weight). The fitted theta loads almost entirely on the late-HKS columns, so these "
                      "decide whether learning did more than select the block",
    "engine": "relaxed regret loop, box x simplex, prune disabled, HiGHS master",
    "stats": "paired bootstrap over held-out PAIRS (1000 resamples), never over vertices; each pair is first "
             "averaged across vertex-subsample seeds (seeds are repeated measures, not independent pairs)",
    "corner": "the fitted theta sits on the box floor/ceiling and the loop reports converged=False; the "
              "weights are therefore 'a feasible point the solver accepted', not a proved optimum",
    "not_sota": "this is not a state-of-the-art claim; deep functional maps reach far lower FAUST errors. The "
                "claim under test is only that a learned mixture beats hand-weighted ones",
}


@dataclasses.dataclass(frozen=True)
class CorrespondenceConfig(RunConfig):
    """CN: 旋钮。EN: knobs."""

    study: str = "faust_correspondence"
    section: str = "faust_correspondence"
    root: str = DEFAULT_ROOT
    train_subjects: tuple[int, ...] = (0, 1, 2, 3, 4)
    test_subjects: tuple[int, ...] = (5, 6, 7, 8, 9)
    train_pair: tuple[int, ...] = (0, 13)
    test_pairs: tuple[int, ...] = ()
    test_pair_count: int = 8
    pairing: str = "inter"        # inter = different subjects | intra = same subject, different pose
    seeds: tuple[int, ...] = (0, 1, 2)
    oracle: int = 1            # refit theta on each evaluated pair (a ceiling, not a method)
    validation: int = 1        # evaluate the validation pairs (used to choose the box width)
    rels: tuple[float, ...] = (0.05, 0.1, 0.25, 0.5, 0.9)
    simplex: int = 1
    val_pair_count: int = 4
    n_sub: int = 800
    rel: float = 0.25
    max_iterations: int = 30
    bootstrap: int = 1000

    @classmethod
    def add_study_arguments(cls, parser) -> None:
        """CN: 只加本研究特有的参数。EN: only this study's flags."""
        parser.add_argument("--root", default=DEFAULT_ROOT)
        parser.add_argument("--train-subjects", type=int, nargs="+", default=[0, 1, 2, 3, 4])
        parser.add_argument("--test-subjects", type=int, nargs="+", default=[5, 6, 7, 8, 9])
        parser.add_argument("--train-pair", type=int, nargs=2, default=[0, 13])
        parser.add_argument("--test-pairs", type=int, nargs="*", default=[])
        parser.add_argument("--test-pair-count", type=int, default=8)
        parser.add_argument("--pairing", default="inter", choices=["inter", "intra"])
        parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
        parser.add_argument("--oracle", type=int, default=1,
                            help="0 skips the per-pair oracle refit (the ceiling); the headline arms are "
                                 "unaffected")
        parser.add_argument("--validation", type=int, default=1,
                            help="0 skips the validation pairs (only meaningful when the box width is already "
                                 "settled")
        parser.add_argument("--rels", type=float, nargs="+", default=[0.05, 0.1, 0.25, 0.5, 0.9])
        parser.add_argument("--simplex", type=int, default=1)
        parser.add_argument("--val-pair-count", type=int, default=4)
        parser.add_argument("--n-sub", type=int, default=800)
        parser.add_argument("--rel", type=float, default=0.25)
        parser.add_argument("--max-iterations", type=int, default=30)
        parser.add_argument("--bootstrap", type=int, default=1000)


def subject_of(vertex_id: int) -> int:
    """
    CN: FAUST 训练集编号是 **subject-major**：`id = 10*subject + pose`（数据检查从面积相关矩阵里恢复出来的，
        不是假设）。EN: FAUST training ids are **subject-major**: `id = 10*subject + pose` (recovered from the
        area-correlation matrix by `data_check.py`, not assumed).
    """
    return int(vertex_id) // 10


def default_pairs(subjects: tuple[int, ...], count: int) -> list[tuple[int, int]]:
    """
    CN: 从一个 subject 集合里确定性地生成 `count` 对：先遍历所有 subject 组合，用完再换一组 pose。
    EN: deterministically build `count` pairs from a subject set: walk all subject combinations first, then
        move to the next pose pair.
    """
    poses = (0, 2, 4, 6, 8, 1, 3, 5, 7, 9)
    uniq = sorted({int(v) for v in subjects})
    if len(uniq) < 2:
        raise SystemExit(f"need at least two subjects, got {uniq}")
    combos = [(a, b) for a in uniq for b in uniq if a < b]
    out: list[tuple[int, int]] = []
    i = 0
    while len(out) < int(count):
        a, b = combos[i % len(combos)]
        k = i // len(combos)
        # CN: pose 同时随 i 和"第几轮 k"变化；只依赖 i 会有周期性，能生成的**不同**对会很快用完。
        # EN: the pose depends on both `i` and the round `k`; depending on `i` alone is periodic and runs out of
        #     distinct pairs quickly.
        pa = poses[(3 * i + 5 * k) % len(poses)]
        pb = poses[(2 * i + 3 + 7 * k) % len(poses)]
        pair = (10 * a + pa, 10 * b + pb)
        if pair not in out:
            out.append(pair)
        i += 1
        if i > 50 * int(count) + 50:
            raise SystemExit("could not build enough distinct pairs from that subject set")
    return out


def intra_pairs(subjects: tuple[int, ...], count: int) -> list[tuple[int, int]]:
    """
    CN: **同受试者、不同姿态**的配对：`10*s + p1` 与 `10*s + p2`。这比跨受试者容易，所以它是"难度轴"的下端；
        已见过的受试者不出现在评估侧（仍由 `check_split` 保证）。
    EN: **same subject, different pose** pairs: `10*s + p1` against `10*s + p2`. Easier than inter-subject, so
        it is the low end of the difficulty axis; the disjointness of the subject sets is still enforced by
        `check_split`.
    """
    poses = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9)
    uniq = sorted({int(v) for v in subjects})
    out: list[tuple[int, int]] = []
    i = 0
    while len(out) < int(count):
        subject = uniq[i % len(uniq)]
        first = poses[(2 * i) % len(poses)]
        second = poses[(2 * i + 3) % len(poses)]
        pair = (10 * subject + first, 10 * subject + second)
        if pair[0] != pair[1] and pair not in out:
            out.append(pair)
        i += 1
        if i > 50 * int(count) + 50:
            raise SystemExit("could not build enough distinct intra-subject pairs")
    return out


def check_split(train_subjects: tuple[int, ...], test_subjects: tuple[int, ...],
                train_pair: tuple[int, int], test_pairs: list[tuple[int, int]]) -> None:
    """
    CN: **主体不相交**：训练 subject 与测试 subject 必须没有交集，且每一对的两端都要落在各自那一侧。
        任何一个条件不满足都直接退出——上一版就是因为没有这个检查，训练和评估共享了 subject。
    EN: **subject-disjoint**: the training and test subject sets must not intersect, and both ends of every pair
        must lie on their own side. Any violation exits: the earlier revision had no such check, which is how
        the training and evaluation pairs came to share subjects.
    """
    tr, te = set(int(v) for v in train_subjects), set(int(v) for v in test_subjects)
    if tr & te:
        raise SystemExit(f"train and test subjects overlap: {sorted(tr & te)}")
    for vid in train_pair:
        if subject_of(vid) not in tr:
            raise SystemExit(f"train pair id {vid} has subject {subject_of(vid)}, not in {sorted(tr)}")
    for pair in test_pairs:
        for vid in pair:
            if subject_of(vid) not in te:
                raise SystemExit(f"test pair id {vid} has subject {subject_of(vid)}, not in {sorted(te)}")


def fit_theta(features: SquaredDifferenceFeatures, *, rel: float | None, max_iterations: int
              ) -> dict[str, object]:
    """
    CN: 在 `pi_hat = I/n` 上拟合 θ。`rel=None` 表示**只用单纯形**（无盒子）。失败返回 `theta=None`，
        **绝不**回退成 uniform。

        `rel` 是盒子的**宽度倒数**：`theta in [rel/K, 1/(rel*K)]`，所以 `rel` 越小盒子越宽，
        `rel=1` 时盒子退化到 uniform 一点。`rel=None` 是最大的可行集（整个单纯形）。
    EN: fit `theta` on `pi_hat = I/n`. `rel=None` means **simplex only** (no box). On failure `theta=None`;
        uniform is **never** substituted.

        `rel` is the inverse box width: `theta in [rel/K, 1/(rel*K)]`, so a *smaller* `rel` is a *wider* box
        and `rel = 1` collapses it to the uniform point. `rel=None` is the largest feasible set.
    """
    # CN: **不物化 (n, n, K)**。`SquaredDifferenceFeatures` 只按被请求的边算特征，故峰值与 n 无关地小。
    # EN: **no (n, n, K)**. `SquaredDifferenceFeatures` computes features only for the edges the solver asks
    #     for, so the peak does not grow with `n`.
    n = int(features.n_source)
    k = int(features.n_features)
    plan = sparse.csr_matrix((np.full(n, 1.0 / n), (np.arange(n), np.arange(n))), shape=(n, n))
    theta_set = SimplexTheta(int(k)) if rel is None else box_simplex(int(k), box_relative=float(rel))
    problem = InverseOTProblem(features=features, theta_set=theta_set, plan=plan)
    result = solve_inverse_ot_regret(
        problem,
        objective=RelaxedObjective(eps_r_rel=1e-6, check_pot=False, selection="none"),
        params=SolverParameters(tol=None, gamma=1, max_iterations=int(max_iterations),
                                pruning=PruningParameters(enabled=False)))
    return {"theta": np.asarray(result.theta, dtype=np.float64), "regret": float(result.regret),
            "converged": bool(result.converged), "edges": int(result.finite_edges),
            "rel": rel}


def _sample_index(mesh: FaustMesh, config: CorrespondenceConfig, seed: int) -> Int[np.ndarray, "n"]:
    """
    CN: 共享指标子集（两形状同一组下标，双射得以保留）。`seed` 决定抽哪一组，因此可以抽多组。
    EN: the shared index subsample (the same index set on both shapes, so the bijection survives). `seed`
        selects which draw, so several can be taken. Thin wrapper over `harness.faust.shared_index`.
    """
    return shared_index(mesh, int(config.n_sub), int(seed))


def _evaluate_pair(a_mesh: FaustMesh, b_mesh: FaustMesh, *, config: CorrespondenceConfig,
                   thetas: dict[str, Float[np.ndarray, "k"]], seed: int,
                   baselines: bool = True) -> list[RunRecord]:
    """CN: 在一对形状上评估全部臂（θ 由外部提供）。EN: evaluate every arm on one pair."""
    idx = _sample_index(a_mesh, config, seed)
    a, b = standardise(descriptors(a_mesh)[idx], descriptors(b_mesh)[idx])
    n = int(idx.size)
    geo = geodesic_matrix(b_mesh.vertices, b_mesh.faces, idx)
    norm = float(np.sqrt(surface_area(b_mesh.vertices, b_mesh.faces)))
    mirror, _residual = mirror_indices(b_mesh.vertices, idx)

    # CN: **每个臂单独算一个 `n×n` 双线性代价**，不物化 `(n, n, K)`。等权臂就是权重向量是常数的特例，
    #     所以下面全部写成"权重向量 + `_cost_matrix`"，不再需要张量。
    # EN: **one `n x n` bilinear cost per arm**; `(n, n, K)` is never materialised. A uniform arm is just the
    #     case where the weight vector is constant, so every arm below is "weights + `_cost_matrix`".
    k = int(a.shape[1])
    uniform_theta = np.full(k, 1.0 / k)
    position_theta = np.zeros(k); position_theta[:3] = 1.0 / 3.0
    hks_theta = np.zeros(k); hks_theta[HKS] = 1.0 / float(HKS.stop - HKS.start)
    late_theta = np.zeros(k); late_theta[HKS_LATE] = 1.0 / float(HKS_LATE.stop - HKS_LATE.start)

    # CN: **分组等权 = 组内均值的平均**，不是"把组均值广播回该组再对 13 列求平均"——后者按组大小重新加权，
    #     在代数上恒等于 `uniform_theta`（实测逐位相同），于是那个"容量对照"其实什么都没对照。
    # EN: **grouped_uniform = the average of the group means**, NOT "broadcast each group mean back over its
    #     coordinates and average the 13 columns" -- that re-weights by group size and is algebraically
    #     identical to `uniform_theta` (measured: bit-identical), so the "capacity control" controlled nothing.
    grouped_cost = np.zeros((int(a.shape[0]), int(b.shape[0])), dtype=np.float64)
    for sl in GROUPS.values():
        per_group = np.zeros(k); per_group[sl] = 1.0 / float(sl.stop - sl.start)
        grouped_cost += _cost_matrix(a, b, per_group)
    grouped_cost /= float(len(GROUPS))

    costs: dict[str, Float[np.ndarray, "n n"]] = {
        "position_only": _cost_matrix(a, b, position_theta),
        # CN: **单块等权对照**：只对 HKS 列等权，以及只对晚 HKS 列等权。学到的 θ 基本就是晚 HKS，
        #     所以这两行决定"学习"是否只是选块。
        # EN: **single-block uniform controls**: equal weights over the HKS columns only, and over the late HKS
        #     columns only. The fitted theta is essentially late HKS, so these rows decide whether "learning"
        #     amounted to nothing more than picking the block.
        "uniform_hks": _cost_matrix(a, b, hks_theta),
        "uniform_late_hks": _cost_matrix(a, b, late_theta),
        "uniform_theta": _cost_matrix(a, b, uniform_theta),
        "grouped_uniform": grouped_cost,
    }
    if not baselines:
        # CN: 第二遍只算新加入的 θ（平均 θ / 支撑匹配），不重复算基线。
        # EN: the second pass evaluates only the new thetas (averaged / support-matched); no baseline rerun.
        costs = {}
    for name, theta in thetas.items():
        if theta is not None:
            costs[name] = _cost_matrix(a, b, np.asarray(theta, dtype=np.float64))

    geometry = f"pair{a_mesh.path.stem[7:]}-{b_mesh.path.stem[7:]}"
    rows: list[RunRecord] = []
    for arm, cost in costs.items():
        for mode, pred in (("ot", match(cost)), ("nn", np.argmin(cost, axis=1).astype(np.int64))):
            label = f"nn_{arm}" if mode == "nn" else arm
            rows.append(RunRecord(
                study=str(config.study), geometry=geometry, arm=label, k=int(a.shape[1]),
                m=int(n), draw=int(seed), metrics=geodesic_errors(pred, geo, norm, mirror),
                config=config.to_dict()))
    return rows


def main(argv: list[str] | None = None) -> int:
    """CN: 入口。EN: entry point."""
    parser = build_parser(CorrespondenceConfig)
    args = parser.parse_args(list(argv) if argv is not None else None)
    config = CorrespondenceConfig.from_namespace(args, parser)
    run_dir = config.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    log = RunLog(run_dir / "run.log")
    pre = preflight_check(run_tiny=False, need_gpu=False)
    pre.write(run_dir / "preflight.json")
    log.line(f"preflight ok={pre.ok} mem={pre.mem_available_gb:.1f} GB")

    root = Path(config.root)
    train_subjects = tuple(int(v) for v in config.train_subjects)
    test_subjects = tuple(int(v) for v in config.test_subjects)
    train_pair = (int(config.train_pair[0]), int(config.train_pair[1]))
    if config.test_pairs:
        flat = [int(v) for v in config.test_pairs]
        if len(flat) % 2:
            raise SystemExit("--test-pairs needs an even number of ids (i j k l ...)")
        test_pairs = [(flat[i], flat[i + 1]) for i in range(0, len(flat), 2)]
    elif str(config.pairing) == "intra":
        test_pairs = intra_pairs(test_subjects, int(config.test_pair_count))
    else:
        test_pairs = default_pairs(test_subjects, int(config.test_pair_count))
    check_split(train_subjects, test_subjects, train_pair, test_pairs)

    # CN: **验证对来自训练 subject**，与测试 subject 不相交，也不同于训练对；盒子宽度只在这里选。
    # EN: **validation pairs come from the training subjects** -- disjoint from the test subjects and from the
    #     training pair itself. The box width is chosen here and nowhere else.
    # CN: 先生成一个足够大的候选池（多要几对，因为要把训练对剔除），再取前 `val_pair_count` 个。
    # EN: build a large enough pool first (the training pair is filtered out), then take the first
    #     `val_pair_count` of them.
    pool = default_pairs(train_subjects, 4 * int(config.val_pair_count) + 16)
    val_candidates = ([p for p in pool if p != train_pair and p[::-1] != train_pair]
                      [: int(config.val_pair_count)] if int(config.validation) else [])
    for pair in val_candidates:
        for vid in pair:
            if subject_of(vid) not in set(train_subjects):
                raise SystemExit(f"validation pair {pair} leaves the training subjects")

    rels: list[float | None] = [float(r) for r in config.rels]
    if int(config.simplex):
        rels.append(None)

    def arm_of(rel: float | None) -> str:
        return "learned_simplex" if rel is None else f"learned_rel{rel:g}"

    log.line(f"train subjects {train_subjects}  |  test subjects {test_subjects}  (disjoint: PASS)")
    log.line(f"train pair {train_pair}; validation pairs (train subjects) {val_candidates}")
    log.line(f"test pairs (test subjects) {test_pairs}")
    log.line(f"box widths {[arm_of(r) for r in rels]}; vertex subsamples seeds {tuple(config.seeds)}")

    a_tr = read_ply(root / f"tr_reg_{train_pair[0]:03d}.ply")
    b_tr = read_ply(root / f"tr_reg_{train_pair[1]:03d}.ply")

    rows: list[RunRecord] = []
    per_rel_thetas: dict[float | None, list[np.ndarray]] = {rel: [] for rel in rels}
    for seed in [int(v) for v in config.seeds]:
        idx = _sample_index(a_tr, config, seed)
        a, b = standardise(descriptors(a_tr)[idx], descriptors(b_tr)[idx])
        # CN: 特征按需计算，**不建 (n, n, K) 张量**。EN: features on demand; no `(n, n, K)` tensor.
        features = SquaredDifferenceFeatures(a, b)
        fits: dict[float | None, dict[str, object]] = {}
        for rel in rels:
            fit = fit_theta(features, rel=rel, max_iterations=int(config.max_iterations))
            fits[rel] = fit
            theta_r = fit["theta"]
            n_at_bound = int(np.sum(np.isclose(theta_r, theta_r.min(), atol=1e-9))) if theta_r is not None else -1
            if theta_r is not None:
                per_rel_thetas[rel].append(np.asarray(theta_r, dtype=np.float64))
            log.line(f"[seed {seed}] {arm_of(rel):<18} regret={fit['regret']:.4g} "
                     f"converged={fit['converged']} min={0.0 if theta_r is None else theta_r.min():.4f} "
                     f"max={0.0 if theta_r is None else theta_r.max():.4f} at_min={n_at_bound}")
        thetas = {arm_of(rel): fit["theta"] for rel, fit in fits.items()
                  if fit["theta"] is not None}

        for pair in val_candidates:
            a_m = read_ply(root / f"tr_reg_{pair[0]:03d}.ply")
            b_m = read_ply(root / f"tr_reg_{pair[1]:03d}.ply")
            # CN: `RunRecord` 是 frozen 的，不能就地改字段；改用 `dataclasses.replace` 复制。
            # EN: `RunRecord` is frozen and cannot be mutated; retag by copying.
            rows.extend(dataclasses.replace(r, geometry=f"val:{r.geometry}")
                        for r in _evaluate_pair(a_m, b_m, config=config, thetas=thetas, seed=seed))

        for pair in test_pairs:
            a_m = read_ply(root / f"tr_reg_{pair[0]:03d}.ply")
            b_m = read_ply(root / f"tr_reg_{pair[1]:03d}.ply")
            these = dict(thetas)
            if int(config.oracle):
                o_idx = _sample_index(a_m, config, seed)
                ao, bo = standardise(descriptors(a_m)[o_idx], descriptors(b_m)[o_idx])
                oracle_fit = fit_theta(SquaredDifferenceFeatures(ao, bo), rel=0.25,
                                       max_iterations=int(config.max_iterations))
                if oracle_fit["theta"] is not None:
                    these["oracle_theta"] = oracle_fit["theta"]
            # CN: 测试行也打前缀，否则后面的"选中复制"和"去前缀"都匹配不到（上一版就是这个 bug）。
            # EN: tag test rows too, otherwise the later "copy the selected arm" and "strip prefix" steps match
            #     nothing (that was the bug in the previous revision).
            rows.extend(dataclasses.replace(r, geometry=f"test:{r.geometry}")
                        for r in _evaluate_pair(a_m, b_m, config=config, thetas=these, seed=seed))

    # CN: **选择**：按验证对上的 miss@5% 均值选盒子宽度（各种子合并）。
    # EN: **selection**: pick the box width by mean miss@5% on the VALIDATION pairs, pooled over seeds.
    val_scores: dict[str, list[float]] = {}
    for r in rows:
        if r.geometry.startswith("val:") and r.arm in {arm_of(rel) for rel in rels}:
            val_scores.setdefault(r.arm, []).append(r.metrics["miss_5pct"])
    log.line("\nvalidation miss@5% by box width (lower is better; this is the selection criterion):")
    for arm, values in sorted(val_scores.items(), key=lambda kv: float(np.mean(kv[1]))):
        log.line(f"   {arm:<20} {float(np.mean(values)):.4f}  (n={len(values)})")
    if val_scores:
        best_arm = min(val_scores, key=lambda k: float(np.mean(val_scores[k])))
    else:
        # CN: 没有验证对时不做选择，直接取 simplex（它是上一轮在验证对上选出的赢家）。
        # EN: with no validation pairs there is no selection; use simplex, which won the previous sweep.
        best_arm = arm_of(None)
        log.line("no validation pairs: box width not selected here; using learned_simplex")

    # CN: **把选中的那个 θ 的测试读数复制成 `learned_selected`**，其余保留以便对照（它们没有参与选择）。
    # EN: **copy the selected theta's test readings as `learned_selected`**; the others are kept for contrast
    #     (they took no part in the selection).
    for r in list(rows):
        if r.geometry.startswith("test:") and r.arm == best_arm:
            rows.append(RunRecord(study=r.study, geometry=r.geometry, arm="learned_selected",
                                  k=r.k, m=r.m, draw=r.draw, metrics=dict(r.metrics),
                                  config=r.config, extra={"selected_from": best_arm}))
    # CN: 测试集几何名去掉前缀，便于分组（复制，不就地改）。
    # EN: strip the prefix for grouping (by copy -- `RunRecord` is frozen).
    rows = [dataclasses.replace(r, geometry=r.geometry[5:]) if r.geometry.startswith("test:") else r
            for r in rows]
    log.line(f"selected by validation: {best_arm}")

    # CN: **自助平均 θ（bootstrap-averaged）**：跨种子取每个坐标的均值，降低"哪个 HKS 尺度拿多少质量"的方差
    #     （corner 分析表明这一部分未被识别）。**支撑匹配对照（support-matched）**：取平均 θ 的非零支撑并在其上
    #     等权——保留"哪些描述子被选中"，丢掉拟合幅值，从而把"选块"与"块内权重"分开。
    # EN: **bootstrap-averaged theta**: per-coordinate mean across vertex subsamples, which reduces the
    #     variance of "how much mass each HKS scale takes" (the corner analysis showed this is not
    #     identified). **support-matched control**: uniform weights over the nonzero support of that average --
    #     same selected descriptors, no fitted magnitudes, separating "right block" from "right weights".
    averaged: dict[float | None, np.ndarray] = {
        rel: np.mean(np.stack(values), axis=0) for rel, values in per_rel_thetas.items() if values}
    derived: dict[str, np.ndarray] = {}
    for rel, mean_theta in averaged.items():
        tag = "simplex" if rel is None else f"rel{rel:g}"
        derived[f"learned_avg_{tag}"] = np.asarray(mean_theta, dtype=np.float64)
        # CN: 支撑阈值必须**相对**：CuPDLPx 在盒内留下 ~1e-7 的噪声，绝对 1e-9 会把它们全算成支撑
        #     （实测 simplex 臂因此报出 13/13 而不是 8/13）。用最大值的千分之一。
        # EN: the support threshold must be **relative**: CuPDLPx leaves ~1e-7 inside the box and an absolute
        #     1e-9 counts all of it (measured: the simplex arm reported 13/13 instead of 8/13).
        support = np.asarray(mean_theta) > 1e-3 * float(np.max(mean_theta))
        if bool(support.any()):
            matched = np.zeros_like(mean_theta)
            matched[support] = 1.0 / float(np.count_nonzero(support))
            derived[f"support_matched_{tag}"] = np.asarray(matched, dtype=np.float64)
    if derived:
        for name, theta in derived.items():
            log.line(f"derived {name}: support {int(np.count_nonzero(theta > 1e-9))}/{theta.size} "
                     f"theta={np.round(theta, 4).tolist()}")
        for pair in test_pairs:
            a_m = read_ply(root / f"tr_reg_{pair[0]:03d}.ply")
            b_m = read_ply(root / f"tr_reg_{pair[1]:03d}.ply")
            rows.extend(dataclasses.replace(r, geometry=f"test:{r.geometry}")
                        for r in _evaluate_pair(a_m, b_m, config=config, thetas=derived,
                                                seed=int(config.seeds[0]), baselines=False))
        rows = [dataclasses.replace(r, geometry=r.geometry[5:]) if r.geometry.startswith("test:") else r
                for r in rows]

    baselines = ("uniform_theta", "grouped_uniform", "uniform_hks", "uniform_late_hks", "position_only")
    per_pair_seed: dict[str, dict[str, dict[int, float]]] = {}
    for r in rows:
        if not r.geometry.startswith("val:"):
            per_pair_seed.setdefault(r.geometry, {}).setdefault(r.arm, {})[r.draw] = r.metrics["miss_5pct"]
    rng = np.random.default_rng(int(config.seed))
    log.line("")
    compare = ("learned_selected",) + tuple(arm_of(rel) for rel in rels)
    for cand in compare:
        for base in baselines:
            diffs = []
            for _geometry, arms in per_pair_seed.items():
                if cand in arms and base in arms:
                    common = sorted(set(arms[cand]) & set(arms[base]))
                    if common:
                        diffs.append(float(np.mean([arms[cand][s] - arms[base][s] for s in common])))
            diffs_arr = np.asarray(diffs, dtype=np.float64)
            if not diffs_arr.size:
                continue
            boots = np.asarray([rng.choice(diffs_arr, size=diffs_arr.size, replace=True).mean()
                                for _ in range(int(config.bootstrap))], dtype=np.float64)
            lo = float(np.percentile(boots, 2.5))
            hi = float(np.percentile(boots, 97.5))
            wins = int(np.sum(diffs_arr < 0))
            log.line(f"{cand:<20} - {base:<18} miss@5%: mean {diffs_arr.mean():+.4f} "
                     f"95% CI [{lo:+.4f}, {hi:+.4f}]  wins {wins}/{diffs_arr.size}")
            if cand == "learned_selected" or base == "uniform_theta":
                rows.append(RunRecord(study=str(config.study), geometry="bootstrap",
                                      arm=f"{cand}_minus_{base}", k=13, m=int(config.n_sub), draw=-1,
                                      metrics={"mean": float(diffs_arr.mean()), "ci_lo": lo, "ci_hi": hi,
                                               "n_pairs": float(diffs_arr.size), "wins": float(wins)},
                                      config=config.to_dict(),
                                      extra={"selected_from": best_arm,
                                             "train_pair": list(train_pair)}))

    table = summarise([r for r in rows if not r.geometry.startswith("val:")], group=("arm",),
                      metrics=("mean", "miss_1pct", "miss_5pct"))
    log.line("\n" + format_table(table, ("arm", "n_draws", "mean_mean", "miss_1pct_mean", "miss_5pct_mean")))
    write_records(run_dir / "records.json", rows, provenance=PROVENANCE)
    log.line(f"wrote {run_dir / 'records.json'} ({len(rows)} rows)")
    log.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
