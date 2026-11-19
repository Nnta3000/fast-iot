"""
EN: **Inverse cost recovery on Ma et al.'s discrete benchmark** (Ma, Sun, Ye, Zha & Zhou 2021, §5.1;
    `paper/additional_sources/Learning Cost Functions for Optimal Transport.md`), estimated with this
    repository's **relaxed** (regret) master.

    The benchmark, as the paper specifies it: `n = 100`, ground-truth cost `c*_ij = |(i - j) / n|^p` for
    `p in {0.5, 1, 2, 3}`, an entropic optimal plan at `epsilon in {10, 1, 0.1, 0.01}`, 20 random marginal
    pairs, scored by the relative Frobenius cost error `||c_hat - c*|| / ||c*||`.

    Three things the paper leaves open (its hypothesis class is a free matrix) and that are therefore stated
    here, swept or recorded:

      * **hypothesis class** (`--family`). Our cost is `c_ij = sum_k theta_k Phi_ij,k` with `theta` on the
        simplex, so the class is a feature family:
          `p_basis`    members `t^q`, `t = |i-j|/n`, over the `--ps` grid (the truth is one member)
          `asym_basis` members `|i/n - a*j/n|^q` for `a in {1,2}`, `q in {1,2}`: same truth as `p_basis` at
                       `a=1`, plus three distractors, one of which (`a=2`) is the paper's asymmetric example
          `free_dense` one indicator feature per entry (`K = n^2`), i.e. a free cost matrix
          `hat9`       9 hat/ramp functions of `t` (no member equals the truth; the residual is the basis error)
      * **gauge.** `R(theta) = <c, pi_hat> - OT(c)` is positively homogeneous in the cost, so a family whose
        members differ in magnitude is compared on magnitude as well as on shape. Every feature is therefore
        scaled to unit Frobenius norm, and the recovered scale is reported (`best_scale`). A positive scaling of
        a cost leaves the OT plan unchanged, so this removes exactly the part of `theta` that is not identifiable.
      * **noise model.** The observed plan is the entropic plan itself: no thresholding, no renormalisation.

    Estimator: the relaxed master's `argmin_theta R(theta)`, one solve with every `m*n` edge. `R` is defined for
    any plan, including one that is optimal at no `theta`; it is reported as a reading, not as a certificate.

CN: **在 Ma 等（2021, §5.1）的离散基准上做逆代价恢复**，用本仓库的**松弛**（regret）master。基准按原论文：
    `n=100`、`c*=|(i-j)/n|^p`、`epsilon in {10,1,0.1,0.01}` 的熵正则最优计划、20 组随机 marginal、相对
    Frobenius 代价误差。原论文用自由代价矩阵，故以下三点在此显式声明或扫描：特征族（`--family`）、规范
    （每个特征缩放到单位 Frobenius 范数，代价的正缩放不改变 OT 计划，故只去掉不可辨识的分量，恢复出的尺度
    记为 `best_scale`）、噪声模型（观测计划就是熵正则计划本身，不截断、不归一化）。估计量为松弛 master 的
    `argmin_theta R(theta)`，在全部 `m*n` 条边上求解一次。

    Usage / 用法:
        python -m iot_experiments.studies.related.inverse_cost_recovery.run --family asym_basis --instances 5
        python -m iot_experiments.studies.related.inverse_cost_recovery.run --family p_basis --eps 0.1 0.01
"""

from __future__ import annotations

import dataclasses
import sys
import time
from pathlib import Path

import numpy as np
import ot
import scipy.sparse as sparse
from jaxtyping import Float

_REPO_ROOT = Path(__file__).resolve().parents[4]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from iot.core.edges import EdgeSet  # noqa: E402
from iot.core.features import DenseFeatureTensor, FeatureMap, RadialProfileFeatures  # noqa: E402
from iot.core.problem import InverseOTProblem  # noqa: E402
from iot.core.theta import SimplexTheta  # noqa: E402
from iot.relaxed.master_regret import RegretMasterSolver  # noqa: E402
from iot_experiments.harness.config import RunConfig, build_parser  # noqa: E402
from iot_experiments.harness.preflight import check as preflight_check, cuda_usable  # noqa: E402
from iot_experiments.harness.records import RunLog, RunRecord, format_table, summarise, write_records  # noqa: E402

PROVENANCE: dict[str, object] = {
    "question": "on Ma et al. (2021) §5.1's discrete cost-recovery benchmark, what relative cost error does this "
                "repository's relaxed (regret) master reach, and how does it move with the entropy weight?",
    "protocol": "n=100, c*_ij = |(i-j)/n|^p, pi_hat = sinkhorn_log(mu, nu, c*, eps) with mu,nu ~ Dirichlet(1) on "
                "the n-simplex, --instances draws per cell, the same draws in every cell",
    "estimator": "relaxed master: argmin_theta R(theta), R = <c(theta), pi_hat> - OT(c(theta)); one solve over "
                 "all m*n edges (the observed plan is dense, so there is no edge outside its support to separate)",
    "tie_break": "False (the shipped study path); theta is the repaired primal point",
    "gauge": "each feature scaled to unit Frobenius norm; the cost is then identifiable only up to a positive "
             "scale, which is reported as best_scale and used for cost_rel_scaled",
    "metric": "cost_rel = ||c_hat - c*||_F/||c*||_F; cost_rel_scaled = the same after the optimal positive "
              "rescaling; plan_rel = ||EMD(c_hat) - EMD(c*)||_F/||EMD(c*)||_F",
    "reference": "the same cost metric in the paper is ~1e-4 after 500 iterations (Algorithm 1, free cost matrix, "
                 "20 instances); their non-uniqueness test reports plan error 0.026",
}


@dataclasses.dataclass(frozen=True)
class CostRecoveryConfig(RunConfig):
    """EN: knobs. CN: 旋钮。"""

    study: str = "related_cost_recovery"
    section: str = "related_cost_recovery"
    family: str = "asym_basis"
    n: int = 100
    ps: tuple[float, ...] = (0.5, 1.0, 2.0, 3.0)
    eps: tuple[float, ...] = (10.0, 1.0, 0.1, 0.01)
    instances: int = 20
    master_backend: str = "auto"
    sinkhorn_iterations: int = 100_000
    allow_heavy: bool = False

    def __post_init__(self) -> None:
        """EN: normalise and validate. CN: 归一化与校验。"""
        object.__setattr__(self, "ps", tuple(float(v) for v in self.ps))
        object.__setattr__(self, "eps", tuple(float(v) for v in self.eps))
        if str(self.family) not in {"p_basis", "asym_basis", "free_dense", "hat9"}:
            raise ValueError("--family must be one of p_basis, asym_basis, free_dense, hat9")
        if int(self.n) < 2:
            raise ValueError("--n must be at least 2")
        if int(self.instances) < 1:
            raise ValueError("--instances must be at least 1")
        if any(value <= 0.0 for value in self.eps):
            raise ValueError("--eps values must be positive")

    @classmethod
    def add_study_arguments(cls, parser) -> None:
        """EN: only this study's flags. CN: 只加本研究特有的参数。"""
        parser.add_argument("--family", choices=("p_basis", "asym_basis", "free_dense", "hat9"),
                            default="asym_basis")
        parser.add_argument("--n", type=int, default=100)
        parser.add_argument("--ps", type=float, nargs="+", default=[0.5, 1.0, 2.0, 3.0])
        parser.add_argument("--eps", type=float, nargs="+", default=[10.0, 1.0, 0.1, 0.01])
        parser.add_argument("--instances", type=int, default=20)
        parser.add_argument("--master-backend", choices=("auto", "cupdlpx", "highs"), default="auto")
        parser.add_argument("--sinkhorn-iterations", type=int, default=100_000)
        parser.add_argument("--allow-heavy", action="store_true")


def _backend(requested: str) -> str:
    """
    EN: `auto` -> `cupdlpx` only when a CUDA device is usable, else `highs` (the exact CPU simplex).

        Same rule as `studies/real/faust/arms.py`: `cupdlpx_available()` tests only whether the extension
        imports, which it also does on a host whose driver is too old for its torch build.

    CN: `auto` 仅在 CUDA 设备可用时选 `cupdlpx`，否则用精确 CPU 单纯形 `highs`。判断依据同 `arms.py`。
    """
    choice = str(requested)
    if choice in {"cupdlpx", "highs"}:
        return choice
    return "cupdlpx" if cuda_usable() else "highs"


def _unit_norm(tensor: Float[np.ndarray, "n n k"]) -> Float[np.ndarray, "n n k"]:
    """
    EN: scale every feature column to unit Frobenius norm (the gauge choice; see the module docstring).
    CN: 把每个特征列缩放到单位 Frobenius 范数（规范选择，见模块文档）。
    """
    flat = tensor.reshape(-1, tensor.shape[2])
    norms = np.linalg.norm(flat, axis=0)
    if np.any(norms <= 0.0):
        raise ValueError("a feature column is identically zero")
    return np.ascontiguousarray(tensor / norms[None, None, :], dtype=np.float64)


def _family(family: str, n: int, truth_power: float, ps: tuple[float, ...]) -> tuple[FeatureMap, int | None]:
    """
    EN: the hypothesis class for one truth, plus the index of the member that equals the truth (or `None` when no
        member does, which is the case for the smooth `hat9` basis).
    CN: 针对某个真值的假设类，以及在族内等于真值的成员下标（若族内没有真值 —— `hat9` 即如此 —— 返回 `None`）。
    """
    index = np.arange(n, dtype=np.float64) / float(n)
    source = index[:, None]
    target = index[None, :]
    if family == "p_basis":
        members = [(1.0, q) for q in ps]
        tensor = np.stack([np.abs(source - target) ** float(q) for _a, q in members], axis=2)
        truth = members.index((1.0, float(truth_power)))
        return DenseFeatureTensor(_unit_norm(tensor)), int(truth)
    if family == "asym_basis":
        # EN: the truth is `a = 1` with `q = truth_power`; `a = 2` gives the paper's asymmetric example.
        # CN: 真值是 `a=1` 且 `q=truth_power`；`a=2` 即原论文的非对称例子。
        members = [(a, q) for a in (1.0, 2.0) for q in (1.0, 2.0)]
        tensor = np.stack([np.abs(source - a * target) ** float(q) for a, q in members], axis=2)
        wanted = (1.0, float(truth_power))
        truth = members.index(wanted) if wanted in members else None
        return DenseFeatureTensor(_unit_norm(tensor)), truth
    if family == "free_dense":
        k = n * n
        tensor = np.zeros((n, n, k), dtype=np.float64)
        flat = np.arange(k)
        tensor.reshape(k, k)[flat, flat] = 1.0
        # EN: with indicator features the truth is `c* / sum(c*)`; there is no single named member.
        # CN: 指示特征下真值是 `c*/sum(c*)`，没有"某个命名成员"。
        return DenseFeatureTensor(tensor), None
    if family == "hat9":
        coordinate = index[:, None]
        return RadialProfileFeatures(coordinate, coordinate.copy(), basis_size=8), None
    raise ValueError(f"unknown family {family!r}")


def _dense_cost(features: FeatureMap, theta: Float[np.ndarray, "k"], n: int) -> Float[np.ndarray, "n n"]:
    """EN: materialise the n x n cost at `theta` (n = 100, so this is the cheap way). CN: 物化 n x n 代价。"""
    rows = np.repeat(np.arange(n, dtype=np.int64), n)
    cols = np.tile(np.arange(n, dtype=np.int64), n)
    return np.asarray(features.costs(theta, rows, cols), dtype=np.float64).reshape(n, n)


def _relative(a: np.ndarray, b: np.ndarray) -> float:
    """EN: `||a-b||_F / ||b||_F`. CN: 相对 Frobenius 误差。"""
    denominator = float(np.linalg.norm(b))
    return float(np.linalg.norm(np.asarray(a) - np.asarray(b)) / denominator) if denominator > 0 else float("nan")


def _best_scale(cost: np.ndarray, truth: np.ndarray) -> float:
    """EN: `argmin_{alpha>0} ||alpha*cost - truth||`, the least-squares positive rescaling. CN: 最优正缩放。"""
    denominator = float(np.sum(cost * cost))
    return float(np.sum(cost * truth) / denominator) if denominator > 0 else 1.0


def _regret(cost: Float[np.ndarray, "n n"], plan: Float[np.ndarray, "n n"], mu: Float[np.ndarray, "n"],
            nu: Float[np.ndarray, "n"]) -> float:
    """
    EN: `R = <c, pi_hat> - OT(c)`, with `OT(c)` the exact optimal value at the plan's own marginals.
    CN: `R = <c, pi_hat> - OT(c)`，`OT(c)` 按计划自身的 marginal 计算精确最优值。
    """
    return float(np.sum(cost * plan) - ot.emd2(mu, nu, cost, numItermax=1_000_000))


def _fit(features: FeatureMap, plan: sparse.csr_matrix, *, backend: str
         ) -> tuple[Float[np.ndarray, "k"], float, str]:
    """
    EN: one relaxed-master solve over the full edge set.

        The observed plan is dense, so its support is the whole grid and no edge lies outside it. That is the case
        `solve_inverse_ot_regret` refuses by construction: its mandatory connector set is built for the delta
        master, which does not exist in the regret formulation. The regret master itself is well posed on the full
        grid (for fixed `theta` its optimum is `R(theta) >= 0`, and `f, g` are bounded by the constraint rows), so
        the same `RegretMasterSolver` is called once with every edge. When no edge is outside the support, that one
        call solves the same LP the loop would solve on its first round.

    CN: 在全边集上求解一次松弛 master。观测计划是稠密的，支撑集就是整个网格，没有边在其外 —— 这正是
        `solve_inverse_ot_regret` 按构造拒绝的情形（它为 delta master 构造必须的连接边集，而 regret 形式没有 delta）。
        regret master 自身在全网格上良定（固定 `theta` 时最优值为 `R(theta) >= 0`，`f,g` 由约束行界住），故用同一个
        `RegretMasterSolver` 对全部边求解一次。
    """
    n_source = int(features.n_source)
    n_target = int(features.n_target)
    problem = InverseOTProblem(features=features, theta_set=SimplexTheta(int(features.n_features)), plan=plan)
    rows = np.repeat(np.arange(n_source, dtype=np.int64), n_target)
    cols = np.tile(np.arange(n_target, dtype=np.int64), n_source)
    edges = EdgeSet.from_arrays(rows, cols, n_target=n_target)
    master = RegretMasterSolver(eta_scale=1e-9, tie_break=False, backend=str(backend))
    solution = master.solve(problem, edges)
    return np.asarray(solution.theta, dtype=np.float64), float(solution.regret), str(solution.status)


def main(argv: list[str] | None = None) -> int:
    """EN: entry point. CN: 入口。"""
    parser = build_parser(CostRecoveryConfig)
    args = parser.parse_args(list(argv) if argv is not None else None)
    config = CostRecoveryConfig.from_namespace(args, parser)
    for flag in ("family", "n", "instances", "master_backend", "sinkhorn_iterations"):
        parsed, carried = getattr(args, flag, None), getattr(config, flag, None)
        if parsed is not None and carried is not None and str(parsed) != str(carried):
            raise SystemExit(f"flag --{flag}={parsed!r} did not reach the config ({carried!r})")

    run_dir = config.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    log = RunLog(run_dir / "run.log")
    backend = _backend(str(config.master_backend))
    preflight = preflight_check(run_tiny=False, need_gpu=False)
    preflight.write(run_dir / "preflight.json")
    log.line(f"preflight ok={preflight.ok} mem={preflight.mem_available_gb:.1f} GB")
    log.line(f"family={config.family} n={config.n} instances={config.instances} ps={config.ps} eps={config.eps}")
    log.line(f"master backend: requested={config.master_backend} resolved={backend}")

    n = int(config.n)
    if str(config.family) == "free_dense":
        estimate_gb = (n * n) ** 2 * 8 / 1e9
        log.line(f"free_dense feature tensor: {estimate_gb:.2f} GB")
        if estimate_gb > 6.0 and not bool(config.allow_heavy):
            raise SystemExit(f"free_dense would allocate ~{estimate_gb:.1f} GB; pass --allow-heavy to accept")

    index = np.arange(n, dtype=np.float64) / float(n)
    distance = np.abs(index[:, None] - index[None, :])
    records: list[RunRecord] = []
    for p in config.ps:
        truth = distance ** float(p)
        features, truth_member = _family(str(config.family), n, float(p), tuple(config.ps))
        log.line(f"p={p:g}: K={features.n_features} truth_member={truth_member}")
        for eps in config.eps:
            for instance in range(int(config.instances)):
                # EN: the same `instances` marginal pairs in every (p, eps) cell, so cells differ only in p, eps.
                # CN: 所有 (p, eps) 格子共用同一组 marginal 对，格子之间只差 p 与 eps。
                rng = np.random.default_rng(int(instance))
                mu = rng.dirichlet(np.ones(n))
                nu = rng.dirichlet(np.ones(n))
                plan = np.asarray(ot.sinkhorn(mu, nu, truth, reg=float(eps), method="sinkhorn_log",
                                              numItermax=int(config.sinkhorn_iterations)))
                plan = np.clip(plan, 0.0, None)
                started = time.perf_counter()
                try:
                    theta, regret, status = _fit(features, sparse.csr_matrix(plan), backend=backend)
                except Exception as error:  # EN: record the failure; never substitute a default theta.
                    records.append(RunRecord(
                        study=str(config.study), geometry=f"p{p:g}_eps{eps:g}", arm=str(config.family),
                        k=int(features.n_features), m=n, draw=int(instance), metrics={},
                        config=config.to_dict(), excluded=True, status=f"FAILED {type(error).__name__}",
                        extra={"message": str(error)[:400]}))
                    log.line(f"[p={p:g} eps={eps:g} i={instance}] FAILED {type(error).__name__}: {str(error)[:120]}")
                    continue
                seconds = float(time.perf_counter() - started)
                cost_hat = _dense_cost(features, theta, n)
                alpha = _best_scale(cost_hat, truth)
                induced = np.asarray(ot.emd(mu, nu, cost_hat, numItermax=1_000_000))
                induced_truth = np.asarray(ot.emd(mu, nu, truth, numItermax=1_000_000))
                # EN: `plan_rel` is a Frobenius distance between two plans, and costs of the form |i-j|^p have ties,
                #     so the LP has a whole optimal face and POT may return a different vertex for a scaled cost.
                #     `plan_cost_rel` is therefore also reported: the excess cost of the induced plan under the TRUE
                #     cost, relative to the true optimum. It is 0 exactly when the induced plan is optimal, and it
                #     does not depend on which vertex the solver returned.
                # CN: `plan_rel` 是两个计划之间的 Frobenius 距离，而 `|i-j|^p` 有并列值，LP 的最优面不止一点，
                #     代价缩放后 POT 可能返回不同顶点。故同时报 `plan_cost_rel`：诱导计划在**真代价**下的相对超出量，
                #     为 0 恰好等价于诱导计划最优，且与求解器返回哪个顶点无关。
                plan_value = float(np.sum(truth * induced))
                truth_value = float(ot.emd2(mu, nu, truth, numItermax=1_000_000))
                metrics = {
                    "cost_rel": _relative(cost_hat, truth),
                    "cost_rel_scaled": _relative(alpha * cost_hat, truth),
                    "plan_rel": _relative(induced, induced_truth),
                    "plan_cost_rel": plan_value / truth_value - 1.0 if truth_value > 0 else float("nan"),
                    "regret": regret,
                    "regret_reference": float("nan") if truth_member is None else _regret(
                        _dense_cost(features, np.eye(int(features.n_features))[int(truth_member)], n), plan, mu, nu),
                    "regret_uniform": _regret(
                        _dense_cost(features, np.full(int(features.n_features),
                                                      1.0 / int(features.n_features)), n), plan, mu, nu),
                }
                records.append(RunRecord(
                    study=str(config.study), geometry=f"p{p:g}_eps{eps:g}", arm=str(config.family),
                    k=int(features.n_features), m=n, draw=int(instance), metrics=metrics,
                    config=config.to_dict(),
                    extra={"theta": [float(v) for v in theta], "status": status, "p": float(p),
                           "eps": float(eps), "best_scale": alpha, "truth_member": truth_member,
                           "support_size": int(np.count_nonzero(plan)), "seconds": round(seconds, 3)}))
                log.line(f"[p={p:g} eps={eps:g} i={instance:>2}] cost_rel={metrics['cost_rel']:.3e} "
                         f"scaled={metrics['cost_rel_scaled']:.3e} plan_rel={metrics['plan_rel']:.3e} "
                         f"R={regret:.3e} R_ref={metrics['regret_reference']:.3e} {seconds:.2f}s")

    write_records(run_dir / "records.json", records, provenance=PROVENANCE)
    table = summarise(records, group=("arm", "geometry"),
                      metrics=("cost_rel", "cost_rel_scaled", "plan_rel", "plan_cost_rel", "regret"))
    log.line("\n" + format_table(table, ("arm", "geometry", "n_draws", "cost_rel_mean", "cost_rel_scaled_mean",
                                         "plan_rel_mean", "plan_cost_rel_mean", "regret_mean")))
    log.line(f"wrote {run_dir / 'records.json'} ({len(records)} rows)")
    log.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
