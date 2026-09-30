"""
CN: 确定性基准实例：包装 exact 侧 `build_pot_problem` 的语义（先 theta_star，再 source，再
    target；无 mixing），用 POT 求真值方案，并用 `dual_optimality_certificate` 验证后才返回。
EN: Deterministic benchmark instances wrapping the exact-side `build_pot_problem` semantics
    (theta_star first, then source, then target; no mixing), solving the ground-truth plan
    with POT and returning it only after `dual_optimality_certificate` certifies it.

CN: 推理/生成细节（与 `iot_experiments/scratch/common.build_problem` 逐字节一致）：
    `rng = default_rng(seed)`；`theta_star = SimplexTheta(k).sample(rng)`；随后
    `source, target ~ N(0, 1)`；可选 mixing 与 common 相同。标签为历史/非基准的
    `e3_scaling.py` 换种子约定不被采用。
EN: Generation details (byte-identical to `iot_experiments/scratch/common.build_problem`):
    `rng = default_rng(seed)`; `theta_star = SimplexTheta(k).sample(rng)`; then
    `source, target ~ N(0, 1)`; optional mixing exactly as in common. The historical
    `e3_scaling.py` seed convention is deliberately not used.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import ot
from scipy import sparse

from iot import (
    DiagonalWeightFeatures,
    DualCertificate,
    InverseOTProblem,
    SimplexTheta,
    dense_cost_matrix,
    dual_optimality_certificate,
)

from .noisy import noisy_plan

DEFAULT_K: int = 8
DEFAULT_NUM_ITERMAX: int = 2_000_000
NOISE_MULTIPLIERS: tuple[int, ...] = (5, 20, 100)


class BenchmarkError(RuntimeError):
    """
    CN: 基准实例构造失败（真值方案未通过对偶证书，或噪声方案退化）。
    EN: Benchmark instance construction failed (ground-truth plan not certified, or a
        degenerate noisy plan).
    """


@dataclass(frozen=True, eq=False)
class BenchmarkInstance:
    """
    CN: 一个已认证的基准实例。`observed_plan` 是两种方法都会看到的 pi_hat：干净实例下等于
        真值方案，噪声实例下是 M 次抽样得到的经验方案。
    EN: One certified benchmark instance. `observed_plan` is the pi_hat both methods see:
        the ground-truth plan for clean instances, the M-sample empirical plan for noisy
        ones.

    CN: `true_plan`/`true_source_mass`/`true_target_mass` 定义在**同一评估宇宙**上，用于
        R 指标；噪声实例中评估宇宙是丢弃零边际行列后的子问题，`true_plan` 是在该子问题上
        重新求解的干净 OT 方案（见 PROTOCOL.md）。
    EN: `true_plan`/`true_source_mass`/`true_target_mass` live on the same **evaluation
        universe** and define R. For noisy instances the universe is the sub-problem after
        dropping zero-marginal rows/columns, and `true_plan` is the clean OT plan re-solved
        on that sub-problem (see PROTOCOL.md).
    """

    instance_id: str
    seed: int
    n_source: int
    n_target: int
    n_features: int
    mixing_strength: float
    num_itermax: int
    noise_multiplier: int | None
    noise_pairs: int | None
    theta_star: np.ndarray
    features: DiagonalWeightFeatures
    theta_set: SimplexTheta
    observed_plan: np.ndarray
    observed_source_mass: np.ndarray
    observed_target_mass: np.ndarray
    true_plan: np.ndarray
    true_source_mass: np.ndarray
    true_target_mass: np.ndarray
    certificate: DualCertificate
    zero_rows_dropped: int
    zero_cols_dropped: int
    source_indices: np.ndarray
    target_indices: np.ndarray

    def __post_init__(self) -> None:
        if self.observed_plan.shape != (int(self.n_source), int(self.n_target)):
            raise BenchmarkError("observed_plan shape does not match the instance")
        if self.true_plan.shape != (int(self.n_source), int(self.n_target)):
            raise BenchmarkError("true_plan shape does not match the instance")
        if not self.certificate.certified:
            raise BenchmarkError(
                "ground-truth plan failed dual_optimality_certificate; refusing to build "
                f"the instance (max_dual_violation={self.certificate.max_dual_violation:.3e}, "
                f"support_residual={self.certificate.support_residual:.3e})"
            )

    @property
    def is_noisy(self) -> bool:
        """CN: 是否为抽样（噪声）实例。EN: Whether this is a sampled (noisy) instance."""
        return self.noise_multiplier is not None

    def observed_problem(self) -> InverseOTProblem:
        """
        CN: 构造 pi_hat 的 `InverseOTProblem`（边际取自 pi_hat 自身）。
        EN: Build the `InverseOTProblem` for pi_hat (marginals taken from pi_hat itself).
        """
        return InverseOTProblem(
            features=self.features,
            theta_set=self.theta_set,
            plan=sparse.csr_matrix(self.observed_plan),
        )

    def truth_problem(self) -> InverseOTProblem:
        """
        CN: 构造真值方案的 `InverseOTProblem`（用于证书与 R 的参考项）。
        EN: Build the `InverseOTProblem` for the ground-truth plan (certificate and the R
            reference term).
        """
        return InverseOTProblem(
            features=self.features,
            theta_set=self.theta_set,
            plan=sparse.csr_matrix(self.true_plan),
        )


def _identifier(
    *, n: int, k: int, seed: int, mixing_strength: float, noise_multiplier: int | None
) -> str:
    """CN: 确定性实例 id。EN: Deterministic instance id."""
    base = f"n{int(n)}_k{int(k)}_seed{int(seed)}_mix{float(mixing_strength):g}"
    if noise_multiplier is None:
        return base
    return f"{base}_noise{int(noise_multiplier)}x"


def _draw_points(
    *, rng: np.random.Generator, n: int, k: int, mixing_strength: float
) -> tuple[np.ndarray, np.ndarray]:
    """
    CN: 与 `common.build_problem` 完全相同的点云抽样（含可选 mixing）。
    EN: Point-cloud sampling identical to `common.build_problem` (optional mixing).
    """
    source = rng.normal(size=(int(n), int(k)))
    target = rng.normal(size=(int(n), int(k)))
    if float(mixing_strength) > 0.0:
        lower_source = np.tril(rng.standard_normal((int(k), int(k))), -1)
        source = source @ (np.eye(int(k)) + float(mixing_strength) * lower_source)
        lower_target = np.tril(rng.standard_normal((int(k), int(k))), -1)
        target = target @ (np.eye(int(k)) + float(mixing_strength) * lower_target)
    return (
        np.ascontiguousarray(source, dtype=np.float64),
        np.ascontiguousarray(target, dtype=np.float64),
    )


def build_clean_instance(
    n: int,
    k: int = DEFAULT_K,
    seed: int = 0,
    *,
    mixing_strength: float = 0.0,
    num_itermax: int = DEFAULT_NUM_ITERMAX,
) -> BenchmarkInstance:
    """
    CN: 干净控制实例：pi_hat = pi*（POT 求解并用对偶证书验证）。
    EN: Clean control instance: pi_hat = pi* (solved by POT and certified).
    """
    return _build(n=n, k=k, seed=seed, mixing_strength=mixing_strength,
                  num_itermax=num_itermax, noise_multiplier=None)


def build_noisy_instance(
    n: int,
    k: int = DEFAULT_K,
    seed: int = 0,
    *,
    noise_multiplier: int,
    mixing_strength: float = 0.0,
    num_itermax: int = DEFAULT_NUM_ITERMAX,
) -> BenchmarkInstance:
    """
    CN: 抽样实例：从 pi* 抽 M = noise_multiplier * n 个 (i, j)（委托给唯一的
        `benchmark.noisy.noisy_plan`，种子 10_000 + seed），pi_hat = counts / M，边际取自
        pi_hat 本身。零边际的行列会被丢弃（记录数量），详见 PROTOCOL.md。
    EN: Sampled instance: draw M = noise_multiplier * n pairs (i, j) — delegated to the single
        canonical `benchmark.noisy.noisy_plan` (seed 10_000 + seed) — set pi_hat = counts / M
        with marginals taken from pi_hat itself. Zero-marginal rows and columns are dropped
        (counted), see PROTOCOL.md.
    """
    if int(noise_multiplier) not in NOISE_MULTIPLIERS:
        raise ValueError(
            f"noise_multiplier must be one of {NOISE_MULTIPLIERS}; got {noise_multiplier!r}"
        )
    return _build(n=n, k=k, seed=seed, mixing_strength=mixing_strength,
                  num_itermax=num_itermax, noise_multiplier=int(noise_multiplier))


def build_instance(
    n: int,
    k: int = DEFAULT_K,
    seed: int = 0,
    *,
    mixing_strength: float = 0.0,
    num_itermax: int = DEFAULT_NUM_ITERMAX,
    noise_multiplier: int | None = None,
) -> BenchmarkInstance:
    """
    CN: 统一入口；`noise_multiplier=None` 时是干净控制，否则是相应噪声档。
    EN: Unified entry point; `noise_multiplier=None` gives the clean control, otherwise the
        corresponding noise level.
    """
    if noise_multiplier is None:
        return build_clean_instance(
            n, k, seed, mixing_strength=mixing_strength, num_itermax=num_itermax
        )
    return build_noisy_instance(
        n, k, seed, noise_multiplier=int(noise_multiplier),
        mixing_strength=mixing_strength, num_itermax=num_itermax,
    )


def _build(
    *,
    n: int,
    k: int,
    seed: int,
    mixing_strength: float,
    num_itermax: int,
    noise_multiplier: int | None,
) -> BenchmarkInstance:
    """CN: 构造并认证一个实例。EN: Construct and certify one instance."""
    if int(n) < 1 or int(k) < 1:
        raise ValueError("n and k must be >= 1")
    rng = np.random.default_rng(int(seed))
    theta_set = SimplexTheta(int(k))
    theta_star = theta_set.sample(rng)
    source_full, target_full = _draw_points(
        rng=rng, n=int(n), k=int(k), mixing_strength=float(mixing_strength)
    )
    mass_full = np.full(int(n), 1.0 / float(n), dtype=np.float64)
    features_full = DiagonalWeightFeatures(source_full, target_full)
    cost_full = dense_cost_matrix(features_full, theta_star)
    plan_full = ot.emd(mass_full, mass_full, cost_full, numItermax=int(num_itermax))
    if not np.all(np.isfinite(plan_full)):
        raise BenchmarkError("POT returned a non-finite ground-truth plan")

    if noise_multiplier is None:
        source_indices = np.arange(int(n), dtype=np.int64)
        target_indices = np.arange(int(n), dtype=np.int64)
        observed_plan = np.ascontiguousarray(plan_full, dtype=np.float64)
        true_plan = observed_plan
        zero_rows = 0
        zero_cols = 0
        features = features_full
        noise_pairs: int | None = None
    else:
        pairs = int(noise_multiplier) * int(n)
        noisy = noisy_plan(plan_full, pairs, int(seed))
        empirical = np.asarray(noisy.plan, dtype=np.float64)
        positive_rows = np.nonzero(noisy.source_mass > 0.0)[0]
        positive_cols = np.nonzero(noisy.target_mass > 0.0)[0]
        zero_rows = int(n) - int(positive_rows.size)
        zero_cols = int(n) - int(positive_cols.size)
        if positive_rows.size < 1 or positive_cols.size < 1:
            raise BenchmarkError("noisy plan has no positive-marginal rows or columns")
        source_indices = positive_rows.astype(np.int64)
        target_indices = positive_cols.astype(np.int64)
        observed_plan = np.ascontiguousarray(
            empirical[np.ix_(source_indices, target_indices)], dtype=np.float64
        )
        features = DiagonalWeightFeatures(
            source_full[source_indices], target_full[target_indices]
        )
        true_source_mass = np.full(
            source_indices.size, 1.0 / float(source_indices.size), dtype=np.float64
        )
        true_target_mass = np.full(
            target_indices.size, 1.0 / float(target_indices.size), dtype=np.float64
        )
        cost_true = dense_cost_matrix(features, theta_star)
        true_plan = ot.emd(
            true_source_mass, true_target_mass, cost_true, numItermax=int(num_itermax)
        )
        noise_pairs = pairs

    problem = InverseOTProblem(
        features=features, theta_set=theta_set, plan=sparse.csr_matrix(true_plan)
    )
    certificate = dual_optimality_certificate(
        problem, theta_star, num_itermax=int(num_itermax)
    )
    observed_source_mass = np.ascontiguousarray(observed_plan.sum(axis=1), dtype=np.float64)
    observed_target_mass = np.ascontiguousarray(observed_plan.sum(axis=0), dtype=np.float64)
    if noise_multiplier is None:
        true_source_mass = observed_source_mass.copy()
        true_target_mass = observed_target_mass.copy()

    return BenchmarkInstance(
        instance_id=_identifier(
            n=int(source_indices.size if noise_multiplier is not None else n),
            k=int(k),
            seed=int(seed),
            mixing_strength=float(mixing_strength),
            noise_multiplier=noise_multiplier,
        ),
        seed=int(seed),
        n_source=int(features.n_source),
        n_target=int(features.n_target),
        n_features=int(k),
        mixing_strength=float(mixing_strength),
        num_itermax=int(num_itermax),
        noise_multiplier=noise_multiplier,
        noise_pairs=noise_pairs,
        theta_star=np.ascontiguousarray(theta_star, dtype=np.float64),
        features=features,
        theta_set=theta_set,
        observed_plan=observed_plan,
        observed_source_mass=observed_source_mass,
        observed_target_mass=observed_target_mass,
        true_plan=np.ascontiguousarray(true_plan, dtype=np.float64),
        true_source_mass=np.ascontiguousarray(true_source_mass, dtype=np.float64),
        true_target_mass=np.ascontiguousarray(true_target_mass, dtype=np.float64),
        certificate=certificate,
        zero_rows_dropped=int(zero_rows),
        zero_cols_dropped=int(zero_cols),
        source_indices=np.ascontiguousarray(source_indices, dtype=np.int64),
        target_indices=np.ascontiguousarray(target_indices, dtype=np.int64),
    )


__all__ = [
    "BenchmarkError",
    "BenchmarkInstance",
    "DEFAULT_K",
    "DEFAULT_NUM_ITERMAX",
    "NOISE_MULTIPLIERS",
    "build_clean_instance",
    "build_instance",
    "build_noisy_instance",
]
