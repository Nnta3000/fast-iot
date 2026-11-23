"""
EN: **Diagnostic on the K = 2n reversal**: is the joint fit's collapse a normalisation artefact?

    Three questions, in order:

      1. **Degeneracy.** For the fitted θ at K=80 (and the K=10 reference), which bump bins carry weight, and how
         much *observed* mass of the training plans falls in those bins? If the weighted bins carry almost none, the
         cost is ~0 on the observed edges and the regret is zero by construction.
      2. **A weight floor** `theta_k >= tau/K` (tau = 0.1, 0.3), the same Θ for every arm.
      3. **A stage-2 selection**: among candidates whose *exact* training regret is within `eta` of the best, take the
         one closest to uniform. `eta = fraction * F(uniform)`; reported fractions are fixed in advance.

    Both fixes are reported for arms B and C at K=80, L in {5, 20}, 10 seeds, with the spread across seeds, plus the
    planted-oracle row. No τ or η is chosen by looking at which arm wins.

CN: **关于 K=2n 反转的诊断**：联合拟合的塌缩是不是归一化造成的假象？依次问三件事：(1) 退化 —— 在 K=80（以及 K=10
    参考）下，拟合出的 θ 把权重放在哪些鼓包箱（bump bin），这些箱承载多少**观测**质量；若承载几乎为零，则代价在观测边上
    约为 0，regret 按构造为零。(2) 权重下限 `theta_k >= tau/K`（tau=0.1, 0.3），所有臂共用同一 Θ。(3) 第二阶段选择：在
    **精确**训练 regret 落在最优值 `eta` 以内的候选中，取最接近均匀者；`eta = fraction * F(uniform)`，比例预先固定。
    两个修正都在 K=80、L∈{5,20}、10 seeds 下报告 B 与 C，并给出跨种子离散度与 oracle 行。不以"哪个臂赢"来选 τ 或 η。
"""

from __future__ import annotations

import argparse
import dataclasses
import time
from typing import cast
from pathlib import Path

import numpy as np
import ot
import scipy.sparse as sparse

from iot.core.theta import LinearThetaConstraints
from iot.sparse.bundle_multi import solve_bundle_multi
from iot.sparse.features import DistanceProfileFeatures, RampFeatures
from iot_experiments.harness.records import RunLog, RunRecord, format_table, summarise, write_records
from iot_experiments.studies.iot.synthetic.multi_observation_sparse.arms import (
    Readings,
    _observation,
    _readings,
    sample_discriminating_theta,
    _value,
    planted_theta,
)

OUT = Path("iot_experiments/out/multi_observation_sparse/diagnose")


@dataclasses.dataclass(frozen=True)
class FloorSimplex:
    """
    EN: the simplex with a per-coordinate floor: `lo <= theta_k <= 1`, `sum(theta) = 1`.

        `lo = tau / K`. Deliberately **no upper cap** (unlike `box_simplex`, whose cap `1/(tau K)` could bind).

    CN: 带逐坐标下限的单纯形：`lo <= theta_k <= 1`，`sum=1`，`lo = tau/K`。刻意**不设上界**（`box_simplex` 的上界
        `1/(tau K)` 可能起作用）。
    """

    dimension: int
    tau: float

    def sample(self, rng: np.random.Generator) -> np.ndarray:
        """EN: not used, present for the Protocol. CN: 协议要求，未使用。"""
        values = rng.dirichlet(np.ones(int(self.dimension)))
        return self.project(values)

    def contains(self, theta: np.ndarray, *, tol: float = 1e-9) -> bool:
        """EN: membership test. CN: 成员判定。"""
        vector = np.asarray(theta, dtype=np.float64).reshape(-1)
        return bool(vector.size == int(self.dimension) and vector.min() >= self.lo() - tol
                    and vector.max() <= 1.0 + tol and abs(vector.sum() - 1.0) <= tol)

    def lo(self) -> float:
        """EN: the floor. CN: 下限。"""
        return float(self.tau) / float(self.dimension)

    def project(self, theta: np.ndarray) -> np.ndarray:
        """EN: Euclidean projection onto the floor simplex (bisection on the shift). CN: 投影到下限单纯形（对平移量二分）。"""
        vector = np.asarray(theta, dtype=np.float64).reshape(-1)
        lo = self.lo()
        low, high = float(vector.min() - 1.0), float(vector.max())
        for _ in range(80):
            middle = 0.5 * (low + high)
            shifted = np.clip(vector + middle, lo, 1.0)
            if shifted.sum() > 1.0:
                high = middle
            else:
                low = middle
        return np.clip(vector + 0.5 * (low + high), lo, 1.0)

    def linear_constraints(self) -> LinearThetaConstraints:
        """EN: the LP view. CN: LP 视图。"""
        k = int(self.dimension)
        return LinearThetaConstraints(lower=np.full(k, self.lo()), upper=np.ones(k),
                                      a_eq=np.ones((1, k)), b_eq=np.ones(1))


def _uniform(k: int) -> np.ndarray:
    """EN: the neutral point. CN: 中性点。"""
    return np.full(int(k), 1.0 / float(k))


def _float_field(info: dict[str, object], key: str) -> float:
    """EN: read a numeric diagnostic out of the open `info` dict, failing loudly. CN: 从开放的 `info` 字典读一个数值诊断，类型不对就报错。"""
    value = info.get(key)
    if not isinstance(value, (int, float)):
        raise TypeError(f"diagnostic {key!r} is not numeric: {value!r}")
    return float(value)


def _fit(plans: list[sparse.csr_matrix], maps: list, *, theta_set, stage2_eta: float | None = None,
         iterations: int = 40) -> tuple[np.ndarray, float, dict[str, object]]:
    """
    EN: one bundle fit, optionally followed by the stage-2 selection.

        Stage-2 candidates are the θ's the outer loop queried plus the uniform point; each is **re-evaluated
        exactly**, kept only if `F <= F_best + eta * F(uniform)`, and the one closest to uniform (L1) is returned.

    CN: 一次 bundle 拟合，可选地接第二阶段选择。候选为外层循环查询过的 θ 加均匀点；每个都**精确重算**，只保留
        `F <= F_best + eta*F(uniform)` 者，返回其中最接近均匀（L1）的一个。
    """
    base = solve_bundle_multi(plans, maps, theta_set=theta_set, tol=1e-6, max_iterations=int(iterations),
                              inner_tol=1e-9)
    fit_info: dict[str, object] = {"stage2": "off", "gap": float(base.gap),
                                   "iterations": len(base.iterations), "lower": float(base.lower),
                                   "stopped_at_zero": bool(base.stopped_at_zero)}
    if stage2_eta is None:
        return np.asarray(base.theta), float(base.value), fit_info
    k = int(base.theta.size)
    neutral = _uniform(k)
    # EN: the outer loop already evaluated F exactly at every θ it queried, so those pairs are free candidates;
    #     only the uniform point costs one extra evaluation.
    # CN: 外层循环已在每个查询过的 θ 上精确算过 F，故这些（θ,F）是免费候选；只有均匀点需要额外一次求值。
    candidates = [(np.asarray(iteration.theta, dtype=np.float64), float(iteration.value))
                  for iteration in base.iterations]
    f_uniform = float(_value(neutral, maps, plans))
    candidates.append((neutral, f_uniform))
    filtered = [(l1, value, candidate) for candidate, value in candidates
                if theta_set.contains(candidate, tol=1e-8)
                for l1 in (float(np.abs(candidate - neutral).sum()),)]
    if not filtered:
        return np.asarray(base.theta), float(base.value), {"stage2": "no feasible candidate"}
    budget = float(base.value) + float(stage2_eta) * max(f_uniform, 1e-12)
    admissible = [item for item in filtered if item[1] <= budget]
    chosen = min(admissible, key=lambda item: item[0]) if admissible else min(filtered, key=lambda item: item[1])
    return np.asarray(chosen[2]), float(chosen[1]), {
        "stage2": "uniform", "eta_fraction": float(stage2_eta),
        "eta_absolute": float(stage2_eta) * f_uniform, "f_uniform": f_uniform,
        "n_candidates": len(filtered), "n_admissible": len(admissible), "l1_to_uniform": chosen[0],
        **fit_info}


@dataclasses.dataclass(frozen=True)
class RampMassReading:
    """
    EN: how the observed hinge mass and the fitted weights are distributed over the ramp columns.

        `mass_share` is `sum_{k: theta_k>0} m_k / sum_k m_k`; `weight_on_zero_mass` is `sum_{k: m_k=0} theta_k`.
        The second reading is the one that detects the hole: a ramp that is zero on every observed edge carries no
        mass, so weight placed there is invisible in training and only shows up out of sample.

    CN: 观测 hinge 质量与拟合权重在 ramp 列上的分布。`mass_share` 为 `sum_{k: theta_k>0} m_k / sum_k m_k`；
        `weight_on_zero_mass` 为 `sum_{k: m_k=0} theta_k`。后者才是检测"洞"的读数：在所有观测边上为零的 ramp
        不承载质量，放在其上的权重在训练中不可见，只会在样本外显现。
    """

    mass_share: float
    zero_mass_weighted: int
    weight_on_zero_mass: float


def _ramp_mass_share(theta: np.ndarray, maps: list, plans: list[sparse.csr_matrix]) -> RampMassReading:
    """
    EN: observed hinge mass per ramp and the two readings above.

        For the hinge basis the columns are not a partition of unity, so "mass per bin" is defined as
        `m_k = sum_l sum_edges phi_ij,k pi_hat_l,ij`.

    CN: 每条 ramp 的观测 hinge 质量及上面两个读数。hinge 基的列不是单位分解，故"每箱质量"定义为
        `m_k = sum_l sum_edges phi_ij,k pi_hat_l,ij`。
    """
    mass = np.zeros(int(np.asarray(theta).size), dtype=np.float64)
    for features, plan in zip(maps, plans, strict=True):
        coo = plan.tocoo()
        rows = coo.row.astype(np.int64)
        cols = coo.col.astype(np.int64)
        mass += features.features(rows, cols).T @ coo.data.astype(np.float64)
    weights = np.asarray(theta, dtype=np.float64)
    weighted = weights > 1e-6
    share = float(mass[weighted].sum() / mass.sum()) if mass.sum() > 0 else float("nan")
    return RampMassReading(
        mass_share=share,
        zero_mass_weighted=int(np.sum(weighted & (mass <= 0.0))),
        weight_on_zero_mass=float(weights[mass <= 0.0].sum()),
    )


def _bin_mass(maps: list, plans: list[sparse.csr_matrix], bumps: int) -> np.ndarray:
    """
    EN: observed mass per bump bin: each support edge goes to its nearest bump centre, mass normalised per plan.

    CN: 每个鼓包箱的观测质量：每条支撑边归到最近的鼓包中心，质量按计划归一。
    """
    centres = np.linspace(0.0, 1.0, int(bumps))
    mass = np.zeros(int(bumps), dtype=np.float64)
    for features, plan in zip(maps, plans, strict=True):
        coo = plan.tocoo()
        difference = features.source_points[coo.row.astype(np.int64)] - features.target_points[coo.col.astype(np.int64)]
        spacing = np.sqrt(np.einsum("ed,ed->e", difference, difference)) / float(features.scale)
        bins = np.abs(spacing[:, None] - centres[None, :]).argmin(axis=1)
        np.add.at(mass, bins, coo.data.astype(np.float64) / float(plan.sum()))
    return mass / max(1, len(plans))


def _ramp_features(points_rng: np.random.Generator, n: int, ramps: int, thresholds: tuple[float, ...]
                   ) -> RampFeatures:
    """EN: a fresh window on the fixed basis: quantile thresholds, first-difference columns. CN: 固定基上的新窗口：分位阈值、一阶差分列。"""
    return RampFeatures(points_rng.random((n, 2)), points_rng.random((n, 2)), ramps=int(ramps),
                        seed=30_000, thresholds=thresholds, threshold_rule="quantile", differences=True)


def _ramp_cell(n: int, ramps: int, n_observations: int, seed: int
               ) -> tuple[list[RampFeatures], list[sparse.csr_matrix], RampFeatures, sparse.csr_matrix,
                          np.ndarray, int]:
    """
    EN: one ramp cell, with `theta*` chosen by the pre-registered rejection rule (see
        `arms.sample_discriminating_theta`): fresh instances only, neutral plan reproduces at most half.

    CN: 一个 ramp 格子，`theta*` 由预先注册的拒绝规则选取（见 `arms.sample_discriminating_theta`）：只用全新实例，
        中性计划最多复现一半。
    """
    template_rng = np.random.default_rng((30_000 + seed, 0))
    template = RampFeatures(template_rng.random((n, 2)), template_rng.random((n, 2)), ramps=int(ramps),
                            seed=30_000 + seed, threshold_rule="quantile", differences=True)
    thresholds = tuple(template.thresholds)
    planted, draws = sample_discriminating_theta(
        k=int(ramps), n=int(n), seed=(30_000 + seed, 77),
        make_features=lambda rng: _ramp_features(rng, n, ramps, thresholds))
    observations = []
    for window in range(int(n_observations)):
        features = _ramp_features(np.random.default_rng((30_000 + seed, window + 1)), n, ramps, thresholds)
        rows = np.repeat(np.arange(n, dtype=np.int64), n)
        cols = np.tile(np.arange(n, dtype=np.int64), n)
        cost = np.asarray(features.costs(planted, rows, cols)).reshape(n, n)
        plan = np.asarray(ot.emd(_uniform(n), _uniform(n), cost, numItermax=2_000_000))
        observations.append((features, sparse.csr_matrix(plan)))
    held_features = _ramp_features(np.random.default_rng((30_000 + seed, 10_000)), n, ramps, thresholds)
    rows = np.repeat(np.arange(n, dtype=np.int64), n)
    cols = np.tile(np.arange(n, dtype=np.int64), n)
    held_cost = np.asarray(held_features.costs(planted, rows, cols)).reshape(n, n)
    held_plan = sparse.csr_matrix(
        np.asarray(ot.emd(_uniform(n), _uniform(n), held_cost, numItermax=2_000_000)))
    return ([item[0] for item in observations], [item[1] for item in observations],
            held_features, held_plan, planted, draws)


def _cell(n: int, bumps: int, n_observations: int, seed: int) -> tuple[
        list[DistanceProfileFeatures], list[sparse.csr_matrix], DistanceProfileFeatures, sparse.csr_matrix, np.ndarray]:
    """EN: build the training and held-out observations of one cell (bump basis). CN: 构造一个格子的训练与留出观测（鼓包基）。"""
    planted = planted_theta(int(bumps))
    observations = [_observation(n, bumps, planted, seed, window, marginal_mode="uniform", noise=0.0,
                                 outside=False, base_seed=20_000 + seed) for window in range(int(n_observations))]
    maps = [item[0] for item in observations]
    plans = [item[1] for item in observations]
    held_features, held_plan = _observation(n, bumps, planted, seed, 10_000, marginal_mode="uniform", noise=0.0,
                                            outside=False, base_seed=20_000 + seed)
    return maps, plans, held_features, held_plan, planted


def mode_degeneracy(log: RunLog, *, n: int, seeds: list[int]) -> list[RunRecord]:
    """EN: which bins carry weight, and how much observed mass they hold. CN: 哪些箱带权重，以及它们承载多少观测质量。"""
    rows: list[RunRecord] = []
    for bumps in (10, 80):
        for n_observations in (5, 20):
            shares, zero_weighted, weighted_counts, zero_bins_all = [], [], [], []
            for seed in seeds:
                maps, plans, _hf, _hp, _planted = _cell(n, bumps, n_observations, seed)
                theta, value, _ = _fit(plans, maps, theta_set=_simplex(bumps))
                mass = _bin_mass(maps, plans, bumps)
                weighted = np.asarray(theta) > 1e-6
                share = float(mass[weighted].sum())
                shares.append(share)
                weighted_counts.append(int(weighted.sum()))
                zero_weighted.append(int(np.sum(weighted & (mass <= 0.0))))
                zero_bins_all.append(int(np.sum(mass <= 0.0)))
                log.line(f"[degeneracy] K={bumps} L={n_observations} s={seed}: weighted bins={int(weighted.sum())} "
                         f"mass share={share:.4f} zero-mass weighted={int(np.sum(weighted & (mass <= 0.0)))} "
                         f"zero-mass bins={int(np.sum(mass <= 0.0))}/{bumps} F={value:.2e} "
                         f"top bins={np.argsort(theta)[-3:][::-1].tolist()} "
                         f"theta_top={np.sort(theta)[-3:][::-1].round(3).tolist()}")
            log.line(f"[degeneracy] K={bumps} L={n_observations}: mass share {np.mean(shares):.4f}"
                     f"±{np.std(shares):.4f}; zero-mass weighted bins {np.mean(zero_weighted):.1f}; "
                     f"zero-mass bins {np.mean(zero_bins_all):.1f}/{bumps}")
            rows.append(RunRecord(
                study="multi_observation_sparse", geometry=f"degeneracy_K{bumps}_L{n_observations}",
                arm="C_joint", k=int(bumps), m=int(n), draw=len(seeds),
                metrics={"observed_mass_in_weighted_bins": float(np.mean(shares)),
                         "zero_mass_weighted_bins": float(np.mean(zero_weighted)),
                         "zero_mass_bins": float(np.mean(zero_bins_all)),
                         "weighted_bins": float(np.mean(weighted_counts))},
                config={"mode": "degeneracy", "K": bumps, "L": n_observations, "n": n,
                        "Theta": "simplex (no floor)"},
                extra={"seeds": seeds}))
    return rows


def _project(theta_set, vector: np.ndarray) -> np.ndarray:
    """
    EN: project onto Θ: the floor simplex has an exact projection; for the plain simplex, clip and renormalise.
    CN: 投影到 Θ：下限单纯形有精确投影；纯单纯形则裁剪后归一。
    """
    values = np.asarray(vector, dtype=np.float64).reshape(-1)
    if isinstance(theta_set, FloorSimplex):
        return theta_set.project(values)
    clipped = np.clip(values, 0.0, None)
    total = float(clipped.sum())
    return clipped / total if total > 0 else _uniform(values.size)


def _simplex(k: int):
    """EN: the plain simplex as a `LinearThetaSet`. CN: 纯单纯形。"""
    from iot.core.theta import SimplexTheta  # noqa: PLC0415

    return SimplexTheta(int(k))


def _observed_cost_ratio(theta: np.ndarray, maps: list, plans: list[sparse.csr_matrix]) -> float:
    """
    EN: cost of the observed edges under θ, relative to the cost of the same edges under the uniform θ.

        This is the ramp-basis degeneracy reading: ~0 means the fitted cost vanishes where the observations are.

    CN: 观测边在 θ 下的代价，相对同一批边在均匀 θ 下的代价。这是 ramp 基的退化读数：约 0 表示拟合代价在观测处消失。
    """
    total_theta = 0.0
    total_uniform = 0.0
    for features, plan in zip(maps, plans, strict=True):
        coo = plan.tocoo()
        rows = coo.row.astype(np.int64)
        cols = coo.col.astype(np.int64)
        mass = coo.data.astype(np.float64)
        total_theta += float(mass @ features.costs(theta, rows, cols))
        total_uniform += float(mass @ features.costs(_uniform(theta.size), rows, cols))
    return total_theta / total_uniform if total_uniform > 0 else float("nan")


def mode_ramp(log: RunLog, *, n: int, seeds: list[int], ks: list[int], ls: list[int],
              iterations: int = 40) -> list[RunRecord]:
    """
    EN: the ramp-basis reading: neutral baseline, oracle, the joint fit, and averaging with and without the windows
        that carry no information (a zero subgradient at the initial θ).

    CN: ramp 基读数：中性基线、oracle、联合拟合，以及在有/无"无信息窗口"（初始 θ 处零次梯度）两种情况下的平均。
    """
    rows: list[RunRecord] = []
    for ramps in ks:
        for n_observations in ls:
            for seed in seeds:
                maps, plans, held_features, held_plan, planted, draws = _ramp_cell(n, ramps, n_observations, seed)
                readings_o = _readings(planted, planted, held_features, held_plan)
                neutral = _uniform(ramps)
                readings_n = _readings(neutral, planted, held_features, held_plan)
                theta_c, value_c, info_c = _fit(plans, maps, theta_set=_simplex(ramps), iterations=iterations)
                weighted = int(np.sum(np.asarray(theta_c) > 1e-6))
                ratio = _observed_cost_ratio(theta_c, maps, plans)
                share_c = _ramp_mass_share(theta_c, maps, plans)
                readings_c = _readings(theta_c, planted, held_features, held_plan)
                fitted = [_fit([plan], [features], theta_set=_simplex(ramps), iterations=iterations)
                          for plan, features in zip(plans, maps, strict=True)]
                # EN: a window is *uninformative* only if the fit stopped on the first iteration: its observed plan
                #     is already optimal at uniform θ, so the window says nothing about how to leave uniform. A fit
                #     that stops at zero later is a converged fit, not a skipped window.
                # CN: 只有**第一轮**就停止的窗口才无信息：其观测计划在均匀 θ 处已最优，无法说明如何离开均匀点。
                #     后续在某 θ 处到达零次梯度的拟合是收敛，不是被跳过的窗口。
                skipped_flags = [bool(item[2]["stopped_at_zero"])
                                 and int(_float_field(item[2], "iterations")) == 1 for item in fitted]
                informative = [item for item, flag in zip(fitted, skipped_flags, strict=True) if not flag]
                zero_gradient = int(sum(skipped_flags))
                averaged = _project(_simplex(ramps), np.mean(
                    np.stack([item[0] for item in fitted], axis=0), axis=0))
                share_b = _ramp_mass_share(averaged, maps, plans)
                readings_b = _readings(averaged, planted, held_features, held_plan)
                if informative:
                    averaged_i = _project(_simplex(ramps), np.mean(
                        np.stack([item[0] for item in informative], axis=0), axis=0))
                else:
                    averaged_i = _uniform(ramps)
                share_bi = _ramp_mass_share(averaged_i, maps, plans)
                readings_bi = _readings(averaged_i, planted, held_features, held_plan)
                log.line(f"[ramp] K={ramps} L={n_observations} s={seed} draws={draws}: "
                         f"neutral agr={readings_n.agreement:.4f} | "
                         f"C agr={readings_c.agreement:.4f} cost={readings_c.cost_rel:.4f} "
                         f"F={value_c:.2e} gap={_float_field(info_c, 'gap'):.1e} "
                         f"iters={int(_float_field(info_c, 'iterations'))} weighted={weighted} "
                         f"w_on_zero={share_c.weight_on_zero_mass:.3f} | "
                         f"B agr={readings_b.agreement:.4f} (zero-grad windows {zero_gradient}/{len(fitted)}) | "
                         f"B_inf agr={readings_bi.agreement:.4f} zeros={share_bi.zero_mass_weighted} | "
                         f"oracle agr={readings_o.agreement:.4f}")
                share_n = _ramp_mass_share(neutral, maps, plans)
                share_o = _ramp_mass_share(planted, maps, plans)
                diagnostics = {
                    "C_joint": (share_c, _float_field(info_c, "gap"), _float_field(info_c, "iterations"), value_c),
                    "B_averaged": (share_b, float("nan"), float("nan"), float("nan")),
                    "B_informative": (share_bi, float("nan"), float("nan"), float("nan")),
                    "E_neutral": (share_n, float("nan"), float("nan"), float("nan")),
                    "D_oracle": (share_o, float("nan"), float("nan"), float("nan")),
                }
                for arm, theta, readings in (("C_joint", theta_c, readings_c),
                                             ("B_averaged", averaged, readings_b),
                                             ("B_informative", averaged_i, readings_bi),
                                             ("E_neutral", neutral, readings_n),
                                             ("D_oracle", planted, readings_o)):
                    share, gap, iters, objective = diagnostics[arm]
                    rows.append(RunRecord(
                        study="multi_observation_sparse", geometry=f"ramp_K{ramps}_L{n_observations}",
                        arm=arm, k=int(ramps), m=int(n), draw=int(seed),
                        metrics={"agreement": readings.agreement, "cost_rel": readings.cost_rel,
                                 "train_objective": objective,
                                 "outer_gap": gap,
                                 "iterations": iters,
                                 "hit_iteration_cap": float(iters >= float(iterations)),
                                 "observed_mass_in_weighted_bins": share.mass_share,
                                 "mass_share_below_0.2": float(share.mass_share < 0.2),
                                 "weight_on_zero_mass_ramps": share.weight_on_zero_mass,
                                 "zero_mass_weighted_bins": float(share.zero_mass_weighted),
                                 "windows_at_zero_gradient": float(zero_gradient),
                                 "windows_averaged": float(len(fitted) if arm == "B_averaged"
                                                           else len(informative) if arm == "B_informative" else 0),
                                 "theta_star_draws": float(draws),
                                 "observed_cost_ratio": float(ratio) if arm == "C_joint" else float("nan"),
                                 "weighted_ramps": float(weighted) if arm == "C_joint" else float("nan")},
                        config={"mode": "ramp", "K": ramps, "L": n_observations, "n": n,
                                "basis": "difference of hinges, quantile thresholds",
                                "theta_star_rule": "Dirichlet(0.5); fresh-instance neutral agreement <= 0.5",
                                "Theta": "simplex", "iterations": iterations},
                        extra={"theta": [float(v) for v in theta], "theta_star": [float(v) for v in planted]}))
    return rows


def mode_fix(log: RunLog, *, n: int, seeds: list[int], taus: list[float], stage2: list[float],
             iterations: int = 40) -> list[RunRecord]:
    """EN: floor and stage-2, each applied to all arms, for K=80. CN: 下限与第二阶段选择，作用于所有臂，K=80。"""
    rows: list[RunRecord] = []
    bumps = 80
    for n_observations in (5, 20):
        for seed in seeds:
            maps, plans, held_features, held_plan, planted = _cell(n, bumps, n_observations, seed)
            for label, theta_set, eta in (
                    *[("floor", FloorSimplex(bumps, tau), None) for tau in taus],
                    *[("stage2", _simplex(bumps), fraction) for fraction in stage2],
                    ("reference", _simplex(bumps), None)):
                # EN: arm C.
                theta_c, value_c, info_c = _fit(plans, maps, theta_set=theta_set, stage2_eta=eta, iterations=iterations)
                readings_c = _readings(theta_c, planted, held_features, held_plan)
                # EN: arm A (one fit per plan, same Θ and same stage-2 rule) and arm B (their average, projected).
                singles = [_fit([plan], [features], theta_set=theta_set, stage2_eta=eta, iterations=iterations)[0]
                           for plan, features in zip(plans, maps, strict=True)]
                readings_a = [_readings(single, planted, held_features, held_plan) for single in singles]
                averaged = _project(theta_set, np.mean(np.stack(singles, axis=0), axis=0))
                readings_b = _readings(averaged, planted, held_features, held_plan)
                for arm, readings, theta in (("C_joint", readings_c, theta_c),
                                             ("A_single", None, None),
                                             ("B_averaged", readings_b, averaged)):
                    if arm == "A_single":
                        metrics = {"agreement": float(np.mean([r.agreement for r in readings_a])),
                                   "cost_rel": float(np.mean([r.cost_rel for r in readings_a]))}
                        theta_values: list[float] = []
                    else:
                        chosen = cast(Readings, readings)
                        metrics = {"agreement": chosen.agreement, "cost_rel": chosen.cost_rel}
                        theta_values = [float(v) for v in cast(np.ndarray, theta)]
                    rows.append(RunRecord(
                        study="multi_observation_sparse",
                        geometry=f"fix_{label}_L{n_observations}", arm=arm, k=int(bumps), m=int(n), draw=int(seed),
                        metrics=metrics,
                        config={"mode": "fix", "fix": label, "tau": float(theta_set.tau) if isinstance(
                            theta_set, FloorSimplex) else None, "eta_fraction": eta, "L": n_observations,
                            "n": n, "K": bumps, "Theta": ("floor" if isinstance(theta_set, FloorSimplex)
                                                         else "simplex")},
                        extra={"theta": theta_values, "theta_star": [float(v) for v in planted],
                               **{f"stage2_{key}": str(value) if not isinstance(value, float) else value
                                  for key, value in info_c.items()}}))
    # EN: the planted-oracle row, once per L.
    # CN: 每个 L 的 oracle 行一次。
    for n_observations in (5, 20):
        maps, plans, held_features, held_plan, planted = _cell(n, bumps, n_observations, seeds[0])
        readings = _readings(planted, planted, held_features, held_plan)
        rows.append(RunRecord(study="multi_observation_sparse", geometry=f"fix_oracle_L{n_observations}",
                              arm="D_oracle", k=int(bumps), m=int(n), draw=0,
                              metrics={"agreement": readings.agreement, "cost_rel": readings.cost_rel},
                              config={"mode": "fix", "L": n_observations, "n": n, "K": bumps},
                              extra={"theta_star": [float(v) for v in planted]}))
    return rows


def main(argv: list[str] | None = None) -> int:
    """EN: entry point. CN: 入口。"""
    parser = argparse.ArgumentParser(description="diagnose the K=2n reversal")
    parser.add_argument("--mode", required=True, choices=("degeneracy", "fix", "ramp"))
    parser.add_argument("--n", type=int, default=40)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    parser.add_argument("--taus", type=float, nargs="+", default=[0.1, 0.3])
    parser.add_argument("--etas", type=float, nargs="+", default=[0.05, 0.2])
    parser.add_argument("--iterations", type=int, default=40)
    parser.add_argument("--tag", default="")
    parser.add_argument("--ks", type=int, nargs="+", default=[10, 40, 80])
    parser.add_argument("--ls", type=int, nargs="+", default=[5, 20])
    args = parser.parse_args(list(argv) if argv is not None else None)

    out_dir = OUT / (args.tag or args.mode)
    out_dir.mkdir(parents=True, exist_ok=True)
    log = RunLog(out_dir / "run.log")
    started = time.perf_counter()
    if args.mode == "degeneracy":
        records = mode_degeneracy(log, n=int(args.n), seeds=[int(v) for v in args.seeds])
    elif args.mode == "ramp":
        records = mode_ramp(log, n=int(args.n), seeds=[int(v) for v in args.seeds],
                            ks=[int(v) for v in args.ks], ls=[int(v) for v in args.ls],
                            iterations=int(args.iterations))
    else:
        records = mode_fix(log, n=int(args.n), seeds=[int(v) for v in args.seeds],
                           taus=[float(v) for v in args.taus], stage2=[float(v) for v in args.etas],
                           iterations=int(args.iterations))
    log.line(f"{args.mode} done in {time.perf_counter() - started:.1f}s")
    write_records(out_dir / "records.json", records,
                  provenance={"question": "is the K=2n reversal of C_joint a normalisation artefact?",
                              "fixes": "theta_k >= tau/K (no upper cap); stage-2: among candidates with exact "
                                       "F <= F_best + eta*F(uniform), take the one closest to uniform",
                              "eta": "fixed fractions 0.05 and 0.2 of F(uniform); not chosen after seeing a result",
                              "family": "DistanceProfileFeatures (bump basis), uniform marginals, no noise",
                              "held_out": "fresh points, fresh marginals, exact-OT plan at theta*"})
    table = summarise(records, group=("arm", "geometry"),
                      metrics=("agreement", "cost_rel", "observed_mass_in_weighted_bins"))
    log.line("\n" + format_table(table, ("arm", "geometry", "n_draws", "agreement_mean", "cost_rel_mean",
                                         "observed_mass_in_weighted_bins_mean")))
    log.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
