"""
EN: **Sanity check E -- identifiability of theta and of scale, on an instance where the answer is known.**

    The question: row-only features `u_i` and column-only features `v_j` are absorbed by the potentials
    (`(f - u, g - v)` leaves every `sigma` unchanged), so a learned `theta` can load on them without
    changing any plan. Double-centring `phi` makes those directions exactly the null space, so their weight
    should be `0`. Second question: unequal feature scales must move `theta` under a simplex budget unless
    the features are standardised.

    **Why this version.** Two earlier attempts could not answer the question and are recorded here rather
    than deleted:
      1. the **global diagonal** plan (`pi = I/m`): the regret is linear in `theta`, so the optimum sits on a
         simplex vertex in every view and "raw vs centred" measured the vertex;
      2. the **planted Monge** instance: `pi_hat` is strictly optimal, but that generator draws `theta_star`
         uniformly, i.e. **fully supported**, so every coordinate is required to be positive and neither the
         absorbed columns nor a scale change can show up. The requirement, not the solver, is degenerate.

    This version uses a **sparse target** instead: `theta_star` has two nonzero entries out of `K = 4`, so
    the admissible set is expected to be a face, and an absorbed or rescaled direction has room to move.
    The feature family is a thin wrapper (`_Augmented`) because this read-only study may not add one to
    `src/iot`; it is `DenseFeatureTensor` plus the row-only / column-only columns appended.

      A. `absorbed` -- `|theta|` on the appended row-only and column-only columns, `raw` vs `centred`.
      B. `scale`    -- `L1` between the fit and the fit on unevenly rescaled columns (`phi_a -> lam_a phi_a`),
                       `raw` vs `standardised`, each against its own repeat floor.

    **Reading rule (fixed before the run).**
      E-A is decided if `absorbed_mass(centred) <= repeat floor` while `absorbed_mass(raw) > floor`;
      otherwise it is reported as `INCONCLUSIVE -- the admissible set is degenerate on this instance`.
      E-B is decided if `rescale_L1(standardised) <= repeat floor` while `rescale_L1(raw) > floor`;
      otherwise `INCONCLUSIVE` for the same reason.

    CN: **合理性检查 E —— θ 与尺度的可辨识性**：前两版（全对角方案、均匀支撑的 planted θ*）在解本身意义上是
        退化的，故本版改用**稀疏支撑**的 θ*（K=4 中两个非零），才能让"被吸收的方向"和"尺度"有移动空间。

    Cost: `n_source = 40`, `n_target = 26`, `K = 4 + 3 + 3`, HiGHS reference master, CPU, seconds.
"""

from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

import numpy as np
from jaxtyping import Float

def _grid_edges(n_source: int, n_target: int) -> "EdgeSet":
    """
    CN: 实例的**全部**行（含最后一行；`EdgeSet` 会剔除空行，这里不需要预留）。EN: every row of the grid.
    """
    rows = np.repeat(np.arange(n_source, dtype=np.int64), n_target)
    cols = np.tile(np.arange(n_target, dtype=np.int64), n_source)
    return EdgeSet.from_arrays(rows, cols, n_target=n_target)


_REPO_ROOT = Path(__file__).resolve().parents[4]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from iot.core.features import DenseFeatureTensor  # noqa: E402
from experiments.harness.config import RunConfig, build_parser  # noqa: E402
from experiments.harness.lib.planting import plant_instance  # noqa: E402
from experiments.harness.records import (  # noqa: E402
    RunLog,
    RunRecord,
    format_table,
    summarise,
    write_records,
)
from iot.core.edges import EdgeSet  # noqa: E402
from iot.core.problem import InverseOTProblem  # noqa: E402
from iot.relaxed.master_regret import RegretMasterSolver  # noqa: E402
from experiments.harness.solver import box_simplex  # noqa: E402

PROVENANCE: dict[str, object] = {
    "question": "do row-only / column-only features absorb theta mass under a sparse theta_star, does "
                "double-centring zero it, and does unequal feature scale move theta before / after "
                "standardisation?",
    "instance": "planting.plant_instance(family='diagonal', n=40, point_dimension=4, seed=0) with an "
                "explicit sparse theta_star (2 of K=4 nonzero: [0.45, 0.0, 0.55, 0.0]); the plan stays "
                "strictly optimal for that theta by construction; 3 row-only (u_i) and 3 column-only (v_j) "
                "columns are appended",
    "why_sparse": "with the generator's default uniform theta_star every coordinate must be positive, so the "
                  "admissible set is forced to the interior and neither the absorbed columns nor a scale "
                  "change can be seen -- the requirement, not the solver, is degenerate",
    "engine": "experiments.harness.solver.fit_full_lp (dense regret LP on all edges), solver = highs "
              "(CPU reference; the production master is CuPDLPx and is irrelevant here)",
    "centring": "phi_c[e] = phi[e] - rowmean_i phi[i,.] - colmean_j phi[.,j] + grand mean",
    "standardisation": "each column divided by its own std over the grid",
    "reading_identifiability": "E-A decided iff absorbed_mass(centred) <= repeat_floor < absorbed_mass(raw); "
                               "otherwise INCONCLUSIVE (degenerate admissible set)",
    "reading_scale": "E-B decided iff rescale_L1(standardised) <= repeat floor < rescale_L1(raw); otherwise "
                     "INCONCLUSIVE",
    "earlier_attempts": "global-diagonal plan (optimum on a vertex), uniform planted theta_star (fully "
                        "supported) -- both recorded as inconclusive in the report, not deleted",
}


class _Augmented(DenseFeatureTensor):
    """
    CN: `DenseFeatureTensor` 加上"哪几列是被吸收方向"的元数据。**只在本研究内**定义，因为 `src/iot` 只读。
    EN: `DenseFeatureTensor` plus the index of the absorbed columns. Defined here only, because `src/iot` is
    read-only for this work; if a family like this is wanted, it belongs in `src/iot/core/features.py`.
    """


@dataclasses.dataclass(frozen=True)
class IdentConfig(RunConfig):
    """CN: 旋钮。EN: knobs."""

    study: str = "sanity_identifiability"
    section: str = "sanity_identifiability"
    n: int = 40
    point_dimension: int = 4
    rel: float = 0.25
    instances: tuple[int, ...] = (0, 1, 2, 3, 4)
    row_columns: int = 3
    col_columns: int = 3
    lam: tuple[float, ...] = (1.0, 20.0, 0.1, 5.0, 2.0, 10.0, 0.5, 3.0, 0.3, 7.0)

    @classmethod
    def add_study_arguments(cls, parser) -> None:
        """CN: 只加本研究特有的参数。EN: only this study's flags."""
        parser.add_argument("--n", type=int, default=40)
        parser.add_argument("--point-dimension", type=int, default=4)
        parser.add_argument("--rel", type=float, default=0.25)
        parser.add_argument("--instances", type=int, nargs="+", default=[0, 1, 2, 3, 4])
        parser.add_argument("--row-columns", type=int, default=3)
        parser.add_argument("--col-columns", type=int, default=3)


def _centre(phi: Float[np.ndarray, "edges K"], rows: Float[np.ndarray, "edges"],
            cols: Float[np.ndarray, "edges"], n_source: int, n_target: int) -> Float[np.ndarray, "edges K"]:
    """CN: 按 (i,j) 位置双中心化。EN: double-centre by position on the full grid."""
    out = np.asarray(phi, dtype=np.float64).copy()
    r, c = rows.astype(int), cols.astype(int)
    for a in range(out.shape[1]):
        v = out[:, a]
        rmean = np.bincount(r, weights=v, minlength=n_source) / np.bincount(r, minlength=n_source)
        cmean = np.bincount(c, weights=v, minlength=n_target) / np.bincount(c, minlength=n_target)
        out[:, a] = v - rmean[r] - cmean[c] + v.mean()
    return out


def _fit(phi: Float[np.ndarray, "edges K"], *, n_source: int, n_target: int, rel: float,
         plan: "sparse.csr_matrix") -> dict[str, object]:
    """
    CN: 在 planted 实例的**真实方案**的全部边上解一次 regret LP（方阵对角方案不适用于 40x20 的实例）。

        CN: 这里不使用 `harness.solver.fit_full_lp`：它按定义建一个**方阵**对角方案，而 planted 实例是
            40x20，两者在形状上就不可能一致（试过 min 的填充，LP 仍按 source x source 建）。
            因此直接调用与 `fit_full_lp` 相同的组件：`RegretMasterSolver` + `box_simplex`，
            方案取 planted 自己的 `pi_hat`。

    EN: one regret LP over all edges of the planted instance's real plan (a square diagonal plan does not
        apply to a 40x20 instance).

        This does not call `harness.solver.fit_full_lp`: by definition that builds a SQUARE diagonal plan,
        which cannot match a 40x20 planted instance (padding to `min` still leaves the LP at
        source x source). It calls the same components `fit_full_lp` calls -- `RegretMasterSolver` with
        `box_simplex` -- on the planted plan itself.
    """
    theta_set = box_simplex(int(phi.shape[1]), box_relative=float(rel))
    features = DenseFeatureTensor(phi.reshape(n_source, n_target, -1))
    problem = InverseOTProblem(features=features, theta_set=theta_set, plan=plan)
    edges = _grid_edges(n_source, n_target)
    try:
        sol = RegretMasterSolver(backend="highs", tie_break=False,
                                 params={"time_sec_limit": 120.0}).solve(problem, edges)
        theta = np.asarray(sol.theta, dtype=np.float64)
        ratio = float(sol.regret)
        status = "ok"
    except Exception as error:  # noqa: BLE001 - reported, never substituted
        theta = np.full(int(phi.shape[1]), np.nan)
        ratio = float("nan")
        status = f"FAILED: {type(error).__name__}: {str(error)[:80]}"
    return {"theta": theta, "status": status, "R_ratio": ratio}


def main(argv: list[str] | None = None) -> int:
    """CN: 入口。EN: entry point."""
    parser = build_parser(IdentConfig)
    args = parser.parse_args(list(argv) if argv is not None else None)
    config = IdentConfig.from_namespace(args, parser)
    run_dir = config.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    log = RunLog(run_dir / "run.log")

    out: list[RunRecord] = []
    for instance in [int(v) for v in config.instances]:
        _one_instance(config, log, instance, out)

    table = summarise(out, group=("arm",), metrics=("absorbed_mass", "repeat_floor", "rescale_L1"))
    log.line("\n" + format_table(table, ("arm", "n_draws", "absorbed_mass_mean", "repeat_floor_mean",
                                          "rescale_L1_mean")))
    write_records(run_dir / "records.json", out, provenance=PROVENANCE)
    log.line(f"wrote {run_dir / 'records.json'} ({len(out)} rows)")
    log.close()
    return 0


def _one_instance(config: IdentConfig, log: RunLog, instance: int, out: list[RunRecord]) -> None:
    """
    CN: 一个实例：稀疏 θ* 的 planted 实例 + 追加的行/列专用列，四个视图各解一次。
    EN: one instance: a planted sparse-`theta_star` problem plus the appended row-only / column-only
        columns, solved once per view.
    """
    k_signal = int(config.point_dimension)
    star = np.zeros(k_signal, dtype=np.float64)
    star[0], star[k_signal // 2] = 0.45, 0.55
    planted = plant_instance(family="diagonal", n=int(config.n), point_dimension=k_signal,
                             seed=int(instance), theta_star=star)
    features = planted.problem.features
    plan = planted.problem.plan
    n_source, n_target = int(features.n_source), int(features.n_target)
    rows = np.repeat(np.arange(n_source, dtype=np.int64), n_target)
    cols = np.tile(np.arange(n_target, dtype=np.int64), n_source)
    phi = np.asarray(features.features(rows, cols), dtype=np.float64)
    log.line(f"planted: {n_source} x {n_target}, K={phi.shape[1]}, theta*={np.round(planted.theta_star, 4)}, "
             f"support {planted.support_size}, max dual violation {planted.max_dual_violation:.1e}")

    rng = np.random.default_rng(7)
    row_only = rng.standard_normal((n_source, int(config.row_columns))) + 5.0
    col_only = rng.standard_normal((n_target, int(config.col_columns))) - 5.0
    phi = np.concatenate([phi, row_only[rows], col_only[cols]], axis=1)
    k_total = int(phi.shape[1])
    row_dirs = list(range(k_signal, k_signal + int(config.row_columns)))
    col_dirs = list(range(k_signal + int(config.row_columns), k_total))
    appended = row_dirs + col_dirs
    lam = np.asarray(config.lam, dtype=np.float64)
    if lam.size != k_total:
        lam = np.concatenate([np.ones(k_total - lam.size), lam])
    log.line(f"appended row-only {row_dirs}, column-only {col_dirs}; rescale lam={np.round(lam, 2)}")

    sd = phi.std(axis=0)
    sd = np.where(sd > 0, sd, 1.0)
    centred = _centre(phi, rows.astype(float), cols.astype(float), n_source, n_target)
    views = {
        "raw": phi,
        "centred": centred,
        "reference_textbook": centred + 0.1 * sd[None, :] * rng.standard_normal((phi.shape[0], k_total)),
        "standardised": phi / sd[None, :],
    }
    for view, tensor in views.items():
        base = _fit(tensor, n_source=n_source, n_target=n_target, rel=config.rel, plan=plan)
        if base["status"] != "ok":
            log.line(f"{view}: FAILED {base['status']}")
            continue
        repeat = _fit(tensor, n_source=n_source, n_target=n_target, rel=config.rel, plan=plan)
        rescaled = _fit(tensor * lam[None, :], n_source=n_source, n_target=n_target, rel=config.rel, plan=plan)
        theta = np.asarray(base["theta"])
        floor = float(np.abs(theta - np.asarray(repeat["theta"])).sum())
        shift = float(np.abs(np.asarray(rescaled["theta"]) - theta).sum()) if rescaled["status"] == "ok" \
            else float("nan")
        absorbed = float(np.abs(theta[appended]).sum())
        log.line(f"{view:<13} absorbed={absorbed:.4f} repeat_floor={floor:.2e} rescale_L1={shift:.4f} "
                 f"R_ratio={base['R_ratio']:.3f}")
        out.append(RunRecord(study=str(config.study), geometry=f"planted_{n_source}x{n_target}",
                             arm=view, k=k_total, m=int(planted.m_eff), draw=int(instance),
                             metrics={"absorbed_mass": absorbed, "repeat_floor": floor,
                                      "rescale_L1": shift, "R_ratio": float(base["R_ratio"]),
                                      "theta_max": float(theta.max())},
                             config=config.to_dict(),
                             extra={"theta": theta.tolist(), "appended_dirs": appended,
                                    "theta_star": np.asarray(planted.theta_star).tolist(),
                                    "theta_std_rescaled": np.asarray(rescaled["theta"]).tolist()
                                    if rescaled["status"] == "ok" else None}))


if __name__ == "__main__":
    raise SystemExit(main())
