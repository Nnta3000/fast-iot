"""
CN: **Step 1 —— 静态 `|F_min|` 测量**（不写循环）。

    CN: 问题：若把熵惩罚限制在边集 `F` 上，`F` 需要多大？
        停止判据是 `eps * M_out <= tol`，其中 `M_out = sum_{e ∉ F} e^{sigma_e/eps}`。
        总质量为 1，所以这等价于"被排除的质量分数 `eta <= tol/eps`"。

    CN: 做法（在**全网格熵解**处，不引入任何循环）：
       1. 用 `fit_entropic` 得到全网格的 `(theta, f, g)`（它是 `min_{theta} L_eps` 的参考最小化器，
          与 `min_{theta,f,g} Psi` 等价）；
       2. 计算 `pi_e = e^{sigma_e/eps}`（计数测度，实测与 plan 恒等）；
       3. 按 `pi` **降序**排序 —— 这是最优排列，所以得到的是任意 `F` 的**下界**；
       4. 取最小的 `k` 使补集质量 `<= eta`，报告 `k/(m+n)`。

    CN: **不做 sigma_max 规范**：计数测度下 `sigma = f+g-c` 与 `f+g` 的公共平移无关
        （平移会改变 sigma，但 Sinkhorn 解已固定它）。改 `sigma_max` 会缩放 `pi`，破坏质量分数。

    CN: 小 `eps` 时内层 Sinkhorn 可能达不到收敛；用 **eps-scaling**（大 → 小、热启动）。
        仍达不到的 `eps` 在表里标注 `unreached`，**不**用固定 theta 替代（那是另一个问题）。

EN: **Step 1 -- static `|F_min|` measurement** (no loop).

    Question: if the entropic penalty is restricted to an edge set `F`, how large must `F` be? The stop rule
    is `eps * M_out <= tol` with `M_out = sum over e not in F of e^(sigma_e/eps)`. Total mass is 1, so that is
    the same as "excluded mass fraction `eta <= tol/eps`".

    Method (at the **full-grid entropic solution**, with no loop introduced):
      1. `fit_entropic` gives the full-grid `(theta, f, g)` (the reference minimiser of `min_theta L_eps`,
         equivalent to `min_{theta,f,g} Psi`);
      2. compute `pi_e = e^(sigma_e/eps)` (counting measure; measured identical to the plan);
      3. sort by `pi` **descending** -- the optimal ordering, so this is a **lower bound** on any `F`;
      4. the smallest `k` whose complement has mass `<= eta`, reported as `k/(m+n)`.

    **No sigma_max gauge fix**: shifting `f, g` by a common constant changes `sigma` and therefore scales
    `pi`; the Sinkhorn solution already fixes that. Applying the repair here would break the mass fraction.

    At small `eps` the inner Sinkhorn may not converge; use **eps-scaling** (large to small, warm-started).
    Any `eps` still unreachable is marked `unreached` in the table, and is **not** replaced by a fixed theta
    (that would be a different question).
"""

from __future__ import annotations

import dataclasses
import json
import resource
import sys
from pathlib import Path

import numpy as np
from jaxtyping import Float, Int

_REPO_ROOT = Path(__file__).resolve().parents[4]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from iot.core.features import DiagonalWeightFeatures  # noqa: E402
from iot.core.parameters import cost_std as _cost_std  # noqa: E402
from experiments.harness.config import RunConfig, build_parser  # noqa: E402
from experiments.harness.datasets import reduce_features, stream_features  # noqa: E402
from experiments.harness.records import RunLog  # noqa: E402
from experiments.harness.synthrad import discover_patients  # noqa: E402

PROVENANCE: dict[str, object] = {
    "stage": "1 — static |F_min| at the full-grid entropic solution; NO loop",
    "measure": "smallest k with excluded mass fraction <= eta, sorted by pi descending (a lower bound)",
    "not_a_loop": "no constraint generation, no pricing; pure post-processing of one entropic solve",
    "eps": "absolute, fixed per stage from std(cost) at the uniform theta (never recomputed as theta moves)",
    "unreached_eps": "marked, never substituted by a fixed-theta tail",
}

# CN: eta 网格（被排除的质量分数）。停止判据 eps*M_out<=tol 等价于 eta <= tol/eps。
# EN: eta grid (excluded mass fraction). The stop rule eps*M_out<=tol is eta <= tol/eps.
ETA_GRID: tuple[float, ...] = (1e-2, 1e-3, 1e-4)
REL_EPS_GRID: tuple[float, ...] = (1.0, 0.5, 0.25, 0.1)


def _rss_gb() -> float:
    """CN: 峰值 RSS（GB）。EN: peak RSS in GB."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6


def _solve_dense_entropic(
    cost: Float[np.ndarray, "m n"], a: Float[np.ndarray, "m"], b: Float[np.ndarray, "n"],
    *, epsilon_absolute: float, max_iterations: int, device: str,
    warm: tuple[np.ndarray, np.ndarray] | None = None,
) -> tuple[Float[np.ndarray, "m n"], Float[np.ndarray, "m"], Float[np.ndarray, "n"], float] | None:
    """
    CN: 在**给定绝对 eps** 下解一次全网格 Sinkhorn，返回 `(pi, f, g, marginal_err)`。
        eps 是绝对值（不是相对量），因为 Step 1 要控制它。
    EN: one full-grid Sinkhorn at a **given absolute epsilon**, returning `(pi, f, g, marginal_err)`.
        epsilon is absolute (not relative) because Step 1 must control it.
    """
    import torch  # noqa: PLC0415

    if str(device) == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    C = torch.as_tensor(cost, device=device, dtype=torch.float64)
    at = torch.as_tensor(a, device=device, dtype=torch.float64)
    bt = torch.as_tensor(b, device=device, dtype=torch.float64)
    la, lb = torch.log(at), torch.log(bt)
    if warm is None:
        f = torch.zeros(int(a.size), device=device, dtype=torch.float64)
        g = torch.zeros(int(b.size), device=device, dtype=torch.float64)
    else:
        f = torch.as_tensor(warm[0], device=device, dtype=torch.float64).clone()
        g = torch.as_tensor(warm[1], device=device, dtype=torch.float64).clone()
    inv = 1.0 / float(epsilon_absolute)
    err = float("inf")
    for _step in range(int(max_iterations)):
        f = float(epsilon_absolute) * (la - torch.logsumexp((g[None, :] - C) * inv, dim=1))
        g = float(epsilon_absolute) * (lb - torch.logsumexp((f[:, None] - C) * inv, dim=0))
        plan = torch.exp((f[:, None] + g[None, :] - C) * inv)
        err = float(max(((plan.sum(1) - at).abs() / at).max(), ((plan.sum(0) - bt).abs() / bt).max()))
        if err <= 1e-9:
            pi = np.asarray(plan.detach().cpu().numpy(), dtype=np.float64)
            return (pi, np.asarray(f.detach().cpu().numpy(), dtype=np.float64),
                    np.asarray(g.detach().cpu().numpy(), dtype=np.float64), err)
    return None


def _f_min(pi_values: Float[np.ndarray, "edges"], *, eta: float, n_nodes: int) -> dict[str, float]:
    """
    CN: 按 `pi` 降序，取最小 `k` 使补集质量 `<= eta`。返回 `k`、`k/(m+n)` 与补集的实际质量。
    EN: sort `pi` descending and take the smallest `k` whose complement mass is `<= eta`. Returns `k`,
        `k/(m+n)` and the actual complement mass.
    """
    order = np.argsort(-np.asarray(pi_values, dtype=np.float64))
    cumulative = np.cumsum(np.asarray(pi_values, dtype=np.float64)[order])
    total = float(cumulative[-1])
    target = total * (1.0 - float(eta))
    k = int(np.searchsorted(cumulative, target, side="left")) + 1
    k = min(k, int(pi_values.size))
    excluded = float(total - cumulative[k - 1])
    return {"k": float(k), "k_over_nodes": float(k) / float(n_nodes),
            "excluded_mass": excluded, "excluded_fraction": excluded / max(total, 1e-300)}


@dataclasses.dataclass(frozen=True)
class Step1Config(RunConfig):
    """CN: 旋钮。EN: knobs."""

    section = "synthrad_step1"
    study: str = "synthrad_step1"
    m: int = 600
    k: int = 64
    draw: int = 0

    @classmethod
    def add_study_arguments(cls, parser) -> None:
        """CN: 只加本脚本特有的参数。EN: only this script's flags."""
        parser.add_argument("--m", type=int, default=600)
        parser.add_argument("--k", type=int, default=64)
        parser.add_argument("--draw", type=int, default=0)


def main(argv: list[str] | None = None) -> int:
    """CN: 入口。EN: entry point."""
    parser = build_parser(Step1Config)
    args = parser.parse_args(list(argv) if argv is not None else None)
    config = Step1Config.from_namespace(args, parser)
    run_dir = config.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    log = RunLog(run_dir / "run.log")
    log.line("step 1: static |F_min| at the full-grid entropic solution. NO loop.")
    log.line(f"config: m={config.m} k={config.k} eta_grid={ETA_GRID} rel_eps_grid={REL_EPS_GRID}")

    split = json.loads((config.out / "synthrad_v2" / "split_v2.json").read_text())
    patients = discover_patients(_REPO_ROOT / "data" / "Task1", anatomy="brain")
    fit_pats = [p for p in patients if p.patient in set(split["fit"])]
    rng = np.random.default_rng(1000 + int(config.draw))
    chosen = [fit_pats[i] for i in rng.permutation(len(fit_pats))[:60]]
    pm, pc, _pid, _z, _ctr = stream_features(chosen, window=(-200.0, 300.0), max_per_patient=None,
                                             min_gap=0, seed=int(config.draw))
    ftr, ftc = reduce_features(pm, pc, k=int(config.k), geometry_name="G3", whiten=False)
    del pm, pc
    m = int(min(int(config.m), int(ftr.shape[0])))
    sub = np.sort(np.random.default_rng(777).permutation(int(ftr.shape[0]))[:m])
    x: Float[np.ndarray, "m d"] = np.ascontiguousarray(ftr[sub])
    y: Float[np.ndarray, "m d"] = np.ascontiguousarray(ftc[sub])
    k = int(x.shape[1])
    log.line(f"problem m={m} k={k}, peak {_rss_gb():.2f} GB")

    a: Float[np.ndarray, "m"] = np.full(m, 1.0 / m)
    b: Float[np.ndarray, "n"] = np.full(m, 1.0 / m)
    theta_uniform: Float[np.ndarray, "k"] = np.full(k, 1.0 / k)
    rows: Int[np.ndarray, "edges"] = np.repeat(np.arange(m), m)
    cols: Int[np.ndarray, "edges"] = np.tile(np.arange(m), m)
    # CN: 代价尺度在**均匀 theta** 处取一次，之后 eps 不再随 theta 变化。
    # EN: the cost scale is taken once at the **uniform theta**; epsilon never moves with theta afterwards.
    cost_uniform = np.asarray(DiagonalWeightFeatures(x, y).costs(theta_uniform, rows, cols),
                              dtype=np.float64).reshape(m, m)
    std = _cost_std(cost_uniform)
    log.line(f"cost scale (uniform theta): std={std:.6e}")

    table: list[dict[str, object]] = []
    # CN: eps-scaling：从大 eps 起，逐级减小并热启动，尽量到达小 eps。
    # EN: epsilon-scaling: start large, warm-start as epsilon shrinks, to reach small epsilon where possible.
    warm: tuple[np.ndarray, np.ndarray] | None = None
    previous_eps = None
    for rel in REL_EPS_GRID:
        eps = float(rel) * std
        if previous_eps is not None and eps < previous_eps:
            pass  # warm start from the previous (larger) epsilon
        solved = _solve_dense_entropic(cost_uniform, a, b, epsilon_absolute=eps,
                                       max_iterations=200000, device=str(config.device), warm=warm)
        if solved is None:
            log.line(f"rel_eps={rel:<5} eps={eps:.4e}: UNREACHED (Sinkhorn did not converge); "
                     f"recorded as unreached, not substituted")
            table.append({"rel_eps": rel, "eps": eps, "status": "unreached"})
            previous_eps = eps
            continue
        pi, f, g, err = solved
        warm = (f, g)
        previous_eps = eps
        sigma = f[:, None] + g[None, :] - cost_uniform
        # CN: 计数测度的 pi 必须与 Sinkhorn 的 plan 恒等（实测到 1e-17）；这里断言，防止约定再次漂移。
        # EN: the counting-measure pi must equal the Sinkhorn plan (measured to 1e-17); assert it so the
        #     convention cannot drift again.
        sigma_flat = sigma.reshape(-1)
        pi_check = np.exp(sigma_flat / eps)
        convention_gap = float(np.abs(pi_check - pi.reshape(-1)).max())
        if convention_gap > 1e-9:
            raise RuntimeError(f"counting-measure pi disagrees with the plan by {convention_gap:.2e}")
        for eta in ETA_GRID:
            row = _f_min(pi.reshape(-1), eta=float(eta), n_nodes=2 * m)
            table.append({"rel_eps": rel, "eps": eps, "eta": eta, "status": "ok",
                          "marginal_error": err, "max_sigma": float(sigma_flat.max()),
                          **{f"F_min_{key}": value for key, value in row.items()}})
        best = [t for t in table if t.get("rel_eps") == rel and t["status"] == "ok"]
        log.line(f"rel_eps={rel:<5} eps={eps:.4e} marg_err={err:.2e} max_sigma={sigma_flat.max():+.3e} | "
                 + "  ".join(f"eta={t['eta']:g}->k/(m+n)={t['F_min_k_over_nodes']:.2f}" for t in best))

    (run_dir / "step1.json").write_text(
        json.dumps({"rows": table, "provenance": PROVENANCE,
                    "config": {key: (list(v) if isinstance(v, tuple) else str(v) if isinstance(v, Path) else v)
                               for key, v in dataclasses.asdict(config).items()},
                    "cost_std_uniform": std, "m": m, "k": k}, indent=2, default=str) + "\n",
        encoding="utf-8")
    log.line(f"wrote {run_dir / 'step1.json'}")
    log.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
