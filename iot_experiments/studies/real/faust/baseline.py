"""
EN: **Bit-identity baseline for the `loop_regret` memory change.**

    Purpose: capture, in one deterministic script, every value that a storage-only refactor must leave
    unchanged. Run it before and after the change and diff the two outputs.

    What it records:

      1. **Exact iOT** on a planted instance -- `delta`, `is_optimal_plan`, iteration count, `|F|`,
         `max_violation`. This is the certificate path; nothing here may move.
      2. **Relaxed iOT** on the same instance -- `theta`, `regret`, `lower`, `upper`, `max_sigma_all`,
         iteration count. The storage change must leave all of these bit-identical, including
         `max_sigma_all`, which is the quantity `maximum_slack` produces.
      3. **A FAUST fit** at `n = 800` -- `theta` and `regret`.
      4. **Peak RSS** for a FAUST fit at `n = 2000`, which is what the change is supposed to move.

    Usage:
        .venv/bin/python -m iot_experiments.studies.real.faust.baseline > before.txt
        # ... apply the change ...
        .venv/bin/python -m iot_experiments.studies.real.faust.baseline > after.txt
        diff before.txt after.txt

    CN: **`loop_regret` 内存改动的逐位基线**：记录精确 iOT 的证书值、松弛 iOT 的 `theta`/`regret`/区间/
        `max_sigma_all`、FAUST 拟合结果，以及 `n=2000` 的峰值内存。改动前后各跑一次并 `diff`。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[4]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from iot.core.features import SquaredDifferenceFeatures  # noqa: E402
from iot.core.parameters import (  # noqa: E402
    ExactObjective,
    PruningParameters,
    RelaxedObjective,
    SolverParameters,
)
from iot.exact.loop import solve_inverse_ot  # noqa: E402
from iot.relaxed.loop_regret import solve_inverse_ot_regret  # noqa: E402
from iot_experiments.harness.faust import default_registrations_root, read_ply  # noqa: E402
from iot_experiments.harness.lib.planting import plant_instance  # noqa: E402
from iot_experiments.studies.real.faust.correspondence import (  # noqa: E402
    descriptors,
    fit_theta,
    standardise,
)

ROOT = str(default_registrations_root())


def peak_mb() -> float:
    """CN: 进程峰值 RSS。EN: process peak RSS in MB."""
    for line in Path("/proc/self/status").read_text().splitlines():
        if line.startswith("VmHWM:"):
            return float(line.split()[1]) / 1024.0
    return float("nan")


def _rounded(values: object, digits: int = 12) -> object:
    """CN: 统一浮点格式，便于 `diff`。EN: normalise floats so a plain `diff` is meaningful."""
    if isinstance(values, np.ndarray):
        return [float(v) for v in np.round(values.astype(np.float64), digits)]
    if isinstance(values, float):
        return float(round(values, digits))
    return values


def main() -> int:
    """CN: 入口。EN: entry point."""
    out: dict[str, object] = {}

    # ---- 1/2. exact + relaxed iOT on the same planted instance -------------------------------
    planted = plant_instance(family="diagonal", n=60, point_dimension=4, seed=0)
    problem = planted.problem
    exact = solve_inverse_ot(
        problem,
        objective=ExactObjective(),
        params=SolverParameters(tol=None, gamma=1, max_iterations=200,
                                pruning=PruningParameters(enabled=False)),
    )
    out["exact"] = {
        "delta": _rounded(float(exact.delta)),
        "iterations": int(exact.n_iterations),
        "support_size": int(exact.support_size),
        "max_violation": _rounded(float(exact.final_max_violation)),
        "converged": bool(exact.converged),
        "is_optimal": None if exact.is_optimal is None else bool(exact.is_optimal),
        "relative_gap": None if exact.relative_gap is None else _rounded(float(exact.relative_gap)),
        "theta": _rounded(exact.theta),
    }
    relaxed = solve_inverse_ot_regret(
        problem,
        objective=RelaxedObjective(eps_r_rel=1e-6, check_pot=False, selection="none"),
        params=SolverParameters(tol=None, gamma=1, max_iterations=200,
                                pruning=PruningParameters(enabled=False)),
    )
    out["relaxed"] = {
        "regret": _rounded(float(relaxed.regret)),
        "lower": _rounded(float(relaxed.lower)),
        "upper": _rounded(float(relaxed.upper)),
        "max_sigma_all": _rounded(float(relaxed.max_sigma_all)),
        "iterations": int(relaxed.n_iterations),
        "finite_edges": int(relaxed.finite_edges),
        "converged": bool(relaxed.converged),
        "theta": _rounded(relaxed.theta),
    }

    # ---- 3. a FAUST fit at n = 800 -----------------------------------------------------------
    a_mesh = read_ply(f"{ROOT}/tr_reg_000.ply")
    b_mesh = read_ply(f"{ROOT}/tr_reg_013.ply")
    idx = np.arange(800)
    da, db = standardise(descriptors(a_mesh)[idx], descriptors(b_mesh)[idx])
    fit = fit_theta(SquaredDifferenceFeatures(da, db), rel=None, max_iterations=10)
    out["faust_fit_800"] = {
        "theta": _rounded(np.asarray(fit["theta"])),
        "regret": _rounded(float(fit["regret"])),
        "finite_edges": int(fit["edges"]),
    }

    # ---- 4. peak RSS at n = 2000 -------------------------------------------------------------
    idx2 = np.arange(2000)
    da2, db2 = standardise(descriptors(a_mesh)[idx2], descriptors(b_mesh)[idx2])
    before = peak_mb()
    fit2 = fit_theta(SquaredDifferenceFeatures(da2, db2), rel=None, max_iterations=1)
    out["memory_n2000"] = {
        "peak_mb": round(peak_mb(), 1),
        "growth_mb": round(peak_mb() - before, 1),
        "faust_regret": _rounded(float(fit2["regret"])),
    }

    print(json.dumps(out, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
