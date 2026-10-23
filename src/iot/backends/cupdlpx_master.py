"""
CN: 用 HELLO-OT 自带的 CuPDLPx（GPU 一阶 LP 求解器）作为逆 OT master LP 的后端适配器。
    这是一个**新增的可选** `MasterSolver`，不是默认；默认仍是 HiGHS。
EN: An adapter that uses the CuPDLPx first-order GPU LP solver bundled with HELLO-OT as the
    backend for the inverse-OT master LP. This is a **new, optional** `MasterSolver`; HiGHS
    remains the default.

CN: 设计要点 / Design
--------------------
- 它复用既有的 `master.build_master_system`，因此**同一份** LP：等式行在前、`A_ub x <= b_ub`
  在后，变量 `[delta, theta, f, g]`；适配器只把该系统重排成 CuPDLPx 的
  `loadData(A, c, rhs, lb, ub, nEqs)` 约定（前 `nEqs` 行是等式）。
- `min c^T x` 这里取 `c[delta] = -1`，即 `max delta`。
- CuPDLPx 是一阶（PDHG）求解器，返回的是**近似**解；精度由容差控制，最终正确性仍由
  `is_optimal_plan` / `dual_optimality_certificate` 判定，绝不由求解器状态判定。
- 该模块是唯一允许 import `hello_ot` 的 `iot` 适配器之一（另一个是 `grounded.py`），
  这条边界由 `tests/iot_dependency_boundary_test.py` 明确列出。

EN:
- It reuses `master.build_master_system`, so it is the **same** LP (equalities first, then
  `A_ub x <= b_ub`, variables `[delta, theta, f, g]`); the adapter only reshapes it into
  CuPDLPx's `loadData(A, c, rhs, lb, ub, nEqs)` convention (first `nEqs` rows are equalities).
- `min c^T x` with `c[delta] = -1` is `max delta`.
- CuPDLPx is a first-order (PDHG) solver and returns an **approximate** solution; accuracy is
  set by tolerances and final correctness is decided by `is_optimal_plan` /
  `dual_optimality_certificate`, never by the solver status.
- This module is one of the only `iot` adapters allowed to import `hello_ot` (the other is
  `grounded.py`); `tests/iot_dependency_boundary_test.py` lists the boundary explicitly.

CN: 环境说明：`pycupdlpx` 是 `hello_ot` 的 native CUDA 扩展，只有按
    `scripts/create_hello_ot_env.sh` / `scripts/build_native_wheel.sh` 的
    torch 2.7.1+cu118 + nvcc 11.8 组合构建后才可 import；缺失时 `solve` 抛
    `InverseOTError` 并给出安装提示。
EN: Environment note: `pycupdlpx` is HELLO-OT's native CUDA extension and is importable only
    after building with the torch 2.7.1+cu118 + nvcc 11.8 stack documented in
    `scripts/create_hello_ot_env.sh` / `scripts/build_native_wheel.sh`; when it is missing,
    `solve` raises `InverseOTError` with an install hint.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy import sparse

from ..core.edges import EdgeSet
from ..exact.master import (
    MasterSolution,
    TieBreak,
    _MasterSystem,
    build_master_system,
    draw_generic_direction,
)
from ..core.problem import InverseOTError, InverseOTProblem
from ..alternatives.sinkhorn_theta import project_theta
from ..core.theta import SimplexTheta


def _load_pycupdlpx() -> Any:
    """
    CN: 惰性加载 CuPDLPx native 扩展；先试包内路径，再试顶层模块名（便于直接构建测试）。
        扩展链接 libtorch（libc10.so 等），因此先 import torch，确保其 DSO 已加载。
    EN: Lazily load the CuPDLPx native extension; try the packaged path first, then a
        top-level module name (useful for direct-build tests). The extension links against
        libtorch (libc10.so, ...), so import torch first to make those DSOs available.
    """
    try:
        import torch  # noqa: F401  # ensure libtorch DSOs are loaded before the extension
    except ImportError:
        pass
    try:
        from hello_ot._native import pycupdlpx  # type: ignore[import-not-found]

        return pycupdlpx
    except ImportError:
        pass
    try:
        import pycupdlpx  # type: ignore[import-not-found,no-redef]

        return pycupdlpx
    except ImportError as error:
        raise InverseOTError(
            "CuPDLPx (pycupdlpx) is not available. It is HELLO-OT's native CUDA extension; "
            "build it with the documented stack (`bash scripts/create_hello_ot_env.sh`, then "
            "`HELLO_OT_BUILD_NATIVE=1 python -m pip install -e .`) or the local CUDA-12/13 "
            "helper (`python iot_experiments/methods/exact/build_pycupdlpx_cuda13.py`). "
            "HiGHS remains the default master solver."
        ) from error


@dataclass(frozen=True)
class CuPdlpxLp:
    """
    CN: 重排成 CuPDLPx `loadData` 约定后的 master LP：
        `min c^T x s.t. A x = rhs (前 n_eqs 行), A x <= rhs (其余), lb <= x <= ub`。
    EN: The master LP reshaped into CuPDLPx's `loadData` convention:
        `min c^T x s.t. A x = rhs (first n_eqs rows), A x <= rhs (rest), lb <= x <= ub`.
    """

    a_csc: sparse.csc_matrix
    c: np.ndarray
    rhs: np.ndarray
    lb: np.ndarray
    ub: np.ndarray
    n_eqs: int
    layout: Any

    @property
    def n_variables(self) -> int:
        return int(self.c.size)

    @property
    def n_constraints(self) -> int:
        return int(self.rhs.size)


def build_cupdlpx_lp(
    problem: InverseOTProblem, edges: EdgeSet, *,
    epsilon: float = 0.0, generic_direction: np.ndarray | None = None,
) -> CuPdlpxLp:
    """
    CN: 由既有 `build_master_system` 构造 CuPDLPx 形式的同一 master LP（等式行在前）。
        `epsilon > 0` 时把目标改为 `delta + eps*<r, (theta,f,g)>`（r 为整轮固定的通用方向）：
        可行集与最优值不变，只在水面（delta = delta*）上选出唯一顶点——这是 CuPDLPx 版
        `RegularizedMasterSolver` 的 tie-break（单次求解，无需精确面，故适用于一阶 master）。
    EN: Build the same master LP in CuPDLPx form from the existing `build_master_system`
        (equality rows first). With `epsilon > 0` the objective becomes
        `delta + eps*<r, (theta,f,g)>` (r the run-fixed generic direction): the feasible set and the
        optimal value are unchanged, so it only selects a unique vertex on the face delta = delta* --
        the CuPDLPx analogue of `RegularizedMasterSolver`'s tie-break (one solve, no exact face needed,
        hence valid for a first-order master).
    """
    system: _MasterSystem = build_master_system(problem, edges)
    layout = system.layout
    size = int(layout.size)
    # CN: 目标 min c^T x，c[delta] = -1 <=> max delta。
    # EN: `min c^T x` with `c[delta] = -1` is `max delta`.
    c = np.zeros(size, dtype=np.float64)
    c[int(layout.delta)] = -1.0
    if float(epsilon) > 0.0 and generic_direction is not None:
        direction = np.asarray(generic_direction, dtype=np.float64).reshape(-1)
        expected = size - int(layout.theta_start)
        if direction.size != expected:
            raise ValueError(
                f"generic_direction must have {expected} entries (theta, f, g); got {direction.size}"
            )
        scale = float(np.abs(direction).max()) or 1.0
        c[int(layout.theta_start) :] += -float(epsilon) * (direction / scale)
    a_csc = sparse.csc_matrix(sparse.vstack([system.a_eq, system.a_ub], format="csc"))
    rhs = np.ascontiguousarray(np.concatenate([system.b_eq, system.b_ub]), dtype=np.float64)
    n_eqs = int(system.b_eq.size)
    lb = np.array(
        [-np.inf if lo is None else float(lo) for lo, _ in system.bounds], dtype=np.float64
    )
    ub = np.array(
        [np.inf if hi is None else float(hi) for _, hi in system.bounds], dtype=np.float64
    )
    return CuPdlpxLp(
        a_csc=a_csc, c=c, rhs=rhs, lb=lb, ub=ub, n_eqs=n_eqs, layout=layout
    )


@dataclass(frozen=True)
class CuPdlpxMasterSolver:
    """
    CN: CuPDLPx master 求解器（新增可选）。默认纯 `max delta`；`tie_break` 支持 `"none"` 与
        `"regularized"`。
          - `"none"`：纯 `max delta`（不在终止性证明覆盖范围内，可能循环）。
          - `"regularized"`：目标 `delta + eps*<r,(theta,f,g)>`，r 整轮固定、单次求解。可行集与
            最优值不变，只在 delta=delta* 的水面上选出唯一顶点——`"lex_generic"`/`"random"` 的
            一阶可行替代（无需精确面，因而 CuPDLPx 能做）。
        注意：`"lex_generic"`/`"lex"`/`"random"` 需要精确面的两步重解，CuPDLPx 不支持。
    EN: CuPDLPx master solver (optional). Pure `max delta` by default; `tie_break` supports `"none"`
        and `"regularized"`.
          - `"none"`: pure `max delta` (outside the termination proof; may cycle).
          - `"regularized"`: objective `delta + eps*<r,(theta,f,g)>` with r run-fixed, one solve. The
            feasible set and optimal value are unchanged, so it just selects a unique vertex on the face
            delta = delta* -- a first-order-viable stand-in for `lex_generic`/`"random"` (it needs no
            exact face, hence CuPDLPx can do it).
        Note: `"lex_generic"`/`"lex"`/`"random"` need an exact-face two-stage re-solve; CuPDLPx does not
        support them.
    """

    params: dict[str, Any] = field(default_factory=dict)
    warm_start: bool = False
    epsilon: float = 0.0
    random_seed: int = 0
    generic_direction: tuple[float, ...] | None = None

    def solve(
        self,
        problem: InverseOTProblem,
        edges: EdgeSet,
        *,
        tie_break: TieBreak = "none",
    ) -> MasterSolution:
        if str(tie_break) not in {"none", "regularized"}:
            raise InverseOTError(
                "CuPdlpxMasterSolver supports tie_break='none' or 'regularized' only; CuPDLPx has no "
                "lexicographic re-solve (no exact delta-optimal face)."
            )
        direction: np.ndarray | None = None
        epsilon = float(self.epsilon)
        if str(tie_break) == "regularized":
            if epsilon <= 0.0:
                raise InverseOTError("tie_break='regularized' requires epsilon > 0")
            if self.generic_direction is None:
                direction = draw_generic_direction(
                    random_seed=int(self.random_seed),
                    n_features=int(problem.n_features),
                    n_source=int(problem.n_source),
                    n_target=int(problem.n_target),
                )
            else:
                direction = np.asarray(self.generic_direction, dtype=np.float64).reshape(-1)
        else:
            epsilon = 0.0
        pycupdlpx = _load_pycupdlpx()
        lp = build_cupdlpx_lp(problem, edges, epsilon=epsilon, generic_direction=direction)
        solver = pycupdlpx.cupdlpx()
        solver.loadData(
            A=lp.a_csc,
            c=lp.c,
            rhs=lp.rhs,
            lb=lp.lb,
            ub=lp.ub,
            nEqs=int(lp.n_eqs),
        )
        out = solver.solve(self.params)
        primal = out.get("x") if isinstance(out, dict) else None
        if primal is None:
            raise InverseOTError(
                "CuPDLPx returned no primal solution "
                f"(termination_reason={out.get('termination_reason') if isinstance(out, dict) else 'n/a'})"
            )
        # CN: binding 的 out["x"] 是 x_cache_ 上的零拷贝 numpy 视图（base=None）；必须立刻复制，
        #     否则底层缓冲可能在后续分配中被覆写（实测会得到 inf/垃圾值，且非确定性）。
        # EN: the binding's out["x"] is a zero-copy numpy view over x_cache_ (base=None); copy it
        #     immediately, otherwise the buffer can be overwritten by later allocations (observed
        #     non-deterministic inf/garbage).
        x = np.array(primal, dtype=np.float64, copy=True).reshape(-1)
        if x.size != lp.n_variables:
            raise InverseOTError(
                f"CuPDLPx primal has length {x.size}, expected {lp.n_variables}"
            )
        if not np.all(np.isfinite(x)):
            reason = out.get("termination_reason") if isinstance(out, dict) else "n/a"
            rel_primal = out.get("rel_primal_res") if isinstance(out, dict) else None
            rel_gap = out.get("rel_obj_gap") if isinstance(out, dict) else None
            raise InverseOTError(
                "CuPDLPx returned a non-finite primal solution "
                f"(termination_reason={reason}, rel_primal_res={rel_primal}, rel_obj_gap={rel_gap})"
            )
        layout = lp.layout
        k = int(problem.n_features)
        m = int(problem.n_source)
        n = int(problem.n_target)
        theta_start = int(layout.theta_start)
        source_start = int(layout.source_start)
        target_start = int(layout.target_start)
        status = "optimal"
        if isinstance(out, dict) and out.get("termination_reason") is not None:
            status = str(out["termination_reason"])
        theta = np.ascontiguousarray(x[theta_start : theta_start + k], dtype=np.float64)
        # CN: 一阶求解器只近似满足 theta in Theta；把 theta 投影回集合，使循环末尾的
        #     is_optimal_plan（要求严格可行）能够接受它。delta 仍取求解器原值。
        # EN: A first-order solver only approximately satisfies theta in Theta; project theta
        #     back onto the set so the loop's final is_optimal_plan (which requires exact
        #     feasibility) accepts it. delta keeps the solver's value.
        if isinstance(problem.theta_set, SimplexTheta):
            theta = project_theta(theta, problem.theta_set)
        return MasterSolution(
            delta=float(x[int(layout.delta)]),
            theta=theta,
            source_dual=np.ascontiguousarray(
                x[source_start : source_start + m], dtype=np.float64
            ),
            target_dual=np.ascontiguousarray(
                x[target_start : target_start + n], dtype=np.float64
            ),
            status=status,
        )


__all__ = [
    "CuPdlpxLp",
    "CuPdlpxMasterSolver",
    "build_cupdlpx_lp",
]
