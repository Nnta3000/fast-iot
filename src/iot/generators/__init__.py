"""
CN: **真值生成**：用 HELLO-OT 构造已知 theta 的实例，以及精确 OT 内层求解器。
EN: **Ground truth**: HELLO-OT instances with known theta, plus the exact-OT inner solver.
"""

from .grounded import (
    GroundedInstance,
    make_bilinear_psd_grounded_instance,
    make_diagonal_grounded_instance,
)
from .hello_solver import hello_solver_supported, make_hello_solver

__all__ = [
    "GroundedInstance", "hello_solver_supported", "make_bilinear_psd_grounded_instance",
    "make_diagonal_grounded_instance", "make_hello_solver",
]
