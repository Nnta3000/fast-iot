"""
CN: **实验 harness** —— 共享配置、注册表、数据加载、求解器、驱动与指标。所有研究都只依赖这一层。

    CN: 与 `src/iot` 一样，这里也开启 beartype 的**运行时**类型校验（`beartype_this_package`）：
        注解不只是文档，形状/类型不匹配会立刻抛错。

EN: **Experiment harness** -- shared config, registry, data loading, solver, driver and metrics. Every study
    depends only on this layer.

    Like `src/iot`, runtime type-checking is enabled through `beartype_this_package`: annotations are not
    merely documentation, and a shape/type mismatch raises immediately.
"""

from beartype.claw import beartype_this_package

beartype_this_package()

__all__: list[str] = []
