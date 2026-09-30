"""
CN: 逆 OT 方法的共享基准：确定性实例、噪声方案构造、离线评估与梯度 runner。
EN: Shared benchmark for inverse-OT methods: deterministic instances, noisy-plan
    construction, offline evaluation, and the gradient runner.

CN: 本包只新增文件；它调用 `iot` 的公开 API 但从不修改它。为避免"仅导入 checkpoint schema"
    就付出 POT/torch 的巨大导入足迹，符号通过 PEP 562 的 `__getattr__` **惰性**加载。
EN: This package only adds files; it calls the public `iot` API but never edits it. Symbols are
    loaded lazily via PEP 562 `__getattr__`, so merely importing the checkpoint schema does not
    pay POT/torch's large import footprint.
"""

from __future__ import annotations

import importlib

_SYMBOL_MODULE: dict[str, str] = {
    # checkpoint
    "InstanceParams": ".checkpoint",
    "RunCheckpoint": ".checkpoint",
    "ThetaCheckpoint": ".checkpoint",
    "read_run": ".checkpoint",
    "write_run": ".checkpoint",
    # evaluate
    "ReferenceScores": ".evaluate",
    "ThetaScore": ".evaluate",
    "evaluate_checkpoints": ".evaluate",
    "instance_from_params": ".evaluate",
    "observed_plan_gap": ".evaluate",
    "reference_scores": ".evaluate",
    "score_run_file": ".evaluate",
    "score_theta": ".evaluate",
    "true_cost_regret": ".evaluate",
    # instances
    "DEFAULT_K": ".instances",
    "DEFAULT_NUM_ITERMAX": ".instances",
    "NOISE_MULTIPLIERS": ".instances",
    "BenchmarkError": ".instances",
    "BenchmarkInstance": ".instances",
    "build_clean_instance": ".instances",
    "build_instance": ".instances",
    "build_noisy_instance": ".instances",
    # noisy
    "NOISY_SEED_OFFSET": ".noisy",
    "NoisyPlan": ".noisy",
    "noisy_plan": ".noisy",
}

__all__ = [
    "BenchmarkError",
    "BenchmarkInstance",
    "DEFAULT_K",
    "DEFAULT_NUM_ITERMAX",
    "InstanceParams",
    "NOISE_MULTIPLIERS",
    "NOISY_SEED_OFFSET",
    "NoisyPlan",
    "ReferenceScores",
    "RunCheckpoint",
    "ThetaCheckpoint",
    "ThetaScore",
    "build_clean_instance",
    "build_instance",
    "build_noisy_instance",
    "evaluate_checkpoints",
    "instance_from_params",
    "noisy_plan",
    "observed_plan_gap",
    "read_run",
    "reference_scores",
    "score_run_file",
    "score_theta",
    "true_cost_regret",
    "write_run",
]


def __getattr__(name: str) -> object:
    """
    CN: 惰性加载 `__all__` 中的符号，避免导入本包就加载 POT/torch。
    EN: Lazily load any symbol in `__all__`, so importing this package does not pull POT/torch.
    """
    module_name = _SYMBOL_MODULE.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = importlib.import_module(module_name, __name__)
    value = getattr(module, name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(__all__)
