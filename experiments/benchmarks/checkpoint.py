"""
CN: 小型 JSON checkpoint schema，供共享评估器离线读取。每个 run 一个文件，包含实例参数、
    run 级元数据（方法标签、优化器、收敛标志、峰值 RSS、墙钟）与至少 10 个 theta 快照。
EN: Small JSON checkpoint schema for the shared offline evaluator. One file per run, holding
    instance parameters, run-level metadata (method label, optimiser, convergence flag, peak
    RSS, wall time) and at least 10 theta snapshots.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class InstanceParams:
    """
    CN: 复现一个基准实例所需的全部参数；评估器据此重建同一实例（确定性）。
    EN: Everything needed to deterministically rebuild one benchmark instance, so the
        offline evaluator sees the identical problem.
    """

    instance_id: str
    seed: int
    n: int
    n_features: int
    noise_multiplier: int | None
    noise_pairs: int | None
    mixing_strength: float
    num_itermax: int
    n_source: int
    n_target: int
    zero_rows_dropped: int
    zero_cols_dropped: int
    source_mass_total: float
    target_mass_total: float

    def to_dict(self) -> dict[str, Any]:
        """CN: 转 JSON 兼容字典。EN: Convert to a JSON-compatible dict."""
        return {
            "instance_id": self.instance_id,
            "seed": int(self.seed),
            "n": int(self.n),
            "n_features": int(self.n_features),
            "noise_multiplier": None if self.noise_multiplier is None else int(self.noise_multiplier),
            "noise_pairs": None if self.noise_pairs is None else int(self.noise_pairs),
            "mixing_strength": float(self.mixing_strength),
            "num_itermax": int(self.num_itermax),
            "n_source": int(self.n_source),
            "n_target": int(self.n_target),
            "zero_rows_dropped": int(self.zero_rows_dropped),
            "zero_cols_dropped": int(self.zero_cols_dropped),
            "source_mass_total": float(self.source_mass_total),
            "target_mass_total": float(self.target_mass_total),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "InstanceParams":
        """CN: 从字典恢复。EN: Rebuild from a dict."""
        return cls(
            instance_id=str(payload["instance_id"]),
            seed=int(payload["seed"]),
            n=int(payload["n"]),
            n_features=int(payload["n_features"]),
            noise_multiplier=(
                None if payload["noise_multiplier"] is None else int(payload["noise_multiplier"])
            ),
            noise_pairs=None if payload["noise_pairs"] is None else int(payload["noise_pairs"]),
            mixing_strength=float(payload["mixing_strength"]),
            num_itermax=int(payload["num_itermax"]),
            n_source=int(payload["n_source"]),
            n_target=int(payload["n_target"]),
            zero_rows_dropped=int(payload["zero_rows_dropped"]),
            zero_cols_dropped=int(payload["zero_cols_dropped"]),
            source_mass_total=float(payload["source_mass_total"]),
            target_mass_total=float(payload["target_mass_total"]),
        )


@dataclass(frozen=True)
class ThetaCheckpoint:
    """
    CN: 一次 theta 快照；时间戳为 run 开始以来的墙钟秒。
    EN: One theta snapshot; the timestamp is wall-clock seconds since the run started.
    """

    iteration: int
    elapsed_seconds: float
    theta: tuple[float, ...]
    epsilon: float | None
    sinkhorn_marginal_error: float | None
    entropic_gap: float | None

    def to_dict(self) -> dict[str, Any]:
        """CN: 转 JSON 兼容字典。EN: Convert to a JSON-compatible dict."""
        return {
            "iteration": int(self.iteration),
            "elapsed_seconds": float(self.elapsed_seconds),
            "theta": [float(value) for value in self.theta],
            "epsilon": None if self.epsilon is None else float(self.epsilon),
            "sinkhorn_marginal_error": (
                None if self.sinkhorn_marginal_error is None else float(self.sinkhorn_marginal_error)
            ),
            "entropic_gap": None if self.entropic_gap is None else float(self.entropic_gap),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ThetaCheckpoint":
        """CN: 从字典恢复。EN: Rebuild from a dict."""
        return cls(
            iteration=int(payload["iteration"]),
            elapsed_seconds=float(payload["elapsed_seconds"]),
            theta=tuple(float(value) for value in payload["theta"]),
            epsilon=None if payload["epsilon"] is None else float(payload["epsilon"]),
            sinkhorn_marginal_error=(
                None
                if payload["sinkhorn_marginal_error"] is None
                else float(payload["sinkhorn_marginal_error"])
            ),
            entropic_gap=(
                None if payload["entropic_gap"] is None else float(payload["entropic_gap"])
            ),
        )


@dataclass(frozen=True)
class RunCheckpoint:
    """
    CN: 一个 run 的完整记录：实例参数、方法/优化器标签、收敛与峰值 RSS，以及 theta 快照序列。
    EN: The full record of one run: instance params, method/optimiser labels, convergence
        and peak RSS, plus the sequence of theta snapshots.
    """

    params: InstanceParams
    method: str
    theta_parametrisation: str
    optimiser: str
    weight_decay: float
    learning_rate: float
    epsilon_start: float
    epsilon_decay: float
    epsilon_floor: float
    num_outer: int
    sinkhorn_iterations: int
    hardware: str
    threads: dict[str, int]
    converged: bool
    peak_rss_mb: float
    wall_time_seconds: float
    final_sinkhorn_marginal_error: float
    epsilon_floor_reached: bool
    terminal_theta: tuple[float, ...]
    checkpoints: tuple[ThetaCheckpoint, ...]
    stopped_reason: str = "completed"

    def to_dict(self) -> dict[str, Any]:
        """CN: 转 JSON 兼容字典。EN: Convert to a JSON-compatible dict."""
        return {
            "schema": "iot-benchmark-run-v1",
            "params": self.params.to_dict(),
            "method": str(self.method),
            "theta_parametrisation": str(self.theta_parametrisation),
            "optimiser": str(self.optimiser),
            "weight_decay": float(self.weight_decay),
            "learning_rate": float(self.learning_rate),
            "epsilon_start": float(self.epsilon_start),
            "epsilon_decay": float(self.epsilon_decay),
            "epsilon_floor": float(self.epsilon_floor),
            "num_outer": int(self.num_outer),
            "sinkhorn_iterations": int(self.sinkhorn_iterations),
            "hardware": str(self.hardware),
            "threads": {str(key): int(value) for key, value in self.threads.items()},
            "converged": bool(self.converged),
            "peak_rss_mb": float(self.peak_rss_mb),
            "wall_time_seconds": float(self.wall_time_seconds),
            "final_sinkhorn_marginal_error": float(self.final_sinkhorn_marginal_error),
            "epsilon_floor_reached": bool(self.epsilon_floor_reached),
            "terminal_theta": [float(value) for value in self.terminal_theta],
            "checkpoints": [checkpoint.to_dict() for checkpoint in self.checkpoints],
            "stopped_reason": str(self.stopped_reason),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "RunCheckpoint":
        """CN: 从字典恢复。EN: Rebuild from a dict."""
        if payload.get("schema") != "iot-benchmark-run-v1":
            raise ValueError(f"unexpected checkpoint schema {payload.get('schema')!r}")
        return cls(
            params=InstanceParams.from_dict(payload["params"]),
            method=str(payload["method"]),
            theta_parametrisation=str(payload["theta_parametrisation"]),
            optimiser=str(payload["optimiser"]),
            weight_decay=float(payload["weight_decay"]),
            learning_rate=float(payload["learning_rate"]),
            epsilon_start=float(payload["epsilon_start"]),
            epsilon_decay=float(payload["epsilon_decay"]),
            epsilon_floor=float(payload["epsilon_floor"]),
            num_outer=int(payload["num_outer"]),
            sinkhorn_iterations=int(payload["sinkhorn_iterations"]),
            hardware=str(payload["hardware"]),
            threads={str(key): int(value) for key, value in payload["threads"].items()},
            converged=bool(payload["converged"]),
            peak_rss_mb=float(payload["peak_rss_mb"]),
            wall_time_seconds=float(payload["wall_time_seconds"]),
            final_sinkhorn_marginal_error=float(payload["final_sinkhorn_marginal_error"]),
            epsilon_floor_reached=bool(payload["epsilon_floor_reached"]),
            terminal_theta=tuple(float(value) for value in payload["terminal_theta"]),
            checkpoints=tuple(
                ThetaCheckpoint.from_dict(item) for item in payload["checkpoints"]
            ),
            stopped_reason=str(payload.get("stopped_reason", "completed")),
        )


def write_run(path: Path, run: RunCheckpoint) -> None:
    """
    CN: 原子式写出 run checkpoint JSON（先写临时文件再替换）。
    EN: Write a run checkpoint JSON atomically (temp file then replace).
    """
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(run.to_dict(), indent=2) + "\n", encoding="utf-8")
    temporary.replace(destination)


def read_run(path: Path) -> RunCheckpoint:
    """CN: 读取并校验一个 run checkpoint。EN: Read and validate one run checkpoint."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return RunCheckpoint.from_dict(payload)


__all__ = [
    "InstanceParams",
    "RunCheckpoint",
    "ThetaCheckpoint",
    "read_run",
    "write_run",
]
