"""
CN: **harness 配置骨干** —— 整个实验树**唯一**的命令行解析器。

    设计要点：
      - 基类 `RunConfig` 声明所有**跨实验**的旋钮（输出目录、求解器、扫描后端、设备、draws、seed …），
        每个实验只通过 `StudyConfig` 子类添加**自己的**旋钮，且必须给出 `section` 名字。
      - **严格**：子类参数名与基类（或其它 section）冲突时**立即抛错**；未知的几何/求解器名也抛错。
      - **可追溯**：`RunConfig.to_dict()` 写进每个 `RunRecord`，因此"说不出来源的数字"无法被产出。

    CN: 之所以不用 hydra：本仓库需要的是"能被 diff 的单一解析器 + 冻结的旧树"，而不是配置组合；
        子类 + 冲突检查已经足够，且不引入新依赖。

EN: **harness configuration backbone** -- the **single** argument parser for the whole experiment tree.

    - The base `RunConfig` owns every **cross-cutting** knob (output dir, solver, scan backend, device,
      draws, seed, ...). A study subclasses `StudyConfig` and adds **only its own** knobs, under a
      `section` name.
    - **Strict**: a study flag that collides with a base flag (or another section) raises immediately;
      unknown geometry/solver names also raise.
    - **Traceable**: `to_dict()` is embedded in every `RunRecord`, so a number cannot be produced without
      its configuration.

用法 / EN: usage
    from iot_experiments.harness.config import RunConfig, build_parser
    cfg = RunConfig.from_args(argv)                 # base only
    from studies.geometry.config import GeometryConfig
    cfg = GeometryConfig.from_args(argv)            # base + study
"""

from __future__ import annotations

import argparse
import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from collections.abc import Sequence
from typing import ClassVar, Literal

SolverName = Literal["cupdlpx", "highs"]
ScanName = Literal["hello", "torch", "numpy"]
DeviceName = Literal["auto", "cuda", "cpu"]

_KNOWN_SOLVERS: frozenset[str] = frozenset({"cupdlpx", "highs"})
_KNOWN_SCANS: frozenset[str] = frozenset({"hello", "torch", "numpy"})
_KNOWN_DEVICES: frozenset[str] = frozenset({"auto", "cuda", "cpu"})

_REPO_ROOT = Path(__file__).resolve().parents[2]


class ConfigError(RuntimeError):
    """CN: 配置不合法（重复参数、未知名字、非法取值）。EN: invalid configuration."""


@dataclass(frozen=True)
class RunConfig:
    """
    CN: 所有实验共享的旋钮。子类**不得**重复声明这里已有的字段，否则 `__init_subclass__` 抛错。
    EN: Knobs shared by every experiment. A subclass must not redeclare any of these fields;
        `__init_subclass__` raises when it does.
    """

    # CN: 实验名决定输出目录 `iot_experiments/out/<study>/`。输出**不再**放在 `gpu_bench/` 下：
    #     所有运行结果统一到 `out/`（git-ignored），这样驱动目录与结果目录不再互相纠缠。
    # EN: the study name decides the output directory `iot_experiments/out/<study>/`. Output is **no longer**
    #     nested under `gpu_bench/`: all run results live in `out/` (git-ignored), so driver directories and
    #     result directories stop being entangled.
    study: str = "unnamed"
    out: Path = field(default_factory=lambda: _REPO_ROOT / "iot_experiments" / "out")
    solver: SolverName = "cupdlpx"
    inner_solver: str = "none"
    scan: ScanName = "torch"
    device: DeviceName = "auto"
    draws: tuple[int, ...] = (0,)
    seed: int = 0
    solve_limit_s: float = 300.0
    resume: bool = False
    json_only: bool = False

    # CN: 子类必须声明自己的 section 名，用于报错信息。
    # EN: a study must name its section; used in error messages.
    section: ClassVar[str] = "base"

    def __post_init__(self) -> None:
        from iot.backends.lp_common import cupdlpx_available

        if str(self.solver) not in _KNOWN_SOLVERS:
            raise ConfigError(
                f"unknown solver {self.solver!r}; known: {sorted(_KNOWN_SOLVERS)}"
            )
        if str(self.scan) not in _KNOWN_SCANS:
            raise ConfigError(
                f"unknown scan {self.scan!r}; known: {sorted(_KNOWN_SCANS)}"
            )
        if str(self.device) not in _KNOWN_DEVICES:
            raise ConfigError(
                f"unknown device {self.device!r}; known: {sorted(_KNOWN_DEVICES)}"
            )
        if float(self.solve_limit_s) <= 0.0:
            raise ConfigError(
                f"solve_limit_s must be positive, got {self.solve_limit_s!r}"
            )
        if not self.draws:
            raise ConfigError("draws must be non-empty")
        if str(self.solver) == "cupdlpx" and not cupdlpx_available():
            raise ConfigError(
                "solver='cupdlpx' requested but CuPDLPx is unavailable; use solver='highs' "
                "(CPU reference) or install the native extension"
            )

    # ---------------- serialisation ----------------
    def to_dict(self) -> dict[str, object]:
        """
        CN: 可 JSON 序列化的配置快照（嵌入 `RunRecord`）。路径转字符串。
        EN: JSON-serialisable config snapshot (embedded in `RunRecord`); paths become strings.
        """
        out: dict[str, object] = {}
        for f in dataclasses.fields(self):
            value = getattr(self, f.name)
            if isinstance(value, Path):
                out[f.name] = str(value)
            elif isinstance(value, tuple):
                out[f.name] = list(value)
            else:
                out[f.name] = value
        out["__section__"] = type(self).section
        out["__class__"] = type(self).__name__
        return out

    @property
    def run_dir(self) -> Path:
        """CN: 本次运行的结果目录 `iot_experiments/out/<study>/`。EN: this run's output directory."""
        return Path(self.out) / str(self.study)

    # ---------------- construction ----------------
    @classmethod
    def from_args(cls, argv: Sequence[str] | None = None) -> "RunConfig":
        """
        CN: 解析命令行并构造配置。子类通过 `add_study_arguments` 挂钩添加自己的参数。
        EN: Parse argv and build the config. Subclasses add their own flags via `add_study_arguments`.
        """
        parser = build_parser(cls)
        namespace = parser.parse_args(list(argv) if argv is not None else None)
        return cls.from_namespace(namespace, parser)

    @classmethod
    def add_study_arguments(cls, parser: argparse.ArgumentParser) -> None:
        """
        CN: **子类覆盖此方法**添加自己的参数。默认无操作；基类只声明共享旋钮。
        EN: **Subclasses override this** to add their own flags. No-op by default; the base owns the
            shared knobs only.
        """
        return None

    @classmethod
    def from_namespace(
        cls, namespace: argparse.Namespace, parser: argparse.ArgumentParser
    ) -> "RunConfig":
        """
        CN: 从已解析的 namespace 构造（便于测试与后续直接调用）。
        EN: Build from an already-parsed namespace (useful for tests and direct calls).
        """
        values = {
            f.name: getattr(namespace, f.name)
            for f in dataclasses.fields(cls)
            if hasattr(namespace, f.name)
        }
        if "draws" in values:
            values["draws"] = tuple(int(d) for d in values["draws"])
        if "out" in values:
            values["out"] = Path(values["out"])
        # CN: 子类的私有参数字段以 section_ 前缀存储，这里剥掉前缀。
        # EN: study fields are stored with a `section_` prefix; strip it here.
        for f in dataclasses.fields(cls):
            if f.name in values:
                continue
            prefixed = f"{cls.section}_{f.name}"
            if hasattr(namespace, prefixed):
                values[f.name] = getattr(namespace, prefixed)
        try:
            return cls(**values)
        except TypeError as error:
            parser.error(f"cannot build {cls.__name__}: {error}")


def base_argument_specs() -> dict[str, dict[str, object]]:
    """
    CN: 共享旋钮的 argparse 规格（单一来源，供解析器与冲突检查使用）。
    EN: argparse spec for the shared knobs (single source, used by both parser and collision check).
    """
    return {
        "study": {
            "type": str,
            "default": "unnamed",
            "help": "study name (output subdirectory)",
        },
        "out": {
            "type": Path,
            # CN: 与 dataclass 字段保持同一默认值；两处不一致会让 argparse 悄悄覆盖字段（曾经如此）。
            # EN: keep this identical to the dataclass field; when the two disagreed argparse silently
            #     overrode the field (which is what happened).
            "default": _REPO_ROOT / "iot_experiments" / "out",
            "help": "output root",
        },
        "solver": {
            "choices": sorted(_KNOWN_SOLVERS),
            "default": "cupdlpx",
            "help": "master LP backend",
        },
        "inner_solver": {
            "type": str,
            "default": "none",
            "help": "exact OT inner solver",
        },
        "scan": {
            "choices": sorted(_KNOWN_SCANS),
            "default": "torch",
            "help": "separation scan backend",
        },
        "device": {
            "choices": sorted(_KNOWN_DEVICES),
            "default": "auto",
            "help": "torch device",
        },
        "draws": {"type": int, "nargs": "+", "default": [0], "help": "split draws"},
        "seed": {"type": int, "default": 0, "help": "master seed"},
        "solve_limit_s": {
            "type": float,
            "default": 300.0,
            "help": "per-solve wall cap (seconds)",
        },
        "resume": {
            "action": "store_true",
            "help": "skip configs that already have output",
        },
        "json_only": {
            "action": "store_true",
            "help": "write JSON only, no printed table",
        },
    }


def build_parser(config_cls: type[RunConfig]) -> argparse.ArgumentParser:
    """
    CN: 构造**唯一**的解析器：先加共享旋钮，再加该研究的参数，并在加之前做冲突检查。
    EN: Build the **single** parser: shared knobs first, then the study's, with a collision check.
    """
    parser = argparse.ArgumentParser(
        prog=f"python -m {config_cls.__module__}",
        description=f"harness driver for study '{config_cls.section}'",
    )
    base = base_argument_specs()
    # CN: 若子类用自己的 dataclass 默认值覆盖了某个共享字段（例如 `study="geometry"`），
    #     解析器的默认值必须跟随子类，否则命令行默认值会把它盖掉。
    # EN: when a subclass overrides a shared field's dataclass default (e.g. `study="geometry"`), the parser
    #     default must follow the subclass, otherwise the argparse default silently wins.
    cls_defaults = {f.name: f.default for f in dataclasses.fields(config_cls)
                    if f.default is not dataclasses.MISSING}
    for name, spec in base.items():
        spec = dict(spec)
        if name in cls_defaults:
            spec["default"] = cls_defaults[name]
        parser.add_argument(f"--{name}", **spec)

    before = {action.dest for action in parser._actions}
    # CN: 先在一个**隔离的**解析器上跑一次子类挂钩，用它自己的 option 字符串做冲突检查，
    #     这样报错来自我们而不是 argparse（argparse 的报错看不到 section 名）。
    # EN: run the subclass hook on a throwaway parser first and check its option strings ourselves, so the
    #     error message names the section (argparse's own message would not).
    probe = argparse.ArgumentParser(add_help=False)
    config_cls.add_study_arguments(probe)
    base_flags = {f"--{n}" for n in base}
    clashes = sorted(
        {opt for a in probe._actions for opt in a.option_strings} & base_flags
    )
    if clashes:
        raise ConfigError(
            f"study '{config_cls.section}' redeclares base flag(s) {clashes}; "
            "remove them (the base already declares those knobs)"
        )
    config_cls.add_study_arguments(parser)
    added = {action.dest for action in parser._actions} - before
    # CN: 子类若重新声明基类参数，argparse 会静默覆盖 default —— 这里必须报错。
    # EN: if a subclass redeclares a base flag, argparse silently overrides its default; raise instead.
    reserved = set(base)
    collisions = {d for d in added if d.split("_")[0] in reserved} if added else set()
    hard = {d for d in added if d in reserved}
    if hard:
        raise ConfigError(
            f"study '{config_cls.section}' redeclares base flag(s): {sorted(hard)}"
        )
    if collisions:
        raise ConfigError(
            f"study '{config_cls.section}' declares flag(s) {sorted(collisions)} that share a prefix "
            f"with a base flag; rename them or use a different section name"
        )
    return parser


def resolve_device(device: str) -> str:
    """
    CN: `auto` → 有 CUDA 则 `cuda`，否则 `cpu`。
    EN: `auto` -> `cuda` when available, else `cpu`.
    """
    if device != "auto":
        return str(device)
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:  # pragma: no cover - torch is a hard dependency elsewhere
        return "cpu"


__all__ = [
    "ConfigError",
    "DeviceName",
    "RunConfig",
    "ScanName",
    "SolverName",
    "base_argument_specs",
    "build_parser",
    "resolve_device",
]
