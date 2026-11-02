"""
CN: **一次性迁移脚本** —— 把 `solve_inverse_ot` / `solve_inverse_ot_regret` 的旧关键字
    改成 `objective=` / `params=` 形式。放错地方的东西会被报告出来，不被静默改写。

    CN: 为什么用脚本而不是手改：61 个调用点，手改极易漏掉默认值差异（例如精确循环的 `tol`
        是 `1e-9`、松弛循环的是 `None`），而这种漏改只会表现为"结果有一点不同"。

    CN: 每个调用点的改写规则都在下面的表里，是显式的、可审阅的。

EN: **One-shot migration** -- rewrite `solve_inverse_ot` / `solve_inverse_ot_regret` calls from legacy
    keywords to `objective=` / `params=`.

    Why a script rather than hand edits: 61 call sites, and a missed default (the exact loop's `tol` is
    `1e-9`, the relaxed loop's is `None`) would show up only as "the numbers are slightly different".

    Every rewrite rule is an explicit table below, so the mapping is reviewable.
"""

from __future__ import annotations

import pathlib
import re

# CN: 旧关键字 -> (目标对象, 字段名)。目标对象：P = params, O = objective, S = params.scan, R = params.pruning
EXACT_MAP: dict[str, tuple[str, str]] = {
    "tol": ("P", "tol"),
    "gamma": ("P", "gamma"),
    "max_iterations": ("P", "max_iterations"),
    "hierarchical": ("P", "hierarchical"),
    "tie_break": ("O", "tie_break"),
    "random_seed": ("O", "random_seed"),
    "validate": ("O", "validate"),
    "validate_max_elements": ("O", "validate_max_elements"),
    "scan_backend": ("S", "backend"),
    "scan_device": ("S", "device"),
    "max_tile_elements": ("S", "max_tile_elements"),
    "scan_max_tile_bytes": ("S", "max_tile_bytes"),
    "prune": ("R", "enabled"),
    "prune_hysteresis": ("R", "hysteresis"),
    "tau_prune": ("R", "tau_prune"),
}

RELAXED_MAP: dict[str, tuple[str, str]] = {
    "tol": ("P", "tol"),
    "gamma": ("P", "gamma"),
    "max_iterations": ("P", "max_iterations"),
    "hierarchical": ("P", "hierarchical"),
    "scan_backend": ("S", "backend"),
    "scan_device": ("S", "device"),
    "scan_max_tile_bytes": ("S", "max_tile_bytes"),
    "prune": ("R", "enabled"),
    "prune_hysteresis": ("R", "hysteresis"),
    "tau_prune": ("R", "tau_prune"),
    "eps_r_rel": ("O", "eps_r_rel"),
    "selection": ("O", "selection"),
    "eta": ("O", "eta"),
    "check_pot": ("O", "check_pot"),
    "initial_extra": ("O", "initial_extra"),
    "identity_rtol": ("O", "identity_rtol"),
    "warm_start": ("O", "warm_start"),
}


def _split_args(text: str) -> list[str]:
    """CN: 在顶层逗号处切分（忽略括号/引号内的逗号）。EN: split on top-level commas."""
    parts: list[str] = []
    depth = 0
    quote: str | None = None
    current: list[str] = []
    for ch in text:
        if quote:
            current.append(ch)
            if ch == quote:
                quote = None
            continue
        if ch in "\"'":
            quote = ch
            current.append(ch)
            continue
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        if ch == "," and depth == 0:
            parts.append("".join(current))
            current = []
            continue
        current.append(ch)
    if "".join(current).strip():
        parts.append("".join(current))
    return parts


def rewrite(text: str) -> tuple[str, list[str]]:
    """
    CN: 返回 `(改写后的文本, 未处理的问题列表)`。只处理 `solve_inverse_ot(` / `solve_inverse_ot_regret(`。
    EN: returns `(rewritten text, list of unhandled problems)`. Only touches `solve_inverse_ot(` and
        `solve_inverse_ot_regret(`.
    """
    problems: list[str] = []
    out: list[str] = []
    index = 0
    pattern = re.compile(r"solve_inverse_ot(_regret)?\(")
    while True:
        match = pattern.search(text, index)
        if match is None:
            out.append(text[index:])
            break
        out.append(text[index:match.end()])
        is_relaxed = match.group(1) is not None
        mapping = RELAXED_MAP if is_relaxed else EXACT_MAP
        # find the matching close paren
        depth = 1
        cursor = match.end()
        while cursor < len(text) and depth:
            if text[cursor] == "(":
                depth += 1
            elif text[cursor] == ")":
                depth -= 1
            cursor += 1
        inner = text[match.end():cursor - 1]
        # CN: 行内注释（`# type: ignore`）会被当成参数的一部分；先剥掉。
        # EN: inline comments (`# type: ignore`) would be swallowed into an argument; strip them first.
        inner = re.sub(r"#[^\n]*", "", inner)
        args = [a.strip() for a in _split_args(inner) if a.strip()]
        positional: list[str] = []
        buckets: dict[str, list[str]] = {"P": [], "O": [], "S": [], "R": []}
        passthrough: list[str] = []
        for arg in args:
            if "=" not in arg.split("(")[0]:
                positional.append(arg)
                continue
            key, value = arg.split("=", 1)
            key = key.strip()
            value = value.strip()
            if key in ("objective", "params", "master", "solver"):
                passthrough.append(f"{key}={value}")
                continue
            if key == "_flat_only":
                problems.append(f"unexpected _flat_only at offset {match.start()}")
                continue
            if key not in mapping:
                problems.append(f"unknown legacy keyword {key!r} at offset {match.start()}")
                passthrough.append(f"{key}={value}")
                continue
            bucket, field = mapping[key]
            buckets[bucket].append(f"{field}={value}")
        pieces = positional + passthrough
        params_fields = list(buckets["P"])
        # CN: `scan` 与 `pruning` 是**嵌套**对象，不能平铺进 `SolverParameters`。
        # EN: `scan` and `pruning` are **nested** objects; they must not be flattened into `SolverParameters`.
        if buckets["S"]:
            params_fields.append("scan=ScanParameters(" + ", ".join(buckets["S"]) + ")")
        if buckets["R"]:
            params_fields.append("pruning=PruningParameters(" + ", ".join(buckets["R"]) + ")")
        if params_fields:
            pieces.append("params=SolverParameters(" + ", ".join(params_fields) + ")")
        if buckets["O"]:
            name = "RelaxedObjective" if is_relaxed else "ExactObjective"
            pieces.append(f"objective={name}(" + ", ".join(buckets["O"]) + ")")
        out.append(", ".join(pieces) + ")")
        index = cursor
    return "".join(out), problems


def main() -> int:
    """CN: 入口：遍历仓库里所有调用点。EN: entry: walk every call site in the repository."""
    roots = [pathlib.Path("src/iot"), pathlib.Path("iot_experiments"), pathlib.Path("tests")]
    changed = 0
    for root in roots:
        for path in sorted(root.rglob("*.py")):
            if "legacy" in path.parts or path.name == "migrate_legacy_calls.py":
                continue
            text = path.read_text(encoding="utf-8")
            if "solve_inverse_ot(" not in text and "solve_inverse_ot_regret(" not in text:
                continue
            new_text, problems = rewrite(text)
            if problems:
                for problem in problems:
                    print(f"{path}: {problem}")
            if new_text != text:
                path.write_text(new_text, encoding="utf-8")
                changed += 1
                print(f"rewrote {path}")
    print(f"{changed} files rewritten")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
