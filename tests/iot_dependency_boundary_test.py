from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

PACKAGE_SRC = Path(__file__).resolve().parents[1] / "src" / "iot"

# CN: 只有这些模块允许 import hello_ot；核心算法模块必须与 HELLO-OT 解耦。
# EN: Only these modules may import hello_ot; the core algorithm must stay decoupled.
HELLO_OT_ADAPTERS = {"grounded.py"}


def test_only_adapters_import_hello_ot() -> None:
    offenders: list[str] = []
    for path in sorted(PACKAGE_SRC.glob("*.py")):
        if path.name in HELLO_OT_ADAPTERS:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            else:
                continue
            if any(name == "hello_ot" or name.startswith("hello_ot.") for name in names):
                offenders.append(f"{path.name}:{node.lineno}")
    assert offenders == []


def test_import_iot_does_not_load_hello_ot() -> None:
    code = "import iot, sys; assert 'hello_ot' not in sys.modules, 'hello_ot was eagerly imported'"
    completed = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
